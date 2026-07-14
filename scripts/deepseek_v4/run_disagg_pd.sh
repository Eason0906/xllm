#!/bin/bash
# ==============================================================================
# DeepSeek-V4 PD（Prefill-Decode）分离部署启动脚本
# 分支: br_pd_syj_0714
#
# 启动顺序：
#   1. etcd
#   2. xLLM Service（xllm_master_serving）
#   3. Prefill 实例（本脚本 --role=prefill）
#   4. Decode 实例（本脚本 --role=decode）
#
# DeepSeek-V4 特有配置说明：
#   - BlockManagerPool: [swa, c4_ratio, c128_ratio] 三个 pool
#   - KV cache 传输以 block 为单位，对 pool 格式透明
#   - Prefill 实例需要 enable_chunked_prefill=true（DSV4 默认）
#   - Decode 实例需要 enable_prefix_cache=true（best_of_n 场景必需）
#
# 用法：
#   # Prefill 启动
#   bash run_disagg_pd.sh \
#       --role=prefill \
#       --model=/path/to/DeepSeek-V4-Flash-w8a8-mtp/ \
#       --nnodes=8 --start_device=0 --start_port=18004
#
#   # Decode 启动（不同窗口）
#   bash run_disagg_pd.sh \
#       --role=decode \
#       --model=/path/to/DeepSeek-V4-Flash-w8a8-mtp/ \
#       --nnodes=8 --start_device=8 --start_port=19004
# ==============================================================================

set -euo pipefail

# ==============================================================================
# 默认参数
# ==============================================================================
ROLE="prefill"
MODEL_PATH="/export/home/models/DeepSeek-V4-Flash-w8a8-mtp/"
DRAFT_MODEL_PATH="/export/home/models/DeepSeek-V4-Flash-w8a8-mtp-mtp"
XLLM_BINARY="../build/xllm/core/server/xllm"
XLLM_SERVICE_ADDR="127.0.0.1:28889"
ETCD_ADDR="127.0.0.1:12389"
MASTER_NODE_ADDR="127.0.0.1:8773"

NNODES=8
START_DEVICE=0
START_PORT=18004
TRANSFER_START_PORT=26000
DISAGG_PD_PORT=7777

LOG_DIR="log"
MAX_MEMORY_UTILIZATION=0.80
MAX_TOKENS_PER_BATCH=51200
MAX_TOKENS_PER_CHUNK=4096
MAX_SEQS_PER_BATCH=32
BLOCK_SIZE=128
COMM_BACKEND="hccl"
EP_SIZE=16
DP_SIZE=2
NUM_SPECULATIVE_TOKENS=0

# ==============================================================================
# 解析命令行参数
# ==============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --role=*)                ROLE="${1#*=}" ;;
    --model=*)               MODEL_PATH="${1#*=}" ;;
    --draft_model=*)         DRAFT_MODEL_PATH="${1#*=}" ;;
    --binary=*)              XLLM_BINARY="${1#*=}" ;;
    --etcd_addr=*)           ETCD_ADDR="${1#*=}" ;;
    --master_addr=*)         MASTER_NODE_ADDR="${1#*=}" ;;
    --nnodes=*)              NNODES="${1#*=}" ;;
    --start_device=*)        START_DEVICE="${1#*=}" ;;
    --start_port=*)          START_PORT="${1#*=}" ;;
    --transfer_port=*)       TRANSFER_START_PORT="${1#*=}" ;;
    --disagg_port=*)         DISAGG_PD_PORT="${1#*=}" ;;
    --log_dir=*)             LOG_DIR="${1#*=}" ;;
    --ep_size=*)             EP_SIZE="${1#*=}" ;;
    --dp_size=*)             DP_SIZE="${1#*=}" ;;
    --spec_tokens=*)         NUM_SPECULATIVE_TOKENS="${1#*=}" ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
  shift
done

# ==============================================================================
# 角色相关参数
# ==============================================================================
case "$ROLE" in
  prefill)
    INSTANCE_ROLE="PREFILL"
    # Prefill 实例使用 chunked prefill 分片处理长 prompt
    CHUNKED_PREFILL=true
    # Prefill 不需要 prefix cache（先塞满再发到 decode）
    PREFIX_CACHE=false
    # Prefill 的 master 地址
    MASTER_ADDR="$MASTER_NODE_ADDR"
    # PD 通信端口
    DISAGG_PORT=$DISAGG_PD_PORT
    ;;
  decode)
    INSTANCE_ROLE="DECODE"
    # Decode 实例不需要 chunked prefill（只做 token 生成）
    CHUNKED_PREFILL=false
    # Decode 必须开 prefix cache（best_of_n 场景：扩展出的候选序列
    # 通过 prefix cache 复用第一条序列的 prompt KV）
    PREFIX_CACHE=true
    # Decode 实例的 master 需要不同端口（避免与 prefill 冲突）
    MASTER_ADDR="${MASTER_NODE_ADDR%:*}:$(( ${MASTER_NODE_ADDR##*:} + 100 ))"
    # PD 通信端口错开
    DISAGG_PORT=$((DISAGG_PD_PORT + 10))
    ;;
  *)
    echo "Error: --role must be 'prefill' or 'decode'"
    exit 1
    ;;
esac

# ==============================================================================
# 环境准备
# ==============================================================================
mkdir -p "$LOG_DIR"

# HCCL 通信基础端口
export HCCL_IF_BASE_PORT=43432
# 关闭 coredump
ulimit -c 0

echo "=========================================="
echo "  DeepSeek-V4 PD 分离部署"
echo "  分支: br_pd_syj_0714"
echo "=========================================="
echo "  Role:            $INSTANCE_ROLE"
echo "  NNODES:          $NNODES"
echo "  EP_SIZE:         $EP_SIZE"
echo "  DP_SIZE:         $DP_SIZE"
echo "  Start Device:    $START_DEVICE"
echo "  Start Port:      $START_PORT"
echo "  Model:           $MODEL_PATH"
echo "  Master:          $MASTER_ADDR"
echo "  etcd:            $ETCD_ADDR"
echo "=========================================="

# ==============================================================================
# 启动实例
# ==============================================================================
for (( i=0; i<NNODES; i++ )); do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  TRANSFER_PORT=$((TRANSFER_START_PORT + i))
  LOG_FILE="$LOG_DIR/node_${ROLE}_${i}.log"

  echo "  Starting ${INSTANCE_ROLE}[$i]: npu:${DEVICE} port=${PORT}"

  # 基础参数（所有实例通用）
  ARGS=(
    --model "$MODEL_PATH"
    --devices "npu:$DEVICE"
    --port "$PORT"
    --master_node_addr "$MASTER_ADDR"
    --nnodes "$NNODES"
    --node_rank "$i"
    --max_memory_utilization "$MAX_MEMORY_UTILIZATION"
    --max_tokens_per_batch "$MAX_TOKENS_PER_BATCH"
    --max_seqs_per_batch "$MAX_SEQS_PER_BATCH"
    --block_size "$BLOCK_SIZE"
    --communication_backend "$COMM_BACKEND"
    --ep_size "$EP_SIZE"
    --dp_size "$DP_SIZE"
    --enable_graph true
    --enable_disagg_pd true
    --instance_role "$INSTANCE_ROLE"
    --etcd_addr "$ETCD_ADDR"
    --transfer_listen_port "$TRANSFER_PORT"
    --disagg_pd_port "$DISAGG_PORT"
  )

  # 角色特有参数
  case "$ROLE" in
    prefill)
      ARGS+=(
        --enable_chunked_prefill true
        --enable_prefix_cache false
        --max_tokens_per_chunk_for_prefill "$MAX_TOKENS_PER_CHUNK"
      )
      ;;
    decode)
      ARGS+=(
        --enable_chunked_prefill false
        --enable_prefix_cache true
      )
      ;;
  esac

  # MTP 投机推理
  if [ "$NUM_SPECULATIVE_TOKENS" -gt 0 ] && [ -n "$DRAFT_MODEL_PATH" ]; then
    ARGS+=(
      --draft_model "$DRAFT_MODEL_PATH"
      --draft_devices "npu:$DEVICE"
      --num_speculative_tokens "$NUM_SPECULATIVE_TOKENS"
    )
  fi

  # 后台启动
  nohup "$XLLM_BINARY" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
  # 错峰启动，避免 HCCL 资源抢占
  sleep 3
done

echo ""
echo "=========================================="
echo "  所有 ${INSTANCE_ROLE} 实例已启动"
echo "  日志目录: $LOG_DIR"
echo "  查看日志: tail -f $LOG_DIR/node_${ROLE}_0.log"
echo ""
echo "  验证命令:"
echo "    ps aux | grep xllm | grep ${INSTANCE_ROLE}"
echo "    tail -f ${LOG_DIR}/node_${ROLE}_0.log | grep 'Instance info'"
echo "=========================================="
