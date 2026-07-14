# DeepSeek-V4 PD（Prefill-Decode）分离部署

## 概述

PD 分离将 LLM 推理的 Prefill 和 Decode 两阶段拆分到独立的计算实例上并行执行。
Prefill 实例负责处理长 prompt 输入（高计算量，大显存需求），
Decode 实例负责 token 生成（低计算量，低显存需求）。

DeepSeek-V4 使用 MLA（Multi-head Latent Attention）架构，KV cache 采用 pool-based 格式
（`BlockManagerPool`），包含三种 cache pool：
- **swa**: Sliding Window Attention cache
- **c4**: 4x 压缩 cache
- **c128**: 128 压缩 cache（或全精度）

## 架构

```
                  etcd（实例元数据 + 全局 KV cache 状态）
                    ↕
            xLLM Service（请求调度 + 实例管理）
           ↕                        ↕
    ┌── Prefill 实例 ──┐    ┌── Decode 实例 ──┐
    │  ep_size=16      │    │  ep_size=16      │
    │  dp_size=2       │    │  dp_size=2       │
    │  chunked_prefill │    │  prefix_cache    │
    │                  │    │                  │
    │  完成 KV cache   │───→│  接收 KV cache   │
    │  发送到 Decode   │    │  继续生成 token   │
    └──────────────────┘    └──────────────────┘
              ↕                        ↕
          MooncakeStore（KV cache 传输后端）
```

## 部署步骤

### 1. 前置条件

- etcd 已启动
- xLLM Service 已编译并安装
- `/etc/hccn.conf` 已正确配置（PD 分离依赖此文件）
- NPU 驱动和 CANN 已安装

### 2. 启动 etcd

```bash
./etcd --advertise-client-urls http://127.0.0.1:12389 \
       --listen-client-urls http://0.0.0.0:12389
```

### 3. 启动 xLLM Service

```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ./xllm_master_serving \
    --etcd_addr="127.0.0.1:12389" \
    --http_server_port 28888 \
    --rpc_server_port 28889 \
    --tokenizer_path=/path/to/DeepSeek-V4-Flash-w8a8-mtp/
```

### 4. 启动 Prefill 实例

```bash
bash scripts/deepseek_v4/run_disagg_pd.sh \
    --role=prefill \
    --model=/path/to/DeepSeek-V4-Flash-w8a8-mtp/ \
    --draft_model=/path/to/DeepSeek-V4-Flash-w8a8-mtp-mtp \
    --nnodes=8 \
    --start_device=0 \
    --start_port=18004 \
    --master_addr="127.0.0.1:8773" \
    --etcd_addr="127.0.0.1:12389" \
    --ep_size=16 \
    --dp_size=2
```

### 5. 启动 Decode 实例

```bash
bash scripts/deepseek_v4/run_disagg_pd.sh \
    --role=decode \
    --model=/path/to/DeepSeek-V4-Flash-w8a8-mtp/ \
    --draft_model=/path/to/DeepSeek-V4-Flash-w8a8-mtp-mtp \
    --nnodes=8 \
    --start_device=8 \
    --start_port=19004 \
    --master_addr="127.0.0.1:8777" \
    --etcd_addr="127.0.0.1:12389" \
    --ep_size=16 \
    --dp_size=2
```

## 关键参数说明

| 参数 | Prefill | Decode | 说明 |
|------|---------|--------|------|
| `enable_disagg_pd` | `true` | `true` | 启用 PD 分离 |
| `instance_role` | `PREFILL` | `DECODE` | 实例角色 |
| `enable_chunked_prefill` | `true` | `false` | Prefill 需要，Decode 不需要 |
| `enable_prefix_cache` | `false` | `true` | Decode 必须开启（best_of_n 场景需求）|
| `enable_graph` | `true` | `true` | 图模式加速 |
| `ep_size` | 16 | 16 | Expert Parallel 度 |
| `dp_size` | 2 | 2 | Data Parallel 度 |

## DeepSeek-V4 特定说明

### MLA KV cache 格式

DeepSeek-V4 使用 MLA MLA 压缩 KV cache，格式与标准 KV cache 不同：

- `kv_cache_shape` 为 `[swa_count, c4_count, c128_count]` 的三维 pool
- `BlockManagerPool` 管理这三种 pool 的生命周期
- PD 传输时，`KVCacheTransfer` 以 block 为单位进行推送/拉取，对 pool 内部格式透明

### PD 传输注意事项

1. **DP 一致性**：Prefill 和 Decode 的 `dp_size` 必须一致，否则 block 映射会错位
2. **EP 一致性**：两者的 `ep_size` 必须一致
3. **Model ID**：Prefill 和 Decode 必须加载相同的模型权重
4. **NVLink vs HCCL**：当前 NPU 使用 HCCL 通信，传输后端的 `mooncake` 负责跨节点 KV 传输

### 性能预期

- **TTFT**: Prefill 独立部署后不受 decode 干扰，可降低 30-50%
- **TPOT**: Decode 独立部署后不受 prefill 干扰，可降低 10-20%
- **吞吐**: Prefill 和 Decode 各自使用更优的 batch 配置，整体吞吐可提升 15-30%
