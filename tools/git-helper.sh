#!/bin/bash
# xLLM Git Helper
# 在 xLLM 仓库中执行 git 操作
# 用法: bash tools/git-helper.sh <command> [args...]

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR" || { echo "❌ 无法进入仓库目录 $REPO_DIR"; exit 1; }

case "$1" in
  status)  git status ;;
  diff)    git diff ;;
  add)     shift; git add "$@" && echo "✅ 已暂存" ;;
  commit)  shift; git commit -m "$*" && echo "✅ 已提交" ;;
  push)    shift; git push "$@" && echo "✅ 已推送" ;;
  pull)    git pull && echo "✅ 已拉取" ;;
  log)     git log --oneline -10 ;;
  *)
    echo "用法: $0 {status|diff|add|commit|push|pull|log} [args...]"
    exit 1
    ;;
esac
