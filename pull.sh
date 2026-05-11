#!/bin/bash
# 一键拉取脚本 —— pull.sh
# 用法: ./pull.sh

set -e

REMOTE="git@github.com:butemp/web-vlm.git"
BRANCH="wsl_branch"

# ── 代理配置（按需修改端口，不用代理则留空）──
PROXY_PORT=""   # 常见: 7890 / 1080 / 8080，留空则不走代理

# ─────────────────────────────────────────────
cd "$(dirname "$0")"

# 设置代理
if [ -n "$PROXY_PORT" ]; then
  export https_proxy="http://127.0.0.1:$PROXY_PORT"
  export http_proxy="http://127.0.0.1:$PROXY_PORT"
  export GIT_SSH_COMMAND="ssh -o ProxyCommand='nc -x 127.0.0.1:$PROXY_PORT %h %p'"
  echo "🔧 已启用代理: 127.0.0.1:$PROXY_PORT"
fi

# 确保 remote 是 SSH 地址
git remote set-url origin "$REMOTE" 2>/dev/null || git remote add origin "$REMOTE"

echo "🔄 正在从 $REMOTE 拉取 $BRANCH ..."
git pull origin "$BRANCH"

echo ""
echo "✅ 拉取完成！"
echo "🔗 https://github.com/butemp/web-vlm"
