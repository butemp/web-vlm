#!/bin/bash
# 一键推送脚本 —— push.sh
# 用法: ./push.sh [commit message]

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

# commit message
MSG="${1:-update: $(date '+%Y-%m-%d %H:%M')}"

echo "📦 正在检查文件变更..."
git add -A

if git diff --cached --quiet; then
  echo "✅ 没有新的变更，无需提交"
else
  git commit -m "$MSG"
  echo "✅ 已提交: $MSG"
fi

echo "🚀 正在推送到 $REMOTE ..."
git push -u origin "$BRANCH"

echo ""
echo "✅ 推送完成！"
echo "🔗 https://github.com/butemp/web-vlm"
