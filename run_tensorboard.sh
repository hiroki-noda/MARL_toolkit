#!/usr/bin/env bash
# Simple helper script to launch TensorBoard.
# Usage:
#   ./run_tensorboard.sh [LOGDIR]
#
# LOGDIR: TensorBoard のログディレクトリ（デフォルト: ./logs）
# PORT:   使用するポート番号（環境変数 PORT で指定、デフォルト: 6006）

set -e

LOGDIR=${1:-"logs"}
PORT=${PORT:-6006}
HOST=${HOST:-"0.0.0.0"}

echo "[INFO] Starting TensorBoard"
echo "       logdir = ${LOGDIR}"
echo "       port   = ${PORT}"
echo "       host   = ${HOST}"

# Check that tensorboard is available
if ! command -v tensorboard >/dev/null 2>&1; then
  echo "[ERROR] tensorboard command not found. Install it via:"
  echo "       pip install tensorboard"
  exit 1
fi

exec tensorboard --logdir "${LOGDIR}" --port "${PORT}" --host "${HOST}"