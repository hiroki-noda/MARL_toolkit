#!/bin/bash

echo "=== GPU環境診断スクリプト ==="
echo ""

echo "[1/5] Dockerのバージョン確認"
docker --version
echo ""

echo "[2/5] nvidia-container-toolkitの確認"
if command -v nvidia-ctk &> /dev/null; then
    echo "✓ nvidia-container-toolkit がインストールされています"
    nvidia-ctk --version
else
    echo "✗ nvidia-container-toolkit がインストールされていません"
    echo "インストールコマンド:"
    echo "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y nvidia-container-toolkit"
    echo "  sudo systemctl restart docker"
fi
echo ""

echo "[3/5] Docker デーモンの設定確認"
if [ -f /etc/docker/daemon.json ]; then
    echo "✓ daemon.json が存在します"
    cat /etc/docker/daemon.json
else
    echo "✗ daemon.json が存在しません"
    echo "作成コマンド:"
    echo "  sudo tee /etc/docker/daemon.json <<EOF"
    echo '  {'
    echo '    "runtimes": {'
    echo '      "nvidia": {'
    echo '        "path": "nvidia-container-runtime",'
    echo '        "runtimeArgs": []'
    echo '      }'
    echo '    }'
    echo '  }'
    echo "  EOF"
    echo "  sudo systemctl restart docker"
fi
echo ""

echo "[4/5] ホスト側のGPU確認"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "✗ ホスト側でnvidia-smiが使用できません"
fi
echo ""

echo "[5/5] テストコンテナでGPU確認"
echo "実行中: docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi"
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi 2>&1
if [ $? -eq 0 ]; then
    echo "✓ DockerコンテナからGPUにアクセスできます"
else
    echo "✗ DockerコンテナからGPUにアクセスできません"
    echo ""
    echo "対処方法:"
    echo "1. nvidia-container-toolkitをインストール"
    echo "2. /etc/docker/daemon.jsonを設定"
    echo "3. sudo systemctl restart docker"
    echo "4. WSL2の場合: wsl --shutdown してWSL2を再起動"
fi
echo ""

echo "=== 診断完了 ==="
