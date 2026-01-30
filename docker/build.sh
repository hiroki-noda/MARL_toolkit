#!/bin/bash

# スクリプトのあるディレクトリに移動
cd "$(dirname "$0")"

# ユーザーIDとグループIDを取得
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# イメージ名とタグ
IMAGE_NAME="rl-mujoco-ros2"
IMAGE_TAG="latest"

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "USER_ID: ${USER_ID}"
echo "GROUP_ID: ${GROUP_ID}"

# Dockerイメージをビルド
docker build \
    --build-arg USER_ID=${USER_ID} \
    --build-arg GROUP_ID=${GROUP_ID} \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    -f Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"
else
    echo "✗ Failed to build Docker image"
    exit 1
fi
