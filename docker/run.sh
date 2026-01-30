#!/bin/bash

# スクリプトのあるディレクトリに移動
cd "$(dirname "$0")"

# イメージ名とタグ
IMAGE_NAME="rl-mujoco-ros2"
IMAGE_TAG="latest"
CONTAINER_NAME="rl-container"

# ワークスペースのパス（dockerディレクトリの親ディレクトリ）
WORKSPACE_PATH="$(cd .. && pwd)"

# X11フォワーディングの許可
xhost +local:docker > /dev/null 2>&1

# 既存のコンテナが実行中か確認
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Attaching to container..."
    docker exec -it ${CONTAINER_NAME} /bin/bash -l
    exit 0
fi

# 停止中のコンテナが存在するか確認
if [ "$(docker ps -aq -f status=exited -f name=${CONTAINER_NAME})" ]; then
    echo "Restarting existing container: ${CONTAINER_NAME}"
    docker start ${CONTAINER_NAME}
    docker exec -it ${CONTAINER_NAME} /bin/bash -l
    exit 0
fi

echo "Creating new Docker container: ${CONTAINER_NAME}"
echo "Workspace: ${WORKSPACE_PATH}"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"

# コンテナを起動（--rmを削除してコンテナを残す）
docker run -it \
    --name ${CONTAINER_NAME} \
    --gpus all \
    --net=host \
    --ipc=host \
    --privileged \
    -e DISPLAY=${DISPLAY} \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=${XAUTHORITY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${WORKSPACE_PATH}:/workspace \
    -v ${HOME}/.Xauthority:/home/hiroki/.Xauthority:rw \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    /bin/bash