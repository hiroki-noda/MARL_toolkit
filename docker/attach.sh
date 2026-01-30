#!/bin/bash

# コンテナ名
CONTAINER_NAME="rl-container"

# コンテナが存在するか確認
if [ ! "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Error: Container ${CONTAINER_NAME} does not exist"
    echo "Please run ./run.sh first to create the container"
    exit 1
fi

# コンテナが実行中でない場合は起動
if [ ! "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "Starting container: ${CONTAINER_NAME}"
    docker start ${CONTAINER_NAME}
fi

# コンテナに接続
echo "Attaching to container: ${CONTAINER_NAME}"
docker exec -it ${CONTAINER_NAME} /bin/bash -l
