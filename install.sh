#!/bin/bash
sudo apt update

sudo apt install -y -o APT::Acquire::Retries=3 wget curl git emacs build-essential \
    ca-certificates libssl-dev libffi-dev cmake zip unzip software-properties-common \
    lsb-release gnupg2

sudo apt install -y -o APT::Acquire::Retries=3 \
    python3 python3-venv python3-dev python3-pip

sudo apt install -y -o APT::Acquire::Retries=3 \
    libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev \
    mesa-common-dev libglu1-mesa-dev freeglut3-dev mesa-utils \0
    libglew-dev libegl1-mesa-dev libgles2-mesa-dev libglfw3-dev \
    libosmesa6-dev patchelf

pip3 install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip3 install --no-cache-dir \
    numpy==1.24.3 \
    gymnasium==0.29.1 \
    mujoco==3.1.1 \
    gymnasium-robotics \
    tensorboard \
    matplotlib \
    pyyaml

# # ROS2リポジトリの追加
# curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
#     -o /usr/share/keyrings/ros-archive-keyring.gpg

# echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
#     http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
#     | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# # ROS2のインストール（Ubuntu 22.04にはROS2 humble）
# sudo apt update --fix-missing && \
#     apt install -y -o APT::Acquire::Retries=5 ros-humble-desktop && \
#     apt install -y -o APT::Acquire::Retries=3 python3-colcon-common-extensions python3-rosdep && \
#     rm -rf /var/lib/apt/lists/*

# # rosdepの初期化
# sudo rosdep init && rosdep update