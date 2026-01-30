FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV TZ=Asia/Tokyo

# CUDA/cuDNN環境変数（WSL2 GPU対応 - RTX 3080, Driver 566.14）
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# タイムゾーン設定
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# APTのタイムアウトとリトライ設定
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "600";' >> /etc/apt/apt.conf.d/99timeout && \
    echo 'Acquire::https::Timeout "600";' >> /etc/apt/apt.conf.d/99timeout && \
    echo 'Acquire::ftp::Timeout "600";' >> /etc/apt/apt.conf.d/99timeout

# システムパッケージの更新
RUN apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt update --fix-missing

RUN apt upgrade -y

# 基本パッケージのインストール
RUN apt install -y -o APT::Acquire::Retries=3 sudo wget curl git emacs build-essential \
    ca-certificates libssl-dev libffi-dev cmake zip unzip software-properties-common \
    lsb-release gnupg2

# Python関連
RUN apt install -y -o APT::Acquire::Retries=3 \
    python3 python3-venv python3-dev python3-pip

# Ubuntu 24.04 の externally-managed-environment を解除して pip3 を許可
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# MuJoCo用依存パッケージ
RUN apt install -y -o APT::Acquire::Retries=3 \
    libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev \
    mesa-common-dev libglu1-mesa-dev freeglut3-dev mesa-utils \
    libglew-dev libegl1-mesa-dev libgles2-mesa-dev libglfw3-dev \
    libosmesa6-dev patchelf

# キャッシュのクリア
RUN rm -rf /var/lib/apt/lists/*

# ROS2のGPGキー追加
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    # -o /usr/share/keyrings/ros-archive-keyring.gpg

# ROS2リポジトリの追加
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    # http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    # | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# ROS2のインストール（Ubuntu 22.04にはROS2 humble）
# RUN apt update --fix-missing && \
    # apt install -y -o APT::Acquire::Retries=3 ros-humble-ros-base && \
    # apt install -y -o APT::Acquire::Retries=3 ros-humble-desktop && \
    # apt install -y -o APT::Acquire::Retries=3 python3-colcon-common-extensions python3-rosdep && \
    # rm -rf /var/lib/apt/lists/*

# rosdepの初期化
# RUN rosdep init && rosdep update

# ユーザー作成（sudoをパスワードなしで実行可能）
# 既にUID/GIDが存在していても落ちない安全な定義
ARG USER_ID=1000
ARG GROUP_ID=1000

# グループ：GIDが存在すればそのグループ名をhirokiに変更、なければ作成
RUN if getent group ${GROUP_ID} > /dev/null; then \
            groupmod -n hiroki "$(getent group ${GROUP_ID} | cut -d: -f1)"; \
        else \
            groupadd -g ${GROUP_ID} hiroki; \
        fi

# ユーザー：UIDが既に存在すればそのユーザー名をhirokiに変更、なければ作成
RUN if getent passwd ${USER_ID} > /dev/null; then \
            usermod -l hiroki "$(getent passwd ${USER_ID} | cut -d: -f1)" && \
            usermod -d /home/hiroki -m hiroki && \
            usermod -g ${GROUP_ID} hiroki; \
        elif ! id -u hiroki > /dev/null 2>&1; then \
            useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash hiroki; \
        fi

RUN echo "hiroki ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 作業ディレクトリ
RUN mkdir -p /workspace && chown hiroki:hiroki /workspace

# PyTorchとMuJoCo関連ライブラリのインストール（WSL2 GPU対応 - RTX 3080, Driver 566.14）
# CUDA 12.1対応のPyTorchをインストール（GPU版を明示）
RUN pip3 install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 強化学習関連ライブラリのインストール
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    gymnasium==0.29.1 \
    mujoco==3.1.1 \
    gymnasium-robotics \
    tensorboard \
    matplotlib \
    pyyaml

# ユーザー切り替え
USER hiroki
WORKDIR /workspace

# ディレクトリ作成
RUN mkdir -p /workspace/checkpoints /workspace/logs

# CUDA環境変数をbashrcに追加（セッションごとに確実に読み込む）
RUN echo "" >> ~/.bashrc && \
    echo "# CUDA Environment (CUDA 12.6, RTX 3080, Driver 566.14)" >> ~/.bashrc && \
    echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc && \
    echo "export PATH=/usr/local/cuda/bin:\${PATH}" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:\${LD_LIBRARY_PATH}" >> ~/.bashrc

# ROS2セットアップ（コメントアウト：不要な場合）
# RUN echo "" >> ~/.bashrc && \
#     echo "# ROS2 Setup" >> ~/.bashrc && \
#     echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
#     echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc

# 環境変数
ENV DISPLAY=:0
ENV LIBGL_ALWAYS_INDIRECT=0
ENV MUJOCO_GL=egl

# PyTorch/CUDA最適化設定（WSL2 + RTX 3080用）
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV FORCE_CUDA=1

CMD ["/bin/bash"]
