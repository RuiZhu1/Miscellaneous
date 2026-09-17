#!/bin/bash
set -e

ENV_NAME="quantumcomputing"
PYTHON_VERSION="3.13"
MINICONDA_DIR="$HOME/miniconda3"

echo "=== 1. 检查并安装 Miniconda ==="
if [ ! -d "$MINICONDA_DIR" ]; then
    echo "正在下载并安装 Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
    rm /tmp/miniconda.sh
else
    echo "Miniconda 已安装，跳过下载。"
fi

echo "=== 2. 初始化 Conda 环境 ==="
eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"
"$MINICONDA_DIR/bin/conda" init bash > /dev/null 2>&1

echo "=== 2.1 同意 Anaconda 服务条款 ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "=== 3. 创建 Python $PYTHON_VERSION 环境: $ENV_NAME ==="
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "环境 $ENV_NAME 已存在。"
else
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

echo "=== 4. 环境配置完成！即将在激活的环境中启动新 Shell ==="
exec bash --rcfile <(echo "source ~/.bashrc; conda activate $ENV_NAME")