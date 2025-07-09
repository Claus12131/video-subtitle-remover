FROM python:3.12-slim-bookworm

RUN --mount=type=cache,target=/root/.cache,sharing=private \
    apt update && \
    apt install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && \
    true


COPY requirements.txt .


ADD . /vsr
ARG CUDA_VERSION=11.8
ARG USE_DIRECTML=0

# 如果是 CUDA 版本，执行 CUDA 特定设置
RUN --mount=type=cache,target=/root/.cache,sharing=private \
    if [ "${USE_DIRECTML:-0}" != "1" ]; then \
        pip install paddlepaddle==3.0 && \
        pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu$(echo ${CUDA_VERSION} | tr -d '.') && \
        pip install -r /vsr/requirements.txt; \
    fi

# 如果是 DirectML 版本，执行 DirectML 特定设置
RUN --mount=type=cache,target=/root/.cache,sharing=private \
    if [ "${USE_DIRECTML:-0}" = "1" ]; then \
        pip install paddlepaddle==3.0 && \
        pip install torch_directml==0.2.5.dev240914 && \
        pip install -r /vsr/requirements.txt; \
    fi

#暴露服务端口
EXPOSE 5000

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib/
WORKDIR /vsr
CMD ["python", "web.py"]
