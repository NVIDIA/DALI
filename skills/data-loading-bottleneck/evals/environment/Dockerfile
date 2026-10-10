FROM nvidia/cuda:13.3.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl git gnupg python3 python3-venv && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
        -o /tmp/nvidia-devtools.pub && \
    gpg --dearmor --output /usr/share/keyrings/nvidia-devtools-keyring.gpg /tmp/nvidia-devtools.pub && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/ /" \
        > /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nsight-systems-cli && \
    rm -rf /tmp/nvidia-devtools.pub /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    python -m pip install --no-cache-dir 'torch~=2.13' 'torchvision~=0.28' --index-url https://download.pytorch.org/whl/cu130

RUN python -m pip install --no-cache-dir 'nvidia-dali-cuda130~=2.3' 'nvtx~=0.2' timm==1.0.28

RUN mkdir -p /opt/eval/workloads && \
    git clone --branch v1.0.28 --depth 1 \
        https://github.com/huggingface/pytorch-image-models.git \
        /opt/eval/workloads/pytorch-image-models && \
    git clone --branch v0.28.0 --depth 1 \
        https://github.com/pytorch/vision.git \
        /opt/eval/workloads/vision && \
    chown -R 1000:1000 /opt/eval/workloads

RUN mkdir -p /opt/eval/data && \
    curl -fsSL https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz \
        -o /tmp/imagenette2.tgz && \
    tar -xzf /tmp/imagenette2.tgz -C /opt/eval/data && \
    rm -f /tmp/imagenette2.tgz

RUN command -v nsys && \
    python -c "import nvtx, timm, torch, torchvision; from nvidia.dali.plugin.pytorch.loader_evaluator import LoaderEvaluator; assert torch.version.cuda"

WORKDIR /workspace
