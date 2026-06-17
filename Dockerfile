FROM python:3.12-slim

# All dependencies pinned to exact versions (verified 2026-05-25)
# torch+cu124 requires CUDA 12.4+ host driver (tested: NVIDIA 595.71.05)
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    "flwr[simulation]==1.30.0" \
    scikit-learn==1.8.0 \
    pandas==3.0.3 \
    pillow==12.2.0 \
    PyYAML==6.0.3 \
    tenseal==0.3.16 \
    transformers \
    peft \
    accelerate \
    bitsandbytes \
    && pip install --no-cache-dir \
    torch==2.5.1+cu124 \
    torchvision==0.20.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

WORKDIR /app
COPY fl_common/ /app/fl_common/
COPY models/ /app/models/
COPY tasks/ /app/tasks/
COPY privacy/ /app/privacy/
COPY experiments/ /app/experiments/
COPY scenarios/ /app/scenarios/
COPY secure_inference/ /app/secure_inference/
COPY serverapp/ /app/serverapp/
COPY clientapp/ /app/clientapp/
COPY secagg/ /app/secagg/
COPY psi/ /app/psi/
COPY fl_pets/ /app/fl_pets/
COPY tools/ /app/tools/
COPY runners/ /app/runners/
COPY tests/ /app/tests/

# List available models and scenarios at build time
RUN echo "Models: bilstm, mlp, densenet, mistral, vfl_mlp, split_bilstm" && \
    echo "Scenarios:" && ls /app/scenarios/*.yaml

ENV PYTHONUNBUFFERED=1
ENV SYNTHETIC=0
