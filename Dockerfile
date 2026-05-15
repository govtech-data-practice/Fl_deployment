FROM python:3.12-slim

# Pin torch+cu124 for CUDA 12.8 host driver compatibility
RUN pip install --no-cache-dir "numpy<2" "flwr[simulation]>=1.13" scikit-learn pandas pillow pyyaml && \
    pip install --no-cache-dir torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

WORKDIR /app
COPY fl_common/ /app/fl_common/
COPY models/ /app/models/
COPY tasks/ /app/tasks/
COPY privacy/ /app/privacy/
COPY experiments/ /app/experiments/
COPY scenarios/ /app/scenarios/
COPY run_tests.py run_all.py run_ec2.py /app/

# List available models and scenarios at build time
RUN echo "Models: bilstm, mlp, densenet, mistral, vfl_mlp, split_bilstm" && \
    echo "Scenarios:" && ls /app/scenarios/*.yaml

ENV PYTHONUNBUFFERED=1
ENV SYNTHETIC=0
