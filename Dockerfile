# Use an official NVIDIA CUDA runtime image as a parent image
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Set environment variables for NVIDIA container
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set environment variables to make package installation non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN apt-get update && \
    apt-get install -y python3 python3-venv python3-pip python3-dev \
            python3-distutils python3-setuptools-whl && \
    apt-get clean

WORKDIR /app

# Install uv using pip
RUN python3 -m pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy project files (v1's app/ no longer exists: the API lives in v2/, the
# inference assets in inference/)
COPY ./.python-version /app/.python-version
COPY ./pyproject.toml /app/pyproject.toml
COPY ./uv.lock /app/uv.lock
COPY ./alembic.ini /app/alembic.ini
COPY ./v2 /app/v2
COPY ./inference /app/inference

RUN uv sync

EXPOSE 8000

# Apply migrations, then serve the API. The inference worker runs as its own
# service (see docker-compose.yml) so model reloads never restart the API.
CMD ["sh", "-c", "uv run alembic upgrade head && uv run fastapi run v2/main.py --host 0.0.0.0 --port 8000 --workers 1"]
