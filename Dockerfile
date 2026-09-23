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

# Copy project files (the API package `app/` and the inference assets `inference/`)
COPY ./.python-version /app/.python-version
COPY ./pyproject.toml /app/pyproject.toml
COPY ./uv.lock /app/uv.lock
COPY ./alembic.ini /app/alembic.ini
COPY ./app /app/app
COPY ./inference /app/inference
COPY ./entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN uv sync --no-dev

# Runtime environment for the entrypoint *and* for `docker exec` / `docker
# compose run`, which do not go through it: the runtime uid has no passwd entry
# (so HOME would be unresolvable) and uv must never try to re-sync the
# root-owned venv.
ENV HOME=/tmp \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_NO_SYNC=1

EXPOSE 8000

# Ownership repair, privilege drop (ZLABEL_UID/ZLABEL_GID) and `alembic upgrade
# head` (ZLABEL_MIGRATE) live in the entrypoint; the inference worker service
# overrides CMD (see docker-compose.yml).
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/app/.venv/bin/fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
