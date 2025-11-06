# Use an official NVIDIA CUDA runtime image as a parent image
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Set environment variables for NVIDIA container
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set environment variables to make package installation non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# Install Python 3.13 from deadsnakes PPA
RUN apt-get clean && apt-get update && \
    apt-get install -y python3 python3-venv python3-pip python3-dev \
            python3-distutils python3-setuptools-whl && \
    apt-get clean

WORKDIR /app

# Install uv using pip
RUN python3 -m pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy project files
COPY ./app /app/app
COPY ./assets /app/assets
COPY ./.python-version /app/.python-version
COPY ./pyproject.toml /app/pyproject.toml
COPY ./uv.lock /app/uv.lock

# Sync dependencies using uv
RUN uv sync

EXPOSE 8000

# Define the command to run the application
CMD ["uv", "run", "fastapi", "run", "app/app.py", "--host", "0.0.0.0", "--port", "8000"]
