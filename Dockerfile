FROM nvidia/cuda:12.1.0-base-ubuntu22.04


# Update package lists
RUN apt-get update

# Install essential tools and dependencies
RUN apt-get install -y --no-install-recommends \
    python3-pip \
    libglib2.0-dev \
    apt-utils \
    wget

# Install CUDA Toolkit and cuDNN
RUN apt-get install -y nvidia-cuda-toolkit

# Add the NVIDIA package repository
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update

# Install cuDNN
RUN apt-get install -y libcudnn8 libcudnn8-dev

# Fix broken installs if any
RUN apt-get install -f

# Install the NVIDIA Container Toolkit
RUN apt-get install -y nvidia-container-toolkit

# Copy Python dependencies
COPY requirements.txt .

RUN pip install torch==2.2.0+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Install the Hugging Face CLI
RUN pip install huggingface_hub

# Set environment variable for Hugging Face token
ENV HUGGINGFACE_TOKEN="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"

# Login to Hugging Face CLI
RUN huggingface-cli login --token $HUGGINGFACE_TOKEN

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy your application code
COPY . /

# Set environment variable for library path
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Disable cuDNN
RUN echo "import torch; torch.backends.cudnn.enabled = False" > disable_cudnn.py
RUN python3 disable_cudnn.py && rm disable_cudnn.py

# Run your application
CMD ["python3", "app.py"]
