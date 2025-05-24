FROM robotlocomotion/drake:jammy

# Explicitly set current user to root
USER root

# Add user named "user"
ENV USER_NAME=user
RUN useradd -m "$USER_NAME"
RUN usermod -aG sudo "$USER_NAME"
RUN passwd -d "$USER_NAME"

# Create a directory for documents
# RUN mkdir "/documents"

# 1. Update apt repositories
# 2. Install package to use pip 
# 3. Install package to use virtual environments
# 4. Install package to set default Python to Python 3 (which is technically optional)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y python3-pip python3-venv python-is-python3 xvfb sudo ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools
USER user

ENV PATH="/home/user/.local/bin:$PATH"
EXPOSE 8888