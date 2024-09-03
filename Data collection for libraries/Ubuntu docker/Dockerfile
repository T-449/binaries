# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set non-interactive installation to avoid tzdata hanging the build
ARG DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y \
    git \
    gcc \
    build-essential \
    openssl \
    libssl-dev \
    linux-tools-common \
    linux-tools-generic \
    linux-tools-$(uname -r)

# Replace this with your actual Git repository URL
ARG GIT_REPO=https://github.com/T-449/binaries

# Clone the complete repository
RUN git clone $GIT_REPO
WORKDIR /binaries
RUN chmod +x ./run.sh
CMD ["./run.sh"]
