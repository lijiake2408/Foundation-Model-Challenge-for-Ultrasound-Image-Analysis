# Docker Submission Guide

> ⭐ **IMPORTANT**: This is the final submission method for the competition! Please read this guide carefully.

This guide explains how to build and run the Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA) inference program in a Docker environment with GPU acceleration support.

## 🎯 Must Read Before Submission

1. ✅ Ensure `model.py` output format is correct (see main README.md)
2. ✅ Test Docker on validation set
3. ✅ Upload predictions to Codabench platform for validation
4. ✅ Submit Docker image only after validation passes

---

## 📋 Prerequisites

### 1. Install Docker
Download and install Docker Desktop (supports Windows, macOS, and Linux):
- Download: https://www.docker.com/get-started/

After installation, verify Docker is working:
```sh
docker --version
```
If it shows the version number, Docker is installed successfully.

### 2. Install NVIDIA Container Toolkit (GPU Support)
To use GPU acceleration, install NVIDIA Container Toolkit:
```sh
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify GPU availability:
```sh
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 🔨 Building Docker Image

### Step 1: Prepare Files
Ensure the following files are in the same directory:
```
docker/
├── Dockerfile
├── model.py
├── requirements.txt
└── xxx (best_model.pth(your model))  # Trained model weights file
```

### Step 2: Build Image
```sh
cd /path/to/docker
docker build -f Dockerfile -t multi-task-medical .
```

Parameter description:
- `-f Dockerfile` — Specify Dockerfile
- `-t multi-task-medical` — Image name (customizable)
- `.` — Use current directory as build context

Build process takes about 5-10 minutes, depending on network speed.

## 🚀 Running Docker Container

### Basic Usage
```sh
docker run --gpus all --rm \
  -v /path/to/data:/input/:ro \
  -v /path/to/output:/output \
  -it multi-task-medical
```

### Parameter Description
- `--gpus all` — Enable all available GPUs (omit if no GPU available)
- `--rm` — Automatically remove container after it stops
- `-v /host/path:/container/path` — Mount local directory to container:
  - `/input/:ro` — Input data directory (read-only)
  - `/output` — Output results directory (writable)
- `-it` — Interactive mode, shows running logs
- `multi-task-medical` — Docker image name

## 📁 Data Directory Structure

### Input Data Structure (/input/)
```
/input/
├── csv_files/
│   ├── xxx.csv
│   ├── xxx.csv
│   └── ...
└── (other data files, according to paths in CSV)
```

### Expected Output Structure (/output/)
```
/output/
├── classification_predictions.json  # Classification task results
├── detection_predictions.json       # Detection task results
├── regression_predictions.json      # Regression task results
└── Segmentation/                    # Segmentation task results (maintain same relative directory structure as dataset labels)
    └── ...
```

## 📤 Submitting Docker Image

After building, you can submit the image in two ways:

### Method 1: Share via Docker Hub (Recommended)

#### 1. Create Docker Hub Account
- Register at: https://hub.docker.com/
- Create a new repository after login

#### 2. Login to Docker Hub
```sh
docker login
```
Enter your Docker Hub username and password.

#### 3. Tag Image
```sh
docker tag multi-task-medical [your_dockerhub_username]/multi-task-medical:latest
```
Replace `[your_dockerhub_username]` with your Docker Hub username.

#### 4. Push Image to Docker Hub
```sh
docker push [your_dockerhub_username]/multi-task-medical:latest
```

#### 5. Share Image
Send the image address `[your_dockerhub_username]/multi-task-medical:latest` to us via email.

### Method 2: Share via .tar File (Offline or Private Transfer)

#### 1. Save Image as .tar File
```sh
docker save -o multi-task-medical.tar multi-task-medical
```
This creates a `multi-task-medical.tar` file containing the image.

#### 2. Send .tar File
Share with us via cloud storage (e.g., Baidu Pan, Google Drive, etc.).

## 🔍 Testing and Debugging

### View Image List
```sh
docker images
```

### Enter Container for Debugging
```sh
docker run --gpus all --rm \
  -v /path/to/data:/input/:ro \
  -v /path/to/output:/output \
  -it multi-task-medical /bin/bash
```

### View Container Logs
If container is running in background:
```sh
docker logs [container_id]
```

### Delete Image
```sh
docker rmi multi-task-medical
```

## ⚠️ FAQ

### 1. GPU Not Available
Ensure:
- NVIDIA driver is installed
- NVIDIA Container Toolkit is installed
- Using `--gpus all` parameter

### 2. Out of Memory
Reduce batch size, modify `batch_size` parameter in `model.py`.

### 3. Permission Error
Ensure output directory has write permission:
```sh
chmod -R 777 /path/to/output
```

### 4. Model File Not Found
Ensure file is in docker directory and added to Dockerfile before building.


