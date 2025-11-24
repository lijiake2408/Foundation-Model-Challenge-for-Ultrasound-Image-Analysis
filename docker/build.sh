#!/bin/bash
# Docker Image Build Script

set -e

IMAGE_NAME="multi-task-medical"
TAG="latest"

echo "========================================"
echo "Building Docker image..."
echo "========================================"

# Check required files
echo "Checking required files..."
required_files=("Dockerfile" "model.py" "model_factory.py" "requirements.txt" "best_model.pth")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: File $file not found!"
        exit 1
    fi
    echo "  ✓ $file"
done

echo ""
echo "Building image: ${IMAGE_NAME}:${TAG}"
echo ""

# Build image
docker build -f Dockerfile -t ${IMAGE_NAME}:${TAG} .

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Image built successfully!"
    echo "========================================"
    echo ""
    echo "Image info:"
    docker images ${IMAGE_NAME}:${TAG}
    echo ""
    echo "Run example:"
    echo "docker run --gpus all --rm \\"
    echo "  -v /path/to/data:/input/:ro \\"
    echo "  -v /path/to/output:/output \\"
    echo "  -it ${IMAGE_NAME}:${TAG}"
else
    echo ""
    echo "========================================"
    echo "✗ Image build failed!"
    echo "========================================"
    exit 1
fi

