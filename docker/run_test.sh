#!/bin/bash
# Docker Image Test Script

IMAGE_NAME="multi-task-medical"
TAG="latest"

# Default paths (can be overridden by command line arguments)
INPUT_DIR="${1:-/root/baseline/train}"
OUTPUT_DIR="${2:-/root/baseline/docker_output}"

echo "========================================"
echo "Testing Docker image..."
echo "========================================"
echo ""
echo "Image name: ${IMAGE_NAME}:${TAG}"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Check input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory created: $OUTPUT_DIR"
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, will use GPU acceleration"
    GPU_FLAG="--gpus all"
else
    echo "No GPU detected, will use CPU"
    GPU_FLAG=""
fi
echo ""

echo "Starting container..."
echo "========================================"

# Run container
docker run $GPU_FLAG --rm \
  -v "${INPUT_DIR}":/input/:ro \
  -v "${OUTPUT_DIR}":/output \
  -it ${IMAGE_NAME}:${TAG}

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Inference completed!"
    echo "========================================"
    echo ""
    echo "Output files:"
    ls -lh "$OUTPUT_DIR"
    echo ""
    echo "View results:"
    echo "  - Classification: $OUTPUT_DIR/classification_predictions.json"
    echo "  - Detection: $OUTPUT_DIR/detection_predictions.json"
    echo "  - Regression: $OUTPUT_DIR/regression_predictions.json"
    echo "  - Segmentation: $OUTPUT_DIR/Segmentation/"
else
    echo ""
    echo "========================================"
    echo "✗ Inference failed!"
    echo "========================================"
    exit 1
fi

