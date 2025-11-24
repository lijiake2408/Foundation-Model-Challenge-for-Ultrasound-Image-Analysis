# Quick Start Guide

## 🚀 Complete Docker Packaging and Testing in 3 Steps

### Step 1: Build Docker Image
```bash
cd /root/baseline/docker
./build.sh
```

### Step 2: Test Docker Image
```bash
# Test with default paths
./run_test.sh

# Or specify custom paths
./run_test.sh /path/to/input /path/to/output
```

### Step 3: Submit Image

#### Method A: Docker Hub (Recommended)
```bash
# 1. Login
docker login

# 2. Tag image
docker tag multi-task-medical:latest YOUR_USERNAME/multi-task-medical:latest

# 3. Push
docker push YOUR_USERNAME/multi-task-medical:latest

# 4. Share link
# YOUR_USERNAME/multi-task-medical:latest
```

#### Method B: .tar File
```bash
# 1. Save image
docker save -o multi-task-medical.tar multi-task-medical:latest

# 2. Upload to cloud storage and share
```

## 📋 File Checklist

- ✅ `Dockerfile` - Docker configuration file
- ✅ `model.py` - Inference main program
- ✅ `xxx.py` - Other programs
- ✅ `requirements.txt` - Python dependencies
- ✅ `xxx.pth` - Trained model weights
- ✅ `README.md` - Detailed documentation
- ✅ `build.sh` - Build script
- ✅ `run_test.sh` - Test script


## 🔍 Verify Image

```bash
# View image list
docker images | grep multi-task-medical

# View image size and information
docker inspect multi-task-medical:latest

# Test GPU availability (if GPU available)
docker run --gpus all --rm multi-task-medical:latest python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📊 Expected Output

After successful execution, output directory should contain:

```
output/
├── classification_predictions.json  # Classification task results
├── detection_predictions.json       # Detection task results
├── regression_predictions.json      # Regression task results
└── Segmentation/                    # Segmentation task results
    └── ...
```

## ❓ FAQ

**Q: Build is very slow?**
A: First build needs to download dependencies, takes about 5-10 minutes. Subsequent builds will use cache and be faster.

**Q: Can I run without GPU?**
A: Yes, just remove the `--gpus all` parameter, but it will be slower.

**Q: Output directory permission error?**
A: Run `chmod -R 777 /path/to/output` to ensure directory is writable.

## 📞 Getting Help

See `README.md` for detailed documentation.

---

**Tip**: First build may take a long time, please be patient.

