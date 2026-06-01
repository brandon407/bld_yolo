# BLD-YOLO: An Enhanced YOLO-based Object Detection Model

> Official implementation of the BLD-YOLO paper for real-time object detection tasks.

## 📋 Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Training](#training)
  - [Prediction](#prediction)
  - [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Model Configuration](#model-configuration)
- [Model Export](#model-export)
- [RKNN Deployment](#rknn-deployment)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Introduction

BLD-YOLO is an enhanced YOLO-based object detection model designed for high-performance real-time detection tasks. The model leverages advanced architectural designs to achieve superior accuracy-efficiency trade-offs compared to traditional YOLO models.

**Key Features:**
- Enhanced GELAN backbone for improved feature extraction
- Optimized RepNCSPELAN4 blocks for efficient computation
- Multi-scale feature fusion for robust detection across object sizes
- Support for RKNN deployment on edge devices (RK3588)

---

## 💻 Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.0 (for GPU acceleration)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd bld_yolo
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "from ultralytics.models import YOLO; print('Installation successful!')"
```

---

## 🚀 Quick Start

### Training

Run the training script with default parameters:

```bash
python train.py
```

**Custom Training Parameters:**

```bash
# Training with custom settings
python train.py \
    --model ultralytics/cfg/models/v9/bld_yolo.yaml \
    --data ./datasets/data.yaml \
    --epochs 300 \
    --batch 8 \
    --imgsz 640 \
    --device 0 \
    --project run/train \
    --name exp
```

**Training Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to model configuration file | `ultralytics/cfg/models/v9/bld_yolo.yaml` |
| `--data` | Path to dataset configuration file | `./datasets/data.yaml` |
| `--epochs` | Number of training epochs | 300 |
| `--batch` | Batch size | 1 |
| `--imgsz` | Input image size | 640 |
| `--device` | GPU device ID | 0 |
| `--workers` | Number of data loader workers | 1 |
| `--amp` | Enable mixed precision training | True |
| `--mosaic` | Enable mosaic augmentation | False |

### Prediction

Run prediction on an image or video:

```bash
python predict.py
```

**Custom Prediction:**

```python
from ultralytics.models import YOLO

# Load trained model
model = YOLO('run/train/exp/weights/best.pt')

# Run prediction
results = model.predict(
    source='path/to/image.jpg',  # or video file, camera (0), or directory
    device='0',
    imgsz=640,
    project='runs/detect/',
    name='exp',
    save=True
)
```

### Evaluation

Evaluate model performance on validation dataset:

```python
from ultralytics.models import YOLO

model = YOLO('run/train/exp/weights/best.pt')
results = model.val(data='./datasets/data.yaml', device='0')
```

---

## 📁 Project Structure

```
bld_yolo/
├── datasets/                    # Dataset directory
│   ├── images/                  # Training/Testing images
│   │   ├── train/               # Training images
│   │   └── test/                # Testing images
│   ├── labels/                  # Annotation labels (YOLO format)
│   │   ├── train/               # Training labels
│   │   └── test/                # Testing labels
│   └── data.yaml                # Dataset configuration
├── ultralytics/                 # Ultralytics YOLO framework
│   ├── cfg/                     # Configuration files
│   │   └── models/
│   │       └── v9/
│   │           └── bld_yolo.yaml  # BLD-YOLO model config
│   ├── models/                  # Model definitions
│   ├── engine/                  # Training/Prediction engines
│   └── data/                    # Data handling utilities
├── rknn3588/                    # RKNN deployment tools
│   ├── detect.py                # RKNN inference script
│   ├── func.py                  # Utility functions
│   └── rknnpool.py              # RKNN thread pool
├── train.py                     # Training script
├── predict.py                   # Prediction script
├── export.py                    # Model export script
├── convert_pt_to_onnx.py        # PyTorch to ONNX converter
├── track.py                     # Object tracking script
└── requirements.txt             # Dependencies
```

---

## 📊 Dataset Preparation

### Dataset Format

The dataset should follow the YOLO format:

```
datasets/
├── images/
│   ├── train/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   └── test/
│       ├── 0501.jpg
│       ├── 0502.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── 0001.txt
    │   ├── 0002.txt
    │   └── ...
    └── test/
        ├── 0501.txt
        ├── 0502.txt
        └── ...
```

### Label Format

Each label file contains bounding boxes in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

### data.yaml Configuration

```yaml
train: ./datasets/images/train
val: ./datasets/images/test
test: ./datasets/images/test

nc: 80  # number of classes
names: ['class1', 'class2', ..., 'class80']
```

---

## ⚙️ Model Configuration

The BLD-YOLO model configuration is defined in `ultralytics/cfg/models/v9/bld_yolo.yaml`:

**Architecture Overview:**
- **Backbone:** GELAN with Conv, ELAN1, RepNCSPELAN4, and SPPGELAN blocks
- **Neck:** Multi-scale feature fusion with upsampling and concatenation
- **Head:** Detect layer with P3, P4, P5 outputs

**Key Components:**
| Component | Description |
|-----------|-------------|
| `Conv` | Standard convolution layer |
| `ELAN1` | Enhanced Local Aggregation Network |
| `RepNCSPELAN4` | Reparameterized NCSP ELAN block |
| `PConv` | Point-wise convolution |
| `SPPGELAN` | Spatial Pyramid Pooling with GELAN |

---

## 📤 Model Export

### Export to ONNX

```bash
python convert_pt_to_onnx.py --weights run/train/exp/weights/best.pt
```

Or using the export script:

```bash
python export.py --weights run/train/exp/weights/best.pt --format onnx
```

### Export Options

| Format | Command | Description |
|--------|---------|-------------|
| ONNX | `--format onnx` | ONNX standard format |
| TensorRT | `--format engine` | NVIDIA TensorRT |
| RKNN | See RKNN Deployment | Rockchip NPU |

---

## 🔧 RKNN Deployment

### Requirements

- Rockchip RK3588 device
- RKNN Toolkit Lite 2.0.0

### Step 1: Install RKNN Toolkit

```bash
cd rknn3588/
pip install rknn_toolkit_lite2-2.0.0b0-cp310-cp310-linux_aarch64.whl
```

### Step 2: Convert Model to RKNN

```bash
python export.py --weights run/train/exp/weights/best.pt --format rknn
```

### Step 3: Run Inference on RK3588

```bash
cd rknn3588/
python detect.py --model model.rknn --image test.jpg
```

### Available RKNN Scripts

| Script | Description |
|--------|-------------|
| `detect.py` | Single image detection |
| `test_image.py` | Batch image testing |
| `test_video.py` | Video inference |
| `test_camera_office.py` | Camera real-time detection |
| `dual_model_detect.py` | Dual model inference |

---

## 📝 Citation

If you use BLD-YOLO in your research, please cite our paper:

```bibtex
@article{bldyolo2024,
  title={BLD-YOLO: An Enhanced YOLO-based Object Detection Model for Real-time Applications},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2024},
  volume={XX},
  pages={XX-XX}
}
```

---

## 📄 License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

---

**Last Updated:** April 2024  
**Version:** 1.0.0