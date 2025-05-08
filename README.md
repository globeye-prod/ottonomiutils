# OttonomiUtils

A high-performance Python package for ONNX model inference with GPU acceleration support. This package provides a simple interface for running object detection models with automatic GPU acceleration through TensorRT and CUDA providers.

## Features

- 🚀 **GPU Acceleration**: Automatic support for TensorRT and CUDA providers
- 📦 **Batch Processing**: Efficient batch inference for multiple images
- 🔄 **Automatic Provider Selection**: Smart fallback from TensorRT → CUDA → CPU
- 🖼️ **Built-in Preprocessing**: Automatic image preprocessing for model input
- 🎯 **Confidence Filtering**: Built-in confidence threshold filtering
- 📐 **Bounding Box Utilities**: Automatic conversion between bounding box formats

## Installation

### From PyPI (Coming Soon)
```bash
pip install ottonomiutils
```

### From Source
```bash
# Clone the repository
git clone https://github.com/yourusername/ottonomiutils.git
cd ottonomiutils

# Install the package
pip install -e .

# For development installation with all dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from ottonomiutils import ONNXInferenceModel
import cv2

# Initialize the model
model = ONNXInferenceModel(
    model_path="path/to/your/model.onnx",
    img_size=640,
    conf_thresh=0.5,
    batch_size=4
)

# Process a single image
image = cv2.imread("test.jpg")
results = model([image])[0]

# Process results
for label, class_id, bbox, confidence in results:
    print(f"Detected {label} with confidence {confidence:.2f}")
    print(f"Bounding box: {bbox}")
```

## Development

### Setup Development Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest
```

### Code Style
```bash
# Format code
black .
isort .

# Type checking
mypy .

# Linting
flake8
```

## API Reference

### ONNXInferenceModel

```python
model = ONNXInferenceModel(
    model_path: str,
    img_size: int = 640,
    conf_thresh: float = 0.5,
    input_name: str = "images",
    batch_size: int = 4
)
```

#### Parameters

- `model_path` (str): Path to the ONNX model file
- `img_size` (int, optional): Input image size (both height and width). Defaults to 640.
- `conf_thresh` (float, optional): Confidence threshold for detections. Defaults to 0.5.
- `input_name` (str, optional): Name of the model's input tensor. Defaults to "images".
- `batch_size` (int, optional): Maximum batch size for processing. Defaults to 4.

#### Methods

- `__call__(batch_frames: List[np.ndarray]) -> List[List[List]]`: Run inference on a batch of frames
- `preprocess(image: np.ndarray) -> np.ndarray`: Preprocess a single image for model input
- `postprocess(pred: np.ndarray) -> List[List]`: Postprocess model predictions

## Requirements

- Python 3.7+
- CUDA-compatible GPU (optional, for GPU acceleration)
- See `requirements.txt` for full list of dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 