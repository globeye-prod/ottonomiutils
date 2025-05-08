"""
OttonomiUtils - A utility package for ONNX model inference

This package provides a high-level interface for running inference with ONNX models,
particularly optimized for object detection tasks. It supports GPU acceleration through
TensorRT and CUDA providers, with automatic fallback to CPU when needed.

Key Features:
    - GPU acceleration support (TensorRT and CUDA)
    - Batch processing capabilities
    - Automatic provider selection
    - Built-in image preprocessing
    - Confidence threshold filtering
    - Bounding box format conversion

Example:
    >>> from ottonomiutils import ONNXInferenceModel
    >>> import cv2
    >>> 
    >>> # Initialize the model
    >>> model = ONNXInferenceModel(
    ...     model_path="path/to/your/model.onnx",
    ...     img_size=640,
    ...     conf_thresh=0.5
    ... )
    >>> 
    >>> # Load and process an image
    >>> image = cv2.imread("test.jpg")
    >>> results = model([image])[0]  # Returns list of detections
    >>> 
    >>> # Process results
    >>> for label, class_id, bbox, confidence in results:
    ...     print(f"Detected {label} with confidence {confidence:.2f}")
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ottonomiutils import ONNXInferenceModel, Detection, FrameResults, BatchResults, BoundingBox

__version__: str = "0.1.0"

from .ottonomiutils import ONNXInferenceModel

__all__: list[str] = ["ONNXInferenceModel"]
