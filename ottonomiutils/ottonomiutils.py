"""
Core module for ONNX model inference with GPU acceleration support.
"""

import cv2
import numpy as np
import onnxruntime
from typing import List, Dict, Tuple, Union, Optional, Any

# Type aliases for better readability
Detection = List[Union[str, int, List[float], float]]  # [label, class_id, bbox, confidence]
FrameResults = List[Detection]
BatchResults = List[FrameResults]
BoundingBox = List[float]  # [x1, y1, x2, y2]

class ONNXInferenceModel:
    """Base class for ONNX model inference with GPU acceleration support.
    
    This class provides a high-level interface for running inference with ONNX models,
    with automatic GPU acceleration when available. It supports both single image and
    batch processing, with built-in preprocessing and postprocessing steps.
    
    The model automatically selects the best available execution provider in this order:
    1. TensorRT (if available)
    2. CUDA (if available)
    3. CPU (fallback)
    
    Attributes:
        model (onnxruntime.InferenceSession): The loaded ONNX model
        input_name (str): Name of the model's input tensor
        img_size (int): Input image size (both height and width)
        conf_thresh (float): Confidence threshold for detections
        batch_size (int): Maximum batch size for processing
    
    Example:
        >>> # Initialize the model
        >>> model = ONNXInferenceModel(
        ...     model_path="path/to/your/model.onnx",
        ...     img_size=640,
        ...     conf_thresh=0.5,
        ...     batch_size=4
        ... )
        >>> 
        >>> # Process a single image
        >>> image = cv2.imread("test.jpg")
        >>> results = model([image])[0]
        >>> 
        >>> # Process a batch of images
        >>> images = [cv2.imread(f"image_{i}.jpg") for i in range(3)]
        >>> batch_results = model(images)
        >>> 
        >>> # Access detection results
        >>> for frame_results in batch_results:
        ...     for label, class_id, bbox, confidence in frame_results:
        ...         print(f"Detected {label} with confidence {confidence:.2f}")
        ...         print(f"Bounding box: {bbox}")
    """
    
    def __init__(
        self, 
        model_path: str, 
        img_size: int = 640,
        conf_thresh: float = 0.5,
        input_name: str = "images",
        batch_size: int = 4,
    ) -> None:
        """Initialize ONNX model with specified parameters.
        
        Args:
            model_path (str): Path to ONNX model file
            img_size (int, optional): Input image size (both height and width). Defaults to 640.
            conf_thresh (float, optional): Confidence threshold for detections. Defaults to 0.5.
            input_name (str, optional): Name of the model's input tensor. Defaults to "images".
            batch_size (int, optional): Maximum batch size for processing. Defaults to 4.
            
        Note:
            The model will automatically select the best available execution provider
            (TensorRT > CUDA > CPU) and perform a warmup run if using TensorRT.
        """
        self.model: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            model_path,
            providers=[
                'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'
            ],
        )
        self.input_name: str = input_name
        self.img_size: int = img_size
        self.conf_thresh: float = conf_thresh
        self.batch_size: int = batch_size
        
        # Run warmup if using TensorRT provider
        if self.model.get_providers()[0] == 'TensorrtExecutionProvider':
            print("Warming up TensorRT provider...")
            dummy: np.ndarray = np.zeros((self.batch_size, 3, self.img_size, self.img_size), dtype=np.float32)
            for _ in range(8):
                self.model.run(None, {self.input_name: dummy})
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input.
        
        Performs the following preprocessing steps:
        1. Converts BGR to RGB
        2. Resizes to model input size
        3. Normalizes pixel values to [0, 1]
        4. Transposes to NCHW format
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            np.ndarray: Preprocessed image as numpy array in NCHW format
        """
        img: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))[None, :]

    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert bbox format from [x,y,w,h] to [x1,y1,x2,y2].
        
        Args:
            boxes (np.ndarray): Bounding boxes in [x,y,w,h] format
            
        Returns:
            np.ndarray: Bounding boxes in [x1,y1,x2,y2] format
        """
        boxes = np.array(boxes, dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        if boxes.size == 0:
            return boxes
        x_c, y_c, w, h = boxes.T
        return np.stack([x_c, y_c, x_c + w, y_c + h], axis=1)

    def postprocess(self, pred: np.ndarray) -> FrameResults:
        """Postprocess model output to get detections.
        
        Args:
            pred (np.ndarray): Raw model predictions
            
        Returns:
            FrameResults: List of detections in format [label, class_id, bbox, confidence]
        """
        keep: np.ndarray = pred[:, 4] >= self.conf_thresh
        sel: np.ndarray = pred[keep]
        postprocess_result: Dict[str, List[Any]] = {
            "xyxy": self._xywh_to_xyxy(sel[:, :4]).tolist(),
            "conf": sel[:, 4].tolist(),
            "cls": sel[:, 5].astype(int).tolist(),
        }

        all_detections: FrameResults = []
        for i in range(len(postprocess_result["xyxy"])):
            detection: Detection = [
                "gun",  # label
                postprocess_result["cls"][i],  # class id
                postprocess_result["xyxy"][i],  # bounding box
                postprocess_result["conf"][i],  # confidence score
            ]
            all_detections.append(detection)
        return all_detections

    def __call__(self, batch_frames: List[np.ndarray]) -> BatchResults:
        """Run inference on batch of frames.
        
        Args:
            batch_frames (List[np.ndarray]): List of input frames in BGR format
            
        Returns:
            BatchResults: List of detection results for each frame.
                         Each frame's results is a list of detections,
                         where each detection is [label, class_id, bbox, confidence]
                             
        Example:
            >>> model = ONNXInferenceModel("model.onnx")
            >>> frames = [cv2.imread(f"frame_{i}.jpg") for i in range(3)]
            >>> results = model(frames)
            >>> # results[0] contains detections for first frame
            >>> # results[1] contains detections for second frame
            >>> # etc.
        """
        batch_frames = [self.preprocess(p) for p in batch_frames]

        if len(batch_frames) < self.batch_size:
            last_frame: Optional[np.ndarray] = batch_frames[-1] if batch_frames else None
            if last_frame is None:
                raise ValueError("No frames provided to process")
            padding_needed: int = self.batch_size - len(batch_frames)
            batch_frames.extend([last_frame] * padding_needed)

        batch_frames = np.array(batch_frames).squeeze()
        outs: np.ndarray = self.model.run(None, {self.input_name: batch_frames})[0]

        results: BatchResults = []
        for o in outs[: len(batch_frames) - padding_needed]:
            result: FrameResults = self.postprocess(o)
            results.append(result)
        return results

