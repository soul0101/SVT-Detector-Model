import os
from signver.detector import Detector

DIR = os.path.dirname(__file__)
DETECTOR_MODEL_PATH = "models/detector/small"

detection_model = Detector()
detection_model.load(os.path.join(DIR, DETECTOR_MODEL_PATH))

def detect(input_tensor):
    """
    This function takes in an image tensor and returns the bounding boxes, scores, classes, and detections of the image.

    Parameters
    ----------
    img_tensor : Tensor
        Image tensor of the form (1, img_height, img_width, 3)
        
    Returns
    -------
    boxes : Tensor
        A list of 4 element tuples of the form (y1, x1, y2, x2)
    scores : Tensor
        A list of confidence scores for each of the detected objects
    classes : Tensor
        A list of class labels for each detected object
    detections : Tensor
    """
    return detection_model.detect(input_tensor)
