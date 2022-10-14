import os
from signver.detector import Detector

DIR = os.path.dirname(__file__)
DETECTOR_MODEL_PATH = "models/detector/small"

detection_model = Detector()
detection_model.load(os.path.join(DIR, DETECTOR_MODEL_PATH))

def detect(input_tensor):
    return detection_model.detect(input_tensor)
