import os
from signver.detector import Detector

DIR = os.path.dirname(__file__)
DETECTOR_MODEL_PATH = "models/detector/small"

detector = Detector()
detection_model = detector.load(os.path.join(DIR, DETECTOR_MODEL_PATH))

def detect(input_tensor):
    return detection_model.detect(input_tensor)