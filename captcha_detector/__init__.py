"""
Captcha Fast Detector package.

This package provides ROI calibration and character segmentation for captcha images.
"""

from .roi_calibrator import ROICalibrator, ROIBounds
from .character_segmenter import CharacterSegmenter
from .dataset_organizer import DatasetOrganizer
from .template_builder import TemplateBuilder
from .character_recognizer import CharacterRecognizer
from .evaluation import EvaluationFramework
from .captcha_detector import CaptchaDetector

__version__ = "0.2.0"
__all__ = [
    "ROICalibrator", 
    "ROIBounds", 
    "CharacterSegmenter", 
    "DatasetOrganizer", 
    "TemplateBuilder",
    "CharacterRecognizer",
    "EvaluationFramework",
    "CaptchaDetector"
]
