"""
Captcha Fast Detector package.

This package provides ROI calibration and character segmentation for captcha images.
"""

from .roi_calibrator import ROICalibrator, ROIBounds
from .character_segmenter import CharacterSegmenter

__version__ = "0.2.0"
__all__ = ["ROICalibrator", "ROIBounds", "CharacterSegmenter"]
