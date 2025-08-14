"""
ROI Calibrator (Step 1)

This package currently provides the ROI calibration utility to learn and apply
the fixed text region from a set of training captcha images.
"""

from .roi_calibrator import ROICalibrator, ROIBounds

__version__ = "0.2.0"
__all__ = ["ROICalibrator", "ROIBounds"]
