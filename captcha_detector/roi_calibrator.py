"""
ROI Calibrator module.

This module provides a one-time calibration step to learn a fixed Region of
Interest (ROI) from a set of 25 training captcha images that share the same
layout. The learned ROI removes consistent outer borders so that subsequent
stages (not part of step 1) can work only on the text region.

Design principles:
- No machine learning. This is a deterministic calibration based on pixel
  consistency across the given images.
- The calibration produces a fixed ROI tuple (top, bottom, left, right) that
  can be persisted and reused for all future images from the same generator.

All code comments and docstrings are in English by project convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import json
import math
import numpy as np
from PIL import Image


@dataclass
class ROIBounds:
    """Inclusive-exclusive ROI bounds of the form (top, bottom, left, right).

    All indices follow numpy slicing semantics: [top:bottom, left:right].
    """

    top: int
    bottom: int
    left: int
    right: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.top, self.bottom, self.left, self.right)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)


class ROICalibrator:
    """Calibrate and apply a fixed ROI for captcha images.

    Workflow:
    1) Calibrate from a directory of images (typically 25 samples)
    2) Persist the bounds to JSON
    3) Load bounds later and apply to new images
    """

    def __init__(self, expected_size: Tuple[int, int] | None = (60, 30)) -> None:
        """Initialize calibrator.

        Args:
            expected_size: Optional (width, height). If provided, images are
                resized to this size for consistent stacking.
        """
        self.expected_size = expected_size
        self.roi_bounds: ROIBounds | None = None

    # ---------- Public API ----------
    def calibrate_from_dir(self, input_dir: str | Path) -> ROIBounds:
        """Calibrate ROI from a directory of images (e.g., data/input).

        Args:
            input_dir: Directory containing training images (*.jpg or *.png)

        Returns:
            ROIBounds object with learned bounds
        """
        paths = self._find_images(input_dir)
        if not paths:
            raise ValueError(f"No images found in: {input_dir}")
        stack = self._load_images(paths)
        bounds = self._learn_roi_from_stack(stack)
        self.roi_bounds = bounds
        return bounds

    def calibrate_from_txt_dir(self, input_dir: str | Path) -> ROIBounds:
        """Calibrate ROI from a directory of txt files.

        Args:
            input_dir: Directory containing training txt files

        Returns:
            ROIBounds object with learned bounds
        """
        txt_paths = self._find_txt_files(input_dir)
        if not txt_paths:
            raise ValueError(f"No txt files found in: {input_dir}")
        stack = self._load_images_txt(txt_paths)
        bounds = self._learn_roi_from_stack(stack)
        self.roi_bounds = bounds
        return bounds

    def save_bounds(self, json_path: str | Path) -> None:
        """Persist learned ROI bounds to a JSON file."""
        if self.roi_bounds is None:
            raise ValueError("ROI is not calibrated. Call calibrate_* first.")
        payload = {
            "top": int(self.roi_bounds.top),
            "bottom": int(self.roi_bounds.bottom),
            "left": int(self.roi_bounds.left),
            "right": int(self.roi_bounds.right),
            "expected_size": list(self.expected_size) if self.expected_size else None,
        }
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_bounds(self, json_path: str | Path) -> ROIBounds:
        """Load ROI bounds from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.expected_size = tuple(data.get("expected_size")) if data.get("expected_size") else self.expected_size
        self.roi_bounds = ROIBounds(
            top=int(data["top"]),
            bottom=int(data["bottom"]),
            left=int(data["left"]),
            right=int(data["right"]),
        )
        return self.roi_bounds

    def extract_roi_array(self, img_array: np.ndarray) -> np.ndarray:
        """Crop a numpy image array using the learned ROI."""
        if self.roi_bounds is None:
            raise ValueError("ROI is not calibrated. Call calibrate_* or load_bounds().")
        t, b, l, r = self.roi_bounds.as_tuple()
        return img_array[t:b, l:r]

    def crop_image_file(self, input_path: str | Path, output_path: str | Path) -> None:
        """Crop an image file with the learned ROI and save to a new file."""
        if self.roi_bounds is None:
            raise ValueError("ROI is not calibrated. Call calibrate_* or load_bounds().")
        img = self._open_grayscale(input_path)
        if self.expected_size is not None:
            img = img.resize(self.expected_size, Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
        cropped = self.extract_roi_array(arr)
        out = Image.fromarray(cropped)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.save(output_path)

    def crop_txt_file(self, input_txt: str | Path, output_txt: str | Path) -> None:
        """Crop a txt file with the learned ROI and save cropped version."""
        if self.roi_bounds is None:
            raise ValueError("ROI is not calibrated. Call calibrate_* or load_bounds().")
        
        img_array = self._load_txt_as_array(input_txt)
        cropped = self.extract_roi_array(img_array)
        self._save_array_as_txt(cropped, output_txt)

    def crop_paired_files(self, input_jpg: str | Path, input_txt: str | Path, 
                         output_jpg: str | Path, output_txt: str | Path) -> None:
        """Crop both jpg and txt files with the learned ROI.
        
        This ensures the jpg and txt outputs remain synchronized after cropping.
        
        Args:
            input_jpg: Input image file path
            input_txt: Input txt file path  
            output_jpg: Output image file path
            output_txt: Output txt file path
        """
        if self.roi_bounds is None:
            raise ValueError("ROI is not calibrated. Call calibrate_* or load_bounds().")
        
        # Crop the jpg file (using CV method)
        self.crop_image_file(input_jpg, output_jpg)
        
        # Crop the txt file (using txt method)
        self.crop_txt_file(input_txt, output_txt)

    def crop_paired_dir(self, input_dir: str | Path, output_dir: str | Path) -> None:
        """Crop all paired jpg+txt files from input directory to output directory.
        
        Expects files to be named like: input00.jpg + input00.txt, etc.
        
        Args:
            input_dir: Directory containing paired jpg+txt files
            output_dir: Directory to save cropped paired files
        """
        if self.roi_bounds is None:
            raise ValueError("ROI is not calibrated. Call calibrate_* or load_bounds().")
            
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all jpg files
        jpg_files = sorted(input_path.glob("*.jpg"))
        
        for jpg_file in jpg_files:
            # Find corresponding txt file
            txt_file = jpg_file.with_suffix('.txt')
            
            if not txt_file.exists():
                print(f"Warning: No corresponding txt file found for {jpg_file.name}")
                continue
            
            # Define output paths
            output_jpg = output_path / jpg_file.name
            output_txt = output_path / txt_file.name
            
            # Crop both files
            self.crop_paired_files(jpg_file, txt_file, output_jpg, output_txt)
            
            print(f"Cropped: {jpg_file.name} + {txt_file.name}")

    # ---------- Core calibration logic ----------
    def _learn_roi_from_stack(self, stack: np.ndarray) -> ROIBounds:
        """Learn ROI bounds from a stack of aligned grayscale images.

        Approach:
        - Compute per-pixel standard deviation across images. Border/background
          is highly consistent -> low std. Text region changes -> higher std.
        - Threshold the std-map using Otsu to obtain a binary "dynamic" mask.
        - Optionally filter thin noise via min-area and row/col activation.
        - Take the bounding box of the dynamic mask as the ROI.
        """
        if stack.ndim != 3:
            raise ValueError("Stack must be (N, H, W)")
        n, h, w = stack.shape
        if n < 2:
            raise ValueError("At least 2 images are required for calibration")

        std_map = stack.astype(np.float32).std(axis=0)
        thr = self._otsu_threshold_float(std_map)
        dynamic = (std_map >= thr).astype(np.uint8)

        # Strengthen mask by requiring minimal activity per row/col
        row_active = (dynamic.sum(axis=1) >= max(1, int(0.05 * w)))
        col_active = (dynamic.sum(axis=0) >= max(1, int(0.05 * h)))
        refined = np.outer(row_active.astype(np.uint8), col_active.astype(np.uint8))

        ys, xs = np.where(refined > 0)
        if ys.size == 0 or xs.size == 0:
            # Fallback: if Otsu failed, use top-k percentile of std_map
            kth = np.percentile(std_map, 90.0)
            dynamic = (std_map >= kth).astype(np.uint8)
            row_active = (dynamic.sum(axis=1) >= max(1, int(0.05 * w)))
            col_active = (dynamic.sum(axis=0) >= max(1, int(0.05 * h)))
            refined = np.outer(row_active.astype(np.uint8), col_active.astype(np.uint8))
            ys, xs = np.where(refined > 0)
            if ys.size == 0 or xs.size == 0:
                # As last resort, keep center 80%
                top = int(round(0.10 * h))
                bottom = int(round(0.90 * h))
                left = int(round(0.10 * w))
                right = int(round(0.90 * w))
                return ROIBounds(top=top, bottom=bottom, left=left, right=right)

        top = int(ys.min())
        bottom = int(ys.max() + 1)
        left = int(xs.min())
        right = int(xs.max() + 1)

        # Small safety expansion (clamped) to avoid cutting strokes
        pad_y = max(0, int(round(0.02 * h)))
        pad_x = max(0, int(round(0.02 * w)))
        top = max(0, top - pad_y)
        bottom = min(h, bottom + pad_y)
        left = max(0, left - pad_x)
        right = min(w, right + pad_x)

        return ROIBounds(top=top, bottom=bottom, left=left, right=right)

    # ---------- Image file helpers ----------
    def _find_images(self, input_dir: str | Path) -> List[Path]:
        p = Path(input_dir)
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        out: List[Path] = []
        for pat in patterns:
            out.extend(sorted(p.glob(pat)))
        return out

    def _load_images(self, paths: Sequence[Path]) -> np.ndarray:
        arrays: List[np.ndarray] = []
        for path in paths:
            img = self._open_grayscale(path)
            if self.expected_size is not None:
                img = img.resize(self.expected_size, Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.uint8)
            arrays.append(arr)
        stack = np.stack(arrays, axis=0)  # (N, H, W)
        return stack

    def _open_grayscale(self, path: str | Path) -> Image.Image:
        img = Image.open(path)
        if img.mode != "L":
            img = img.convert("L")
        return img

    # ---------- TXT file helpers ----------
    def _find_txt_files(self, input_dir: str | Path) -> List[Path]:
        """Find all txt files in the input directory."""
        p = Path(input_dir)
        return sorted(p.glob("*.txt"))

    def _load_images_txt(self, txt_paths: Sequence[Path]) -> np.ndarray:
        """Load txt files and convert to grayscale stack for ROI learning."""
        arrays: List[np.ndarray] = []
        for path in txt_paths:
            rgb_array = self._load_txt_as_array(path)
            # Convert RGB to grayscale (simple average)
            gray_array = rgb_array.mean(axis=2).astype(np.uint8)
            arrays.append(gray_array)
        stack = np.stack(arrays, axis=0)  # (N, H, W)
        return stack

    def _load_txt_as_array(self, txt_path: str | Path) -> np.ndarray:
        """Load a txt file as RGB numpy array.
        
        Expected format:
        First line: width height
        Remaining lines: pixels as "R,G,B R,G,B R,G,B ..." (RGB comma-separated, pixels space-separated)
        
        Returns:
            numpy array of shape (height, width, 3) with RGB values
        """
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        # Parse dimensions from first line
        width, height = map(int, lines[0].strip().split())
        
        # Parse RGB values from remaining lines
        rgb_pixels = []  # List to store [R,G,B] for each pixel
        
        for line in lines[1:]:
            line = line.strip()
            if line:  # Skip empty lines
                # Split by spaces to get individual pixels
                pixels = line.split()
                for pixel in pixels:
                    if ',' in pixel:
                        # Parse "R,G,B" format
                        r, g, b = map(int, pixel.split(','))
                        rgb_pixels.append([r, g, b])
        
        # Convert to numpy array
        if len(rgb_pixels) != width * height:
            raise ValueError(f"Expected {width * height} pixels, got {len(rgb_pixels)}")
        
        rgb_array = np.array(rgb_pixels, dtype=np.uint8)  # Shape: (width*height, 3)
        
        # Reshape to (height, width, 3)
        img_array = rgb_array.reshape(height, width, 3)
        return img_array

    def _save_array_as_txt(self, img_array: np.ndarray, output_path: str | Path) -> None:
        """Save a RGB numpy array as txt file.
        
        Args:
            img_array: numpy array of shape (height, width, 3) or (height, width)
            output_path: path to save the txt file
        """
        if img_array.ndim == 2:
            # Grayscale: convert to RGB by repeating the channel
            height, width = img_array.shape
            rgb_array = np.stack([img_array] * 3, axis=2)
        elif img_array.ndim == 3 and img_array.shape[2] == 3:
            # Already RGB
            height, width = img_array.shape[:2]
            rgb_array = img_array
        else:
            raise ValueError("Array must be (H,W) or (H,W,3)")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Write dimensions
            f.write(f"{width} {height}\n")
            
            # Write RGB values row by row
            # Format: pixels separated by spaces, RGB within pixel separated by commas
            for row in range(height):
                row_pixels = []
                for col in range(width):
                    r, g, b = rgb_array[row, col]
                    row_pixels.append(f"{r},{g},{b}")
                f.write(" ".join(row_pixels) + "\n")    

    # ---------- Mathematical helpers ----------
    def _otsu_threshold_float(self, img: np.ndarray) -> float:
        """Otsu threshold for a floating image using 256-bin histogram."""
        if img.size == 0:
            return 0.0
        vmin = float(np.min(img))
        vmax = float(np.max(img))
        if math.isclose(vmin, vmax):
            return vmin
        bins = 256
        hist, bin_edges = np.histogram(img, bins=bins, range=(vmin, vmax))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total <= 0:
            return vmin
        cumulative = np.cumsum(hist)
        cumulative_mean = np.cumsum(hist * np.arange(bins))
        global_mean = cumulative_mean[-1] / total

        inter_class_var = (
            (global_mean * cumulative - cumulative_mean) ** 2
            / (cumulative * (total - cumulative) + 1e-12)
        )
        inter_class_var[:1] = -1  # avoid 0 division at edges
        inter_class_var[-1:] = -1
        idx = int(np.argmax(inter_class_var))
        thr = float(bin_edges[idx])
        return thr