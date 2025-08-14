"""
Character Segmentation module.

This module segments ROI-cropped captcha images into individual character arrays.
Given the consistent structure of captchas (5 characters with fixed positions),
we use simple equal-width segmentation to isolate each character.

Design principles:
- Simple and deterministic: width / 5 for each character
- No complex CV techniques needed due to fixed layout
- Output consistent character dimensions for template building
- Integration with existing ROI calibration workflow
- Support for both image and txt file processing

All code comments and docstrings are in English by project convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image


class CharacterSegmenter:
    """Segment ROI-cropped captcha images into individual character arrays.
    
    Given a ROI array (e.g., 26x12 after ROI calibration), this class extracts 5 character arrays
    of equal width (with remainder pixels distributed) for further processing.
    """
    
    def __init__(self, char_count: int = 5):
        """Initialize the segmenter.
        
        Args:
            char_count: Number of characters to segment (default: 5 for captcha)
        """
        self.char_count = char_count
    
    def segment_chars(self, roi_array: np.ndarray) -> List[np.ndarray]:
        """Segment ROI array into individual character arrays.
        
        Args:
            roi_array: Input ROI array of shape (height, width)
            
        Returns:
            List of character arrays, each of shape (height, char_width)
            
        Raises:
            ValueError: If roi_array is not 2D or width < char_count
        """
        if roi_array.ndim != 2:
            raise ValueError(f"Expected 2D array, got {roi_array.ndim}D")
        
        height, width = roi_array.shape
        if width < self.char_count:
            raise ValueError(f"Width {width} must be >= char_count {self.char_count}")
        
        char_width = width // self.char_count
        remainder = width % self.char_count
        
        # Distribute remainder pixels across first few characters
        chars = []
        current_x = 0
        
        for i in range(self.char_count):
            # Add one extra pixel for first 'remainder' characters
            extra_pixel = 1 if i < remainder else 0
            char_width_actual = char_width + extra_pixel
            
            # Extract character region
            char_array = roi_array[:, current_x:current_x + char_width_actual]
            chars.append(char_array)
            
            current_x += char_width_actual
        
        return chars
    
    def segment_5_chars(self, roi_array: np.ndarray) -> List[np.ndarray]:
        """Convenience method for 5-character captcha segmentation.
        
        Args:
            roi_array: Input ROI array of shape (height, width)
            
        Returns:
            List of 5 character arrays
        """
        return self.segment_chars(roi_array)
    
    def get_char_dimensions(self, roi_array: np.ndarray) -> Tuple[int, int]:
        """Get the dimensions of each character after segmentation.
        
        Args:
            roi_array: Input ROI array
            
        Returns:
            Tuple of (char_height, char_width)
        """
        if roi_array.ndim != 2:
            raise ValueError(f"Expected 2D array, got {roi_array.ndim}D")
        
        height, width = roi_array.shape
        char_width = width // self.char_count
        
        return height, char_width
    
    def validate_segmentation(self, roi_array: np.ndarray, chars: List[np.ndarray]) -> bool:
        """Validate that segmentation covers the entire ROI array.
        
        Args:
            roi_array: Original ROI array
            chars: List of segmented character arrays
            
        Returns:
            True if segmentation is valid, False otherwise
        """
        if not chars:
            return False
        
        # Check total width
        total_width = sum(char.shape[1] for char in chars)
        expected_width = roi_array.shape[1]
        
        if total_width != expected_width:
            return False
        
        # Check height consistency
        expected_height = roi_array.shape[0]
        for char in chars:
            if char.shape[0] != expected_height:
                return False
        
        return True
    
    # ---------- File processing methods ----------
    
    def segment_image_file(self, input_jpg: str | Path, output_dir: str | Path) -> None:
        """Segment a single image file and save 5 character images.
        
        Args:
            input_jpg: Path to input jpg file
            output_dir: Directory to save segmented character images
        """
        input_path = Path(input_jpg)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load image and convert to grayscale
        img = Image.open(input_path)
        if img.mode != "L":
            img = img.convert("L")
        
        # Convert to numpy array and segment
        img_array = np.array(img, dtype=np.uint8)
        chars = self.segment_chars(img_array)
        
        # Save each character
        base_name = input_path.stem
        for i, char_array in enumerate(chars):
            char_img = Image.fromarray(char_array)
            output_file = output_path / f"{base_name}_segment{i}.jpg"
            char_img.save(output_file)
    
    def segment_txt_file(self, input_txt: str | Path, output_dir: str | Path) -> None:
        """Segment a single txt file and save 5 character txt files.
        
        Args:
            input_txt: Path to input txt file
            output_dir: Directory to save segmented character txt files
        """
        input_path = Path(input_txt)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load txt file as RGB array
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        # Parse dimensions from first line
        width, height = map(int, lines[0].strip().split())
        
        # Parse RGB values
        rgb_pixels = []
        for line in lines[1:]:
            line = line.strip()
            if line:
                pixels = line.split()
                for pixel in pixels:
                    if ',' in pixel:
                        r, g, b = map(int, pixel.split(','))
                        rgb_pixels.append([r, g, b])
        
        # Convert to numpy array and reshape
        rgb_array = np.array(rgb_pixels, dtype=np.uint8)
        img_array = rgb_array.reshape(height, width, 3)
        
        # Convert to grayscale for segmentation
        gray_array = img_array.mean(axis=2).astype(np.uint8)
        
        # Segment characters
        chars = self.segment_chars(gray_array)
        
        # Save each character as txt
        base_name = input_path.stem
        for i, char_array in enumerate(chars):
            char_height, char_width = char_array.shape
            
            # Convert back to RGB (repeat grayscale channel)
            rgb_char = np.stack([char_array] * 3, axis=2)
            
            # Save as txt file
            output_file = output_path / f"{base_name}_segment{i}.txt"
            with open(output_file, 'w') as f:
                # Write new dimensions
                f.write(f"{char_width} {char_height}\n")
                
                # Write RGB values row by row
                for row in range(char_height):
                    row_pixels = []
                    for col in range(char_width):
                        r, g, b = rgb_char[row, col]
                        row_pixels.append(f"{r},{g},{b}")
                    f.write(" ".join(row_pixels) + "\n")
    
    def segment_paired_files(self, input_jpg: str | Path, input_txt: str | Path, 
                           output_dir: str | Path) -> None:
        """Segment paired jpg and txt files simultaneously.
        
        Args:
            input_jpg: Path to input jpg file
            input_txt: Path to input txt file
            output_dir: Directory to save segmented character files
        """
        try:
            # Segment image file
            self.segment_image_file(input_jpg, output_dir)
            # Segment txt file
            self.segment_txt_file(input_txt, output_dir)
        except Exception as e:
            print(f"Error processing paired files {input_jpg} + {input_txt}: {e}")
            raise
    
    def segment_directory(self, input_dir: str | Path, output_dir: str | Path) -> None:
        """Segment all paired jpg+txt files in a directory.
        
        Args:
            input_dir: Directory containing paired jpg+txt files
            output_dir: Directory to save segmented character files
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Find all jpg files
        jpg_files = sorted(input_path.glob("*.jpg"))
        processed_count = 0
        error_count = 0
        
        for jpg_file in jpg_files:
            # Find corresponding txt file
            txt_file = jpg_file.with_suffix('.txt')
            
            if not txt_file.exists():
                print(f"Warning: No corresponding txt file found for {jpg_file.name}")
                error_count += 1
                continue
            
            try:
                # Create subdirectory for this file's segments
                file_output_dir = output_path / jpg_file.stem
                self.segment_paired_files(jpg_file, txt_file, file_output_dir)
                processed_count += 1
                print(f"Processed: {jpg_file.name} + {txt_file.name}")
            except Exception as e:
                print(f"Error processing {jpg_file.name}: {e}")
                error_count += 1
        
        # Summary
        print(f"\nSegmentation complete:")
        print(f"  Successfully processed: {processed_count} file pairs")
        if error_count > 0:
            print(f"  Errors encountered: {error_count} files")
