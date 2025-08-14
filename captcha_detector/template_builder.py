"""
Template Builder module.

This module builds character templates from the organized dataset by averaging
samples for each character (A-Z, 0-9). Templates are used for character recognition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
from collections import Counter


class TemplateBuilder:
    """Build character templates from organized dataset.
    
    Creates templates by averaging pixel values across all samples for each character.
    Templates are saved in txt format with metadata for tracking.
    """
    
    def __init__(self, char_dataset_dir: str | Path, target_size: Tuple[int, int] = (6, 12)):
        """Initialize the template builder.
        
        Args:
            char_dataset_dir: Directory containing organized character dataset
            target_size: Target size (width, height) - consistent with txt format
        """
        self.char_dataset_dir = Path(char_dataset_dir)
        self.target_size = target_size  # (width, height)
        
        if not self.char_dataset_dir.exists():
            raise ValueError(f"Character dataset directory does not exist: {char_dataset_dir}")
    
    def build_templates(self, output_dir: str | Path) -> None:
        """Build templates for all characters and save them.
        
        Args:
            output_dir: Directory to save templates and metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Expected labels
        expected_labels = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        templates = {}
        metadata = {
            "target_size": list(self.target_size),
            "char_count": len(expected_labels),
            "samples_per_char": {},
            "created_at": None
        }
        
        print("Building character templates...")
        
        for label in sorted(expected_labels):
            label_dir = self.char_dataset_dir / label
            
            if not label_dir.exists():
                print(f"Warning: No samples found for label '{label}'")
                continue
            
            # Find all sample files for this label
            sample_files = sorted(label_dir.glob(f"{label}_sample*.txt"))
            
            if not sample_files:
                print(f"Warning: No sample files found for label '{label}'")
                continue
            
            print(f"Processing label '{label}' with {len(sample_files)} samples...")
            
            # Load and canonicalize all samples
            canonicalized_samples = []
            for sample_file in sample_files:
                try:
                    sample_array = self._load_txt_as_array(sample_file)
                    canonicalized = self._canonicalize_character(sample_array)
                    canonicalized_samples.append(canonicalized)
                except Exception as e:
                    print(f"Warning: Failed to process {sample_file.name}: {e}")
                    continue
            
            if not canonicalized_samples:
                print(f"Warning: No valid samples for label '{label}'")
                continue
            
            # Build template by averaging samples
            template = self._build_template_from_samples(canonicalized_samples)
            
            # Save template
            template_file = output_path / f"{label}.txt"
            self._save_array_as_txt(template, template_file)
            
            # Store metadata
            templates[label] = str(template_file)
            metadata["samples_per_char"][label] = len(canonicalized_samples)
            
            print(f"  Saved template: {template_file.name}")
        
        # Save metadata
        metadata_file = output_path / "meta.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nTemplate building complete:")
        print(f"  Templates saved to: {output_path}")
        print(f"  Metadata saved to: {metadata_file}")
        print(f"  Total templates: {len(templates)}")
    
    def _load_txt_as_array(self, txt_path: Path) -> np.ndarray:
        """Load a txt file as RGB numpy array.
        
        Args:
            txt_path: Path to txt file
            
        Returns:
            numpy array of shape (height, width, 3) with RGB values
        """
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        # Parse dimensions from first line (height width format)
        height, width = map(int, lines[0].strip().split())
        
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
        return img_array
    
    def _canonicalize_character(self, char_array: np.ndarray) -> np.ndarray:
        """Canonicalize character array to target size.
        
        Args:
            char_array: Input character array
            
        Returns:
            Canonicalized array of target size
        """
        # Convert to grayscale
        if char_array.ndim == 3:
            gray_array = char_array.mean(axis=2).astype(np.uint8)
        else:
            gray_array = char_array
        
        # Note: char_array is (height, width) from reshape(height, width)
        current_height, current_width = gray_array.shape
        target_width, target_height = self.target_size  # (width, height)
        
        # Calculate background color (use the most common pixel value)
        pixel_counts = Counter(gray_array.flatten())
        background_color = pixel_counts.most_common(1)[0][0]
        
        # Handle width padding/truncation
        if current_width < target_width:
            # Pad with background color on both sides (not zeros!)
            pad_left = (target_width - current_width) // 2
            pad_right = target_width - current_width - pad_left
            
            padded = np.full((current_height, target_width), background_color, dtype=np.uint8)
            padded[:, pad_left:pad_left + current_width] = gray_array
            gray_array = padded
        elif current_width > target_width:
            # Truncate from center
            start = (current_width - target_width) // 2
            gray_array = gray_array[:, start:start + target_width]
        
        # Handle height (should already be correct from ROI calibration)
        if current_height != target_height:
            # This shouldn't happen with our ROI calibration, but handle it
            if current_height < target_height:
                # Pad vertically with background color
                pad_top = (target_height - current_height) // 2
                pad_bottom = target_height - current_height - pad_top
                
                padded = np.full((target_height, target_width), background_color, dtype=np.uint8)
                padded[pad_top:pad_top + current_height, :] = gray_array
                gray_array = padded
            else:
                # Truncate vertically
                start = (current_height - target_height) // 2
                gray_array = gray_array[start:start + target_height, :]
        
        # Final shape should be (target_height, target_width)
        assert gray_array.shape == (target_height, target_width), f"Expected ({target_height}, {target_width}), got {gray_array.shape}"
        return gray_array
    
    def _build_template_from_samples(self, samples: List[np.ndarray]) -> np.ndarray:
        """Build template by averaging multiple samples.
        
        Args:
            samples: List of canonicalized sample arrays
            
        Returns:
            Template array (average of all samples)
        """
        if not samples:
            raise ValueError("No samples provided")
        
        # Stack samples and compute mean
        stacked = np.stack(samples, axis=0)
        template = stacked.mean(axis=0).astype(np.uint8)
        
        return template
    
    def _save_array_as_txt(self, array: np.ndarray, output_path: Path) -> None:
        """Save array as txt file.
        
        Args:
            array: numpy array to save
            output_path: path to save the txt file
        """
        height, width = array.shape
        
        with open(output_path, 'w') as f:
            # Write dimensions as height width (following original data format)
            f.write(f"{height} {width}\n")
            
            # Write pixel values row by row
            for row in range(height):
                row_pixels = []
                for col in range(width):
                    pixel_value = int(array[row, col])
                    # Convert to RGB format (repeat grayscale value)
                    row_pixels.append(f"{pixel_value},{pixel_value},{pixel_value}")
                f.write(" ".join(row_pixels) + "\n")
    
    def load_template(self, template_path: str | Path) -> np.ndarray:
        """Load a template from file.
        
        Args:
            template_path: Path to template file
            
        Returns:
            Template array
        """
        return self._load_txt_as_array(Path(template_path))
    
    def get_template_info(self, templates_dir: str | Path) -> Dict:
        """Get information about built templates.
        
        Args:
            templates_dir: Directory containing templates
            
        Returns:
            Dictionary with template information
        """
        templates_path = Path(templates_dir)
        meta_file = templates_path / "meta.json"
        
        if not meta_file.exists():
            return {"error": "Metadata file not found"}
        
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Count actual template files
        template_files = list(templates_path.glob("*.txt"))
        metadata["actual_templates"] = len(template_files)
        metadata["template_files"] = [f.name for f in template_files]
        
        return metadata
