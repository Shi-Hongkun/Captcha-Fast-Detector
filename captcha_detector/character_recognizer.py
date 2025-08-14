"""
Character Recognizer module.

This module implements template matching for character recognition using
normalized cross-correlation (NCC) and mean absolute error (MAE) as fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


class CharacterRecognizer:
    """Recognize characters using template matching.
    
    Uses normalized cross-correlation (NCC) as primary method and
    mean absolute error (MAE) as fallback for robust recognition.
    """
    
    def __init__(self, templates_dir: str | Path):
        """Initialize the character recognizer.
        
        Args:
            templates_dir: Directory containing character templates
        """
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, np.ndarray] = {}
        self.template_info: Dict = {}
        
        if not self.templates_dir.exists():
            raise ValueError(f"Templates directory does not exist: {templates_dir}")
        
        # Load all templates
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all character templates from the templates directory."""
        print("Loading character templates...")
        
        # Expected labels
        expected_labels = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        for label in sorted(expected_labels):
            template_file = self.templates_dir / f"{label}.txt"
            
            if template_file.exists():
                try:
                    template_array = self._load_txt_as_array(template_file)
                    self.templates[label] = template_array
                    print(f"  Loaded template for '{label}': {template_array.shape}")
                except Exception as e:
                    print(f"Warning: Failed to load template for '{label}': {e}")
            else:
                print(f"Warning: Template file not found for '{label}'")
        
        print(f"Loaded {len(self.templates)} templates")
        
        # Load metadata if available
        meta_file = self.templates_dir / "meta.json"
        if meta_file.exists():
            import json
            with open(meta_file, 'r', encoding='utf-8') as f:
                self.template_info = json.load(f)
    
    def _load_txt_as_array(self, txt_path: Path) -> np.ndarray:
        """Load a txt file as numpy array.
        
        Args:
            txt_path: Path to txt file
            
        Returns:
            numpy array of shape (height, width)
        """
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        # Parse dimensions from first line (height width format)
        height, width = map(int, lines[0].strip().split())
        
        # Parse pixel values
        pixels = []
        for line in lines[1:]:
            line = line.strip()
            if line:
                # Handle both RGB and grayscale formats
                if ',' in line.split()[0]:  # RGB format
                    for pixel in line.split():
                        r, g, b = map(int, pixel.split(','))
                        pixels.append((r + g + b) // 3)  # Convert to grayscale
                else:  # Grayscale format
                    pixels.extend(map(int, line.split()))
        
        # Convert to numpy array and reshape
        array = np.array(pixels, dtype=np.uint8)
        return array.reshape(height, width)
    
    def _canonicalize_character(self, char_array: np.ndarray, target_size: Tuple[int, int] = (12, 9)) -> np.ndarray:
        """Canonicalize character array to target size.
        
        Args:
            char_array: Input character array
            target_size: Target size (width, height)
            
        Returns:
            Canonicalized array of target size
        """
        # Convert to grayscale if needed
        if char_array.ndim == 3:
            gray_array = char_array.mean(axis=2).astype(np.uint8)
        else:
            gray_array = char_array
        
        height, width = gray_array.shape
        target_height, target_width = target_size
        
        # Handle width padding/truncation
        if width < target_width:
            # Pad with zeros (black) on both sides
            pad_left = (target_width - width) // 2
            pad_right = target_width - width - pad_left
            
            padded = np.zeros((height, target_width), dtype=np.uint8)
            padded[:, pad_left:pad_left + width] = gray_array
            gray_array = padded
        elif width > target_width:
            # Truncate from center
            start = (width - target_width) // 2
            gray_array = gray_array[:, start:start + target_width]
        
        # Handle height
        if height != target_height:
            if height < target_height:
                # Pad vertically
                pad_top = (target_height - height) // 2
                pad_bottom = target_height - height - pad_top
                
                padded = np.zeros((target_height, target_width), dtype=np.uint8)
                padded[pad_top:pad_top + height, :] = gray_array
                gray_array = padded
            else:
                # Truncate vertically
                start = (height - target_height) // 2
                gray_array = gray_array[start:start + target_height, :]
        
        return gray_array
    
    def _normalized_cross_correlation(self, query: np.ndarray, template: np.ndarray) -> float:
        """Compute normalized cross-correlation between query and template.
        
        Args:
            query: Query character array
            template: Template character array
            
        Returns:
            NCC score (higher is better)
        """
        # Ensure same size
        if query.shape != template.shape:
            # Canonicalize to template's (height, width)
            query = self._canonicalize_character(query, template.shape)
        
        # Convert to float and normalize
        query_norm = query.astype(np.float64) / 255.0
        template_norm = template.astype(np.float64) / 255.0
        
        # Compute means
        query_mean = np.mean(query_norm)
        template_mean = np.mean(template_norm)
        
        # Center the arrays
        query_centered = query_norm - query_mean
        template_centered = template_norm - template_mean
        
        # Compute correlation
        numerator = np.sum(query_centered * template_centered)
        denominator = np.sqrt(np.sum(query_centered ** 2) * np.sum(template_centered ** 2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _mean_absolute_error(self, query: np.ndarray, template: np.ndarray) -> float:
        """Compute mean absolute error between query and template.
        
        Args:
            query: Query character array
            template: Template character array
            
        Returns:
            MAE score (lower is better)
        """
        # Ensure same size
        if query.shape != template.shape:
            query = self._canonicalize_character(query, template.shape)
        
        # Convert to float and normalize
        query_norm = query.astype(np.float64) / 255.0
        template_norm = template.astype(np.float64) / 255.0
        
        # Compute MAE
        mae = np.mean(np.abs(query_norm - template_norm))
        return mae
    
    def recognize_character(self, char_array: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Recognize a single character using template matching.
        
        Args:
            char_array: Input character array
            
        Returns:
            Tuple of (predicted_label, confidence, all_scores)
        """
        if not self.templates:
            raise ValueError("No templates loaded")
        
        # Canonicalize input character
        canonicalized = self._canonicalize_character(char_array, (12, 9))
        
        # Compute scores for all templates
        ncc_scores = {}
        mae_scores = {}
        
        for label, template in self.templates.items():
            ncc_score = self._normalized_cross_correlation(canonicalized, template)
            mae_score = self._mean_absolute_error(canonicalized, template)
            
            ncc_scores[label] = ncc_score
            mae_scores[label] = mae_score
        
        # Primary method: NCC (higher is better)
        best_ncc_label = max(ncc_scores, key=ncc_scores.get)
        best_ncc_score = ncc_scores[best_ncc_label]
        
        # Fallback method: MAE (lower is better)
        best_mae_label = min(mae_scores, key=mae_scores.get)
        best_mae_score = mae_scores[best_mae_label]
        
        # Use NCC as primary, but if it's too low, use MAE
        if best_ncc_score > 0.3:  # Threshold for NCC
            predicted_label = best_ncc_label
            confidence = best_ncc_score
            method = "NCC"
        else:
            predicted_label = best_mae_label
            confidence = 1.0 - (best_mae_score / 1.0)  # Convert MAE to confidence
            method = "MAE"
        
        # Get second best for margin calculation
        ncc_scores_sorted = sorted(ncc_scores.items(), key=lambda x: x[1], reverse=True)
        if len(ncc_scores_sorted) > 1:
            second_best_score = ncc_scores_sorted[1][1]
            margin = best_ncc_score - second_best_score
        else:
            margin = 0.0
        
        all_scores = {
            "ncc_scores": ncc_scores,
            "mae_scores": mae_scores,
            "method": method,
            "margin": margin
        }
        
        return predicted_label, confidence, all_scores
    
    def recognize_captcha(self, char_arrays: List[np.ndarray]) -> Tuple[str, List[Tuple[str, float]]]:
        """Recognize a complete captcha from 5 character arrays.
        
        Args:
            char_arrays: List of 5 character arrays
            
        Returns:
            Tuple of (captcha_text, character_predictions)
        """
        if len(char_arrays) != 5:
            raise ValueError(f"Expected 5 character arrays, got {len(char_arrays)}")
        
        captcha_text = ""
        character_predictions = []
        
        for i, char_array in enumerate(char_arrays):
            label, confidence, scores = self.recognize_character(char_array)
            captcha_text += label
            character_predictions.append((label, confidence))
            
            print(f"Character {i}: '{label}' (confidence: {confidence:.3f})")
        
        print(f"Captcha text: {captcha_text}")
        return captcha_text, character_predictions
    
    def get_recognition_stats(self) -> Dict:
        """Get statistics about the recognizer.
        
        Returns:
            Dictionary with recognition statistics
        """
        return {
            "templates_loaded": len(self.templates),
            "template_labels": sorted(self.templates.keys()),
            "template_info": self.template_info
        }
