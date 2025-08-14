"""
Evaluation Framework module.

This module implements leave-one-captcha-out validation to evaluate
the performance of the character recognition system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import numpy as np
from .dataset_organizer import DatasetOrganizer
from .template_builder import TemplateBuilder
from .character_recognizer import CharacterRecognizer


class EvaluationFramework:
    """Evaluate character recognition performance using leave-one-captcha-out validation."""
    
    def __init__(self, segmented_dir: str | Path, output_dir: str | Path, 
                 templates_dir: str | Path):
        """Initialize the evaluation framework.
        
        Args:
            segmented_dir: Directory containing segmented character files
            output_dir: Directory containing ground truth labels
            templates_dir: Directory containing character templates
        """
        self.segmented_dir = Path(segmented_dir)
        self.output_dir = Path(output_dir)
        self.templates_dir = Path(templates_dir)
        
        # Validation
        for path, name in [(segmented_dir, "segmented"), (output_dir, "output"), (templates_dir, "templates")]:
            if not Path(path).exists():
                raise ValueError(f"{name} directory does not exist: {path}")
    
    def leave_one_captcha_out_validation(self, output_dir: str | Path) -> Dict[str, Any]:
        """Perform leave-one-captcha-out validation.
        
        Args:
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary with evaluation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Starting leave-one-captcha-out validation...")
        
        # Find all captcha files
        output_files = sorted(self.output_dir.glob("output*.txt"))
        captcha_count = len(output_files)
        
        print(f"Found {captcha_count} captchas for validation")
        
        results = {
            "validation_type": "leave_one_captcha_out",
            "total_captchas": captcha_count,
            "folds": [],
            "summary": {}
        }
        
        # Perform validation for each fold
        for fold_idx, test_captcha in enumerate(output_files):
            print(f"\n--- Fold {fold_idx + 1}/{captcha_count}: {test_captcha.name} ---")
            
            fold_result = self._evaluate_single_fold(fold_idx, test_captcha, output_path)
            results["folds"].append(fold_result)
            
            print(f"Fold {fold_idx + 1} results:")
            print(f"  Captcha: {fold_result['captcha_name']}")
            print(f"  Ground truth: {fold_result['ground_truth']}")
            print(f"  Prediction: {fold_result['prediction']}")
            print(f"  Correct: {fold_result['correct']}")
            print(f"  Character accuracy: {fold_result['char_accuracy']:.3f}")
        
        # Compute summary statistics
        results["summary"] = self._compute_summary(results["folds"])
        
        # Save results
        results_file = output_path / "validation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self._print_summary(results["summary"])
        
        print(f"\nValidation results saved to: {results_file}")
        return results
    
    def _evaluate_single_fold(self, fold_idx: int, test_captcha: Path, 
                             output_path: Path) -> Dict[str, Any]:
        """Evaluate a single fold (one captcha).
        
        Args:
            fold_idx: Index of the current fold
            test_captcha: Path to the test captcha file
            output_path: Directory to save fold results
            
        Returns:
            Dictionary with fold evaluation results
        """
        # Extract captcha number
        captcha_num = test_captcha.stem.replace('output', '')
        
        # Read ground truth
        with open(test_captcha, 'r') as f:
            ground_truth = f.read().strip().split('\n')[0]
        
        # Create temporary directories for this fold
        temp_dir = output_path / f"fold_{fold_idx:02d}"
        temp_dir.mkdir(exist_ok=True)
        
        # Build templates using all captchas EXCEPT the test one
        training_chars = self._get_training_characters(captcha_num)
        if not training_chars:
            return {
                "fold_idx": fold_idx,
                "captcha_name": test_captcha.name,
                "ground_truth": ground_truth,
                "error": "No training characters available"
            }
        
        # Build templates from training data
        templates_dir = temp_dir / "templates"
        self._build_templates_from_chars(training_chars, templates_dir)
        
        # Load recognizer with new templates
        recognizer = CharacterRecognizer(templates_dir)
        
        # Test on the held-out captcha
        test_chars = self._load_test_characters(captcha_num)
        if len(test_chars) != 5:
            return {
                "fold_idx": fold_idx,
                "captcha_name": test_captcha.name,
                "ground_truth": ground_truth,
                "error": f"Expected 5 characters, got {len(test_chars)}"
            }
        
        # Recognize characters
        prediction, char_predictions = recognizer.recognize_captcha(test_chars)
        
        # Evaluate results
        correct = prediction == ground_truth
        char_correct = sum(1 for i, (pred, _) in enumerate(char_predictions) 
                          if pred == ground_truth[i])
        char_accuracy = char_correct / 5.0
        
        # Save fold results
        fold_result = {
            "fold_idx": fold_idx,
            "captcha_name": test_captcha.name,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": correct,
            "char_accuracy": char_accuracy,
            "char_predictions": char_predictions,
            "training_samples": len(training_chars)
        }
        
        # Save fold details
        fold_file = temp_dir / "fold_result.json"
        with open(fold_file, 'w', encoding='utf-8') as f:
            json.dump(fold_result, f, indent=2)
        
        return fold_result
    
    def _get_training_characters(self, exclude_captcha: str) -> List[Tuple[str, np.ndarray]]:
        """Get training characters from all captchas except the excluded one.
        
        Args:
            exclude_captcha: Captcha number to exclude from training
            
        Returns:
            List of (label, character_array) tuples
        """
        training_chars = []
        
        # Find all output files
        output_files = sorted(self.output_dir.glob("output*.txt"))
        
        for output_file in output_files:
            captcha_num = output_file.stem.replace('output', '')
            
            # Skip the excluded captcha
            if captcha_num == exclude_captcha:
                continue
            
            # Read ground truth
            with open(output_file, 'r') as f:
                ground_truth = f.read().strip().split('\n')[0]
            
            # Load character segments
            for segment_idx in range(5):
                char_label = ground_truth[segment_idx]
                segment_file = self.segmented_dir / f"input{captcha_num}" / f"input{captcha_num}_segment{segment_idx}.txt"
                
                if segment_file.exists():
                    try:
                        char_array = self._load_txt_as_array(segment_file)
                        training_chars.append((char_label, char_array))
                    except Exception as e:
                        print(f"Warning: Failed to load {segment_file.name}: {e}")
        
        return training_chars
    
    def _load_test_characters(self, captcha_num: str) -> List[np.ndarray]:
        """Load test characters from a specific captcha.
        
        Args:
            captcha_num: Captcha number to load
            
        Returns:
            List of character arrays
        """
        test_chars = []
        
        for segment_idx in range(5):
            segment_file = self.segmented_dir / f"input{captcha_num}" / f"input{captcha_num}_segment{segment_idx}.txt"
            
            if segment_file.exists():
                try:
                    char_array = self._load_txt_as_array(segment_file)
                    test_chars.append(char_array)
                except Exception as e:
                    print(f"Warning: Failed to load {segment_file.name}: {e}")
        
        return test_chars
    
    def _build_templates_from_chars(self, training_chars: List[Tuple[str, np.ndarray]], 
                                  templates_dir: Path) -> None:
        """Build templates from training character arrays.
        
        Args:
            training_chars: List of (label, character_array) tuples
            templates_dir: Directory to save templates
        """
        # Group characters by label
        char_groups = {}
        for label, char_array in training_chars:
            if label not in char_groups:
                char_groups[label] = []
            char_groups[label].append(char_array)
        
        # Build templates for each label
        for label, char_arrays in char_groups.items():
            if len(char_arrays) == 0:
                continue
            
            # Canonicalize all characters to same size
            canonicalized = []
            for char_array in char_arrays:
                canonicalized.append(self._canonicalize_character(char_array, (12, 9)))
            
            # Average the canonicalized characters
            template = np.stack(canonicalized, axis=0).mean(axis=0).astype(np.uint8)
            
            # Save template
            template_file = templates_dir / f"{label}.txt"
            templates_dir.mkdir(parents=True, exist_ok=True)
            self._save_array_as_txt(template, template_file)
    
    def _canonicalize_character(self, char_array: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
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
            pad_left = (target_width - width) // 2
            padded = np.zeros((height, target_width), dtype=np.uint8)
            padded[:, pad_left:pad_left + width] = gray_array
            gray_array = padded
        elif width > target_width:
            start = (width - target_width) // 2
            gray_array = gray_array[:, start:start + target_width]
        
        # Handle height
        if height != target_height:
            if height < target_height:
                pad_top = (target_height - height) // 2
                padded = np.zeros((target_height, target_width), dtype=np.uint8)
                padded[pad_top:pad_top + height, :] = gray_array
                gray_array = padded
            else:
                start = (height - target_height) // 2
                gray_array = gray_array[start:start + target_height, :]
        
        return gray_array
    
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
                if ',' in line.split()[0]:  # RGB format
                    for pixel in line.split():
                        r, g, b = map(int, pixel.split(','))
                        pixels.append((r + g + b) // 3)
                else:  # Grayscale format
                    pixels.extend(map(int, line.split()))
        
        # Convert to numpy array and reshape
        array = np.array(pixels, dtype=np.uint8)
        return array.reshape(height, width)
    
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
                    row_pixels.append(f"{pixel_value},{pixel_value},{pixel_value}")
                f.write(" ".join(row_pixels) + "\n")
    
    def _compute_summary(self, folds: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics from fold results.
        
        Args:
            folds: List of fold results
            
        Returns:
            Dictionary with summary statistics
        """
        if not folds:
            return {}
        
        # Overall accuracy
        total_captchas = len(folds)
        correct_captchas = sum(1 for fold in folds if fold.get('correct', False))
        overall_accuracy = correct_captchas / total_captchas
        
        # Character-level accuracy
        char_accuracies = [fold.get('char_accuracy', 0.0) for fold in folds]
        avg_char_accuracy = np.mean(char_accuracies)
        
        # Per-position accuracy
        position_correct = [0] * 5
        position_total = [0] * 5
        
        for fold in folds:
            if 'char_predictions' in fold and 'ground_truth' in fold:
                ground_truth = fold['ground_truth']
                predictions = fold['char_predictions']
                
                for i, (pred, _) in enumerate(predictions):
                    if i < len(ground_truth):
                        position_total[i] += 1
                        if pred == ground_truth[i]:
                            position_correct[i] += 1
        
        position_accuracies = []
        for correct, total in zip(position_correct, position_total):
            if total > 0:
                position_accuracies.append(correct / total)
            else:
                position_accuracies.append(0.0)
        
        # Confusion matrix (simplified)
        confusion_data = {}
        for fold in folds:
            if 'char_predictions' in fold and 'ground_truth' in fold:
                ground_truth = fold['ground_truth']
                predictions = fold['char_predictions']
                
                for i, (pred, _) in enumerate(predictions):
                    if i < len(ground_truth):
                        true_char = ground_truth[i]
                        pred_char = pred
                        
                        if true_char not in confusion_data:
                            confusion_data[true_char] = {}
                        if pred_char not in confusion_data[true_char]:
                            confusion_data[true_char][pred_char] = 0
                        
                        confusion_data[true_char][pred_char] += 1
        
        return {
            "overall_accuracy": overall_accuracy,
            "correct_captchas": correct_captchas,
            "total_captchas": total_captchas,
            "average_character_accuracy": avg_char_accuracy,
            "position_accuracies": position_accuracies,
            "confusion_matrix": confusion_data
        }
    
    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """Print evaluation summary.
        
        Args:
            summary: Summary statistics dictionary
        """
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        print(f"Overall Captcha Accuracy: {summary['overall_accuracy']:.3f} ({summary['correct_captchas']}/{summary['total_captchas']})")
        print(f"Average Character Accuracy: {summary['average_character_accuracy']:.3f}")
        
        print("\nPer-Position Character Accuracy:")
        for i, acc in enumerate(summary['position_accuracies']):
            print(f"  Position {i}: {acc:.3f}")
        
        print("\nConfusion Matrix (top misclassifications):")
        for true_char, predictions in summary['confusion_matrix'].items():
            # Find most common misclassification
            misclassifications = [(pred_char, count) for pred_char, count in predictions.items() 
                                if pred_char != true_char]
            if misclassifications:
                top_misclass = max(misclassifications, key=lambda x: x[1])
                print(f"  '{true_char}' → '{top_misclass[0]}': {top_misclass[1]} times")
        
        print("="*50)
