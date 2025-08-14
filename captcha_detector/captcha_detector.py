"""
Main Captcha Detector module.

This module orchestrates the complete captcha detection pipeline:
ROI calibration → Character segmentation → Template building → Character recognition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from .roi_calibrator import ROICalibrator
from .character_segmenter import CharacterSegmenter
from .dataset_organizer import DatasetOrganizer
from .template_builder import TemplateBuilder
from .character_recognizer import CharacterRecognizer
from .evaluation import EvaluationFramework


class CaptchaDetector:
    """Main controller for the complete captcha detection pipeline."""
    
    def __init__(self, roi_file: Optional[str | Path] = None):
        """Initialize the captcha detector.
        
        Args:
            roi_file: Path to existing ROI file (optional)
        """
        self.roi_calibrator = ROICalibrator()
        self.segmenter = CharacterSegmenter()
        self.dataset_organizer = None
        self.template_builder = None
        self.character_recognizer = None
        
        # Load ROI if provided
        if roi_file:
            self.roi_bounds = self.roi_calibrator.load_roi(roi_file)
        else:
            self.roi_bounds = None
    
    def calibrate_roi(self, input_dir: str | Path, roi_file: str | Path, 
                      use_txt: bool = False) -> ROIBounds:
        """Calibrate ROI from training images.
        
        Args:
            input_dir: Directory containing training images
            roi_file: Path to save ROI bounds
            use_txt: Whether to use txt files instead of images
            
        Returns:
            ROI bounds
        """
        print("Step 1: Calibrating ROI...")
        
        if use_txt:
            self.roi_bounds = self.roi_calibrator.calibrate_from_txt(input_dir, roi_file)
        else:
            self.roi_bounds = self.roi_calibrator.calibrate_from_images(input_dir, roi_file)
        
        print(f"ROI calibration complete: {self.roi_bounds}")
        return self.roi_bounds
    
    def segment_characters(self, input_dir: str | Path, output_dir: str | Path) -> None:
        """Segment characters from ROI-cropped images.
        
        Args:
            input_dir: Directory with ROI-cropped images
            output_dir: Directory to save segmented characters
        """
        print("Step 2: Segmenting characters...")
        
        self.segmenter.segment_directory(input_dir, output_dir)
        print("Character segmentation complete")
    
    def organize_dataset(self, segmented_dir: str | Path, output_dir: str | Path, 
                        target_dir: str | Path) -> None:
        """Organize segmented characters into labeled dataset.
        
        Args:
            segmented_dir: Directory with segmented characters
            output_dir: Directory with ground truth labels
            target_dir: Directory to create labeled dataset
        """
        print("Step 3: Organizing dataset...")
        
        self.dataset_organizer = DatasetOrganizer(segmented_dir, output_dir)
        self.dataset_organizer.organize_dataset(target_dir)
        print("Dataset organization complete")
    
    def build_templates(self, char_dataset_dir: str | Path, 
                       templates_dir: str | Path) -> None:
        """Build character templates from labeled dataset.
        
        Args:
            char_dataset_dir: Directory with labeled character dataset
            templates_dir: Directory to save templates
        """
        print("Step 4: Building templates...")
        
        self.template_builder = TemplateBuilder(char_dataset_dir)
        self.template_builder.build_templates(templates_dir)
        print("Template building complete")
    
    def load_recognizer(self, templates_dir: str | Path) -> None:
        """Load the character recognizer with templates.
        
        Args:
            templates_dir: Directory containing character templates
        """
        print("Step 5: Loading character recognizer...")
        
        self.character_recognizer = CharacterRecognizer(templates_dir)
        print("Character recognizer loaded")
    
    def recognize_captcha(self, char_arrays: List[np.ndarray]) -> Tuple[str, List[Tuple[str, float]]]:
        """Recognize a complete captcha.
        
        Args:
            char_arrays: List of 5 character arrays
            
        Returns:
            Tuple of (captcha_text, character_predictions)
        """
        if not self.character_recognizer:
            raise RuntimeError("Character recognizer not loaded. Call load_recognizer() first.")
        
        return self.character_recognizer.recognize_captcha(char_arrays)
    
    def recognize_from_segments(self, segmented_dir: str | Path, captcha_num: str) -> Tuple[str, List[Tuple[str, float]]]:
        """Recognize a captcha from segmented character files.
        
        Args:
            segmented_dir: Directory with segmented characters
            captcha_num: Captcha number to recognize
            
        Returns:
            Tuple of (captcha_text, character_predictions)
        """
        if not self.character_recognizer:
            raise RuntimeError("Character recognizer not loaded. Call load_recognizer() first.")
        
        # Load character segments
        char_arrays = []
        for segment_idx in range(5):
            segment_file = Path(segmented_dir) / f"input{captcha_num}" / f"input{captcha_num}_segment{segment_idx}.txt"
            
            if not segment_file.exists():
                raise FileNotFoundError(f"Segment file not found: {segment_file}")
            
            # Load as array
            char_array = self._load_txt_as_array(segment_file)
            char_arrays.append(char_array)
        
        return self.recognize_captcha(char_arrays)
    
    def evaluate_performance(self, segmented_dir: str | Path, output_dir: str | Path, 
                           templates_dir: str | Path, results_dir: str | Path) -> Dict:
        """Evaluate recognition performance using leave-one-captcha-out validation.
        
        Args:
            segmented_dir: Directory with segmented characters
            output_dir: Directory with ground truth labels
            templates_dir: Directory with character templates
            results_dir: Directory to save evaluation results
            
        Returns:
            Dictionary with evaluation results
        """
        print("Step 6: Evaluating performance...")
        
        evaluator = EvaluationFramework(segmented_dir, output_dir, templates_dir)
        results = evaluator.leave_one_captcha_out_validation(results_dir)
        
        print("Performance evaluation complete")
        return results
    
    def run_complete_pipeline(self, input_dir: str | Path, output_dir: str | Path,
                             roi_file: str | Path, use_txt: bool = False) -> Dict:
        """Run the complete captcha detection pipeline.
        
        Args:
            input_dir: Directory with raw input images
            output_dir: Directory with ground truth labels
            roi_file: Path to save/load ROI bounds
            use_txt: Whether to use txt files for ROI calibration
            
        Returns:
            Dictionary with pipeline results
        """
        print("Starting complete captcha detection pipeline...")
        print("="*60)
        
        results = {
            "pipeline_steps": [],
            "final_accuracy": None,
            "error": None
        }
        
        try:
            # Step 1: ROI Calibration
            print("\n1. ROI Calibration")
            roi_bounds = self.calibrate_roi(input_dir, roi_file, use_txt)
            results["pipeline_steps"].append({
                "step": "roi_calibration",
                "status": "completed",
                "roi_bounds": str(roi_bounds)
            })
            
            # Step 2: Character Segmentation
            print("\n2. Character Segmentation")
            cropped_dir = Path("data/input_cropped")
            segmented_dir = Path("data/input_segmented")
            
            # Apply ROI to create cropped images
            self.roi_calibrator.batch_apply_roi(roi_file, input_dir, cropped_dir)
            
            # Segment characters
            self.segment_characters(cropped_dir, segmented_dir)
            results["pipeline_steps"].append({
                "step": "character_segmentation",
                "status": "completed",
                "segmented_dir": str(segmented_dir)
            })
            
            # Step 3: Dataset Organization
            print("\n3. Dataset Organization")
            char_dataset_dir = Path("data/char_dataset")
            self.organize_dataset(segmented_dir, output_dir, char_dataset_dir)
            results["pipeline_steps"].append({
                "step": "dataset_organization",
                "status": "completed",
                "char_dataset_dir": str(char_dataset_dir)
            })
            
            # Step 4: Template Building
            print("\n4. Template Building")
            templates_dir = Path("artifacts/templates")
            self.build_templates(char_dataset_dir, templates_dir)
            results["pipeline_steps"].append({
                "step": "template_building",
                "status": "completed",
                "templates_dir": str(templates_dir)
            })
            
            # Step 5: Load Recognizer
            print("\n5. Loading Character Recognizer")
            self.load_recognizer(templates_dir)
            results["pipeline_steps"].append({
                "step": "recognizer_loading",
                "status": "completed"
            })
            
            # Step 6: Performance Evaluation
            print("\n6. Performance Evaluation")
            results_dir = Path("artifacts/evaluation")
            evaluation_results = self.evaluate_performance(
                segmented_dir, output_dir, templates_dir, results_dir
            )
            
            results["pipeline_steps"].append({
                "step": "performance_evaluation",
                "status": "completed",
                "results_dir": str(results_dir)
            })
            
            # Final results
            results["final_accuracy"] = evaluation_results["summary"]["overall_accuracy"]
            results["evaluation_summary"] = evaluation_results["summary"]
            
            print("\n" + "="*60)
            print("PIPELINE COMPLETE!")
            print(f"Final Accuracy: {results['final_accuracy']:.3f}")
            print("="*60)
            
        except Exception as e:
            error_msg = f"Pipeline failed at step {len(results['pipeline_steps']) + 1}: {str(e)}"
            print(f"\nERROR: {error_msg}")
            results["error"] = error_msg
        
        return results
    
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
    
    def get_pipeline_status(self) -> Dict:
        """Get the current status of the pipeline components.
        
        Returns:
            Dictionary with pipeline component status
        """
        return {
            "roi_bounds": str(self.roi_bounds) if self.roi_bounds else None,
            "segmenter_loaded": self.segmenter is not None,
            "dataset_organizer_loaded": self.dataset_organizer is not None,
            "template_builder_loaded": self.template_builder is not None,
            "character_recognizer_loaded": self.character_recognizer is not None,
            "recognizer_stats": self.character_recognizer.get_recognition_stats() if self.character_recognizer else None
        }
