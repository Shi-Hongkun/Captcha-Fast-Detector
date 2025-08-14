"""
Dataset Organizer module.

This module organizes segmented character files into a labeled dataset structure
by mapping them to ground truth labels from the output files.

The mapping rule: inputNN_segmentK.txt → K-th character from data/output/outputNN.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import shutil


class DatasetOrganizer:
    """Organize segmented character files into labeled dataset structure.
    
    Maps segmented characters to their ground truth labels and organizes them
    into a structured dataset for template building.
    """
    
    def __init__(self, segmented_dir: str | Path, output_dir: str | Path):
        """Initialize the dataset organizer.
        
        Args:
            segmented_dir: Directory containing segmented character files
            output_dir: Directory containing ground truth label files
        """
        self.segmented_dir = Path(segmented_dir)
        self.output_dir = Path(output_dir)
        
        if not self.segmented_dir.exists():
            raise ValueError(f"Segmented directory does not exist: {segmented_dir}")
        if not self.output_dir.exists():
            raise ValueError(f"Output directory does not exist: {output_dir}")
    
    def organize_dataset(self, target_dir: str | Path) -> None:
        """Organize segmented characters into labeled dataset.
        
        Args:
            target_dir: Directory to create the labeled dataset
        """
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Find all output files (ground truth)
        output_files = sorted(self.output_dir.glob("output*.txt"))
        
        organized_count = 0
        error_count = 0
        
        for output_file in output_files:
            try:
                # Read ground truth label
                with open(output_file, 'r') as f:
                    content = f.read().strip()
                
                # Extract the 5-character label (first line)
                lines = content.split('\n')
                if not lines or not lines[0]:
                    print(f"Warning: Empty or invalid output file: {output_file.name}")
                    error_count += 1
                    continue
                
                label = lines[0].strip()
                if len(label) != 5:
                    print(f"Warning: Invalid label length in {output_file.name}: '{label}'")
                    error_count += 1
                    continue
                
                # Extract captcha number from filename (e.g., "output00.txt" → "00")
                captcha_num = output_file.stem.replace('output', '')
                
                # Process each character segment
                for segment_idx in range(5):
                    char_label = label[segment_idx]
                    
                    # Create label directory if it doesn't exist
                    label_dir = target_path / char_label
                    label_dir.mkdir(exist_ok=True)
                    
                    # Find corresponding segment file
                    segment_file = self.segmented_dir / f"input{captcha_num}" / f"input{captcha_num}_segment{segment_idx}.txt"
                    
                    if not segment_file.exists():
                        print(f"Warning: Segment file not found: {segment_file}")
                        error_count += 1
                        continue
                    
                    # Create labeled filename
                    # Count existing files for this label to generate unique name
                    existing_files = list(label_dir.glob(f"{char_label}_sample*.txt"))
                    sample_num = len(existing_files)
                    labeled_filename = f"{char_label}_sample{sample_num}.txt"
                    
                    # Copy file to labeled dataset
                    target_file = label_dir / labeled_filename
                    shutil.copy2(segment_file, target_file)
                    organized_count += 1
                    
                    print(f"Organized: {segment_file.name} → {char_label}/{labeled_filename}")
                
            except Exception as e:
                print(f"Error processing {output_file.name}: {e}")
                error_count += 1
        
        # Summary
        print(f"\nDataset organization complete:")
        print(f"  Successfully organized: {organized_count} character files")
        if error_count > 0:
            print(f"  Errors encountered: {error_count} files")
        
        # Report label distribution
        self._report_label_distribution(target_path)
    
    def _report_label_distribution(self, target_dir: Path) -> None:
        """Report the distribution of labels in the organized dataset.
        
        Args:
            target_dir: Directory containing the organized dataset
        """
        print(f"\nLabel distribution:")
        
        # Find all label directories
        label_dirs = [d for d in target_dir.iterdir() if d.is_dir()]
        label_dirs.sort()
        
        for label_dir in label_dirs:
            label = label_dir.name
            sample_count = len(list(label_dir.glob("*.txt")))
            print(f"  {label}: {sample_count} samples")
    
    @staticmethod
    def validate_organization(target_dir: str | Path) -> bool:
        """Validate that the organized dataset is complete and correct.
        
        Args:
            target_dir: Directory containing the organized dataset
            
        Returns:
            True if validation passes, False otherwise
        """
        target_path = Path(target_dir)
        
        if not target_path.exists():
            print(f"Validation failed: Target directory does not exist: {target_dir}")
            return False
        
        # Check that all expected labels exist
        expected_labels = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        existing_labels = {d.name for d in target_path.iterdir() if d.is_dir()}
        
        missing_labels = expected_labels - existing_labels
        if missing_labels:
            print(f"Validation failed: Missing labels: {missing_labels}")
            return False
        
        # Check that each label has at least one sample
        for label in sorted(expected_labels):
            label_dir = target_path / label
            sample_count = len(list(label_dir.glob("*.txt")))
            if sample_count == 0:
                print(f"Validation failed: Label {label} has no samples")
                return False
        
        print("Dataset validation passed!")
        return True
