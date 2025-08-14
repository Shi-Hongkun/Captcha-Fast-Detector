"""
Tests for Dataset Organizer module.
"""

import tempfile
from pathlib import Path
import pytest
from captcha_detector.dataset_organizer import DatasetOrganizer


class TestDatasetOrganizer:
    """Test cases for DatasetOrganizer class."""
    
    def test_init_valid_directories(self):
        """Test initialization with valid directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock directories
            segmented_dir = Path(temp_dir) / "segmented"
            output_dir = Path(temp_dir) / "output"
            segmented_dir.mkdir()
            output_dir.mkdir()
            
            organizer = DatasetOrganizer(segmented_dir, output_dir)
            assert organizer.segmented_dir == segmented_dir
            assert organizer.output_dir == output_dir
    
    def test_init_invalid_segmented_dir(self):
        """Test initialization with invalid segmented directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()
            
            with pytest.raises(ValueError, match="Segmented directory does not exist"):
                DatasetOrganizer("nonexistent", output_dir)
    
    def test_init_invalid_output_dir(self):
        """Test initialization with invalid output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            segmented_dir = Path(temp_dir) / "segmented"
            segmented_dir.mkdir()
            
            with pytest.raises(ValueError, match="Output directory does not exist"):
                DatasetOrganizer(segmented_dir, "nonexistent")
    
    def test_organize_dataset_simple(self):
        """Test basic dataset organization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock segmented directory structure
            segmented_dir = Path(temp_dir) / "segmented"
            segmented_dir.mkdir()
            
            # Create mock input00 directory with segments
            input00_dir = segmented_dir / "input00"
            input00_dir.mkdir()
            
            # Create mock segment files
            for i in range(5):
                segment_file = input00_dir / f"input00_segment{i}.txt"
                segment_file.write_text(f"mock segment {i}")
            
            # Create mock output directory with ground truth
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()
            
            # Create mock output00.txt with label "ABCDE"
            output00_file = output_dir / "output00.txt"
            output00_file.write_text("ABCDE\n\n")
            
            # Create target directory
            target_dir = Path(temp_dir) / "char_dataset"
            
            # Organize dataset
            organizer = DatasetOrganizer(segmented_dir, output_dir)
            organizer.organize_dataset(target_dir)
            
            # Verify organization
            assert (target_dir / "A" / "A_sample0.txt").exists()
            assert (target_dir / "B" / "B_sample0.txt").exists()
            assert (target_dir / "C" / "C_sample0.txt").exists()
            assert (target_dir / "D" / "D_sample0.txt").exists()
            assert (target_dir / "E" / "E_sample0.txt").exists()
    
    def test_validate_organization_valid(self):
        """Test validation of valid organization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock organized dataset
            char_dataset_dir = Path(temp_dir) / "char_dataset"
            char_dataset_dir.mkdir()
            
            # Create mock label directories with samples
            for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
                label_dir = char_dataset_dir / label
                label_dir.mkdir()
                
                # Create a mock sample file
                sample_file = label_dir / f"{label}_sample0.txt"
                sample_file.write_text("mock sample")
            
            # Test validation
            assert DatasetOrganizer.validate_organization(char_dataset_dir) is True
    
    def test_validate_organization_missing_label(self):
        """Test validation with missing label."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock organized dataset missing some labels
            char_dataset_dir = Path(temp_dir) / "char_dataset"
            char_dataset_dir.mkdir()
            
            # Only create some label directories
            for label in "ABCDEF":  # Missing most labels
                label_dir = char_dataset_dir / label
                label_dir.mkdir()
                
                # Create a mock sample file
                sample_file = label_dir / f"{label}_sample0.txt"
                sample_file.write_text("mock sample")
            
            # Test validation
            assert DatasetOrganizer.validate_organization(char_dataset_dir) is False
    
    def test_validate_organization_empty_label(self):
        """Test validation with empty label directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock organized dataset
            char_dataset_dir = Path(temp_dir) / "char_dataset"
            char_dataset_dir.mkdir()
            
            # Create label directories
            for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
                label_dir = char_dataset_dir / label
                label_dir.mkdir()
                
                # Only create sample for first label
                if label == "A":
                    sample_file = label_dir / f"{label}_sample0.txt"
                    sample_file.write_text("mock sample")
            
            # Test validation
            assert DatasetOrganizer.validate_organization(char_dataset_dir) is False
