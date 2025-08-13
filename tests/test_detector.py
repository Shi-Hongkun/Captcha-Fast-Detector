"""
Tests for the main Captcha detector class.
"""

import pytest
import tempfile
import os
from pathlib import Path

from captcha_detector import Captcha


class TestCaptcha:
    """Test cases for the Captcha class."""
    
    def test_initialization(self):
        """Test that Captcha class initializes correctly."""
        detector = Captcha()
        assert detector is not None
        assert not detector.is_trained
    
    def test_training(self):
        """Test training functionality."""
        detector = Captcha()
        
        # Test training with data directory
        data_dir = "data"
        if os.path.exists(data_dir):
            detector.train(data_dir)
            assert detector.is_trained
            assert detector.template_manager.get_template_count() > 0
    
    def test_inference_without_training(self):
        """Test that inference fails without training."""
        detector = Captcha()
        
        with pytest.raises(ValueError):
            detector.template_manager.recognize_character(None)
    
    def test_file_not_found(self):
        """Test handling of non-existent input file."""
        detector = Captcha()
        
        with pytest.raises(FileNotFoundError):
            detector("non_existent_file.jpg", "output.txt")
    
    def test_save_result(self):
        """Test result saving functionality."""
        detector = Captcha()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_output.txt")
            result = "TEST1"
            
            detector._save_result(result, output_path)
            
            # Check that file was created
            assert os.path.exists(output_path)
            
            # Check content
            with open(output_path, 'r', encoding='utf-8') as f:
                saved_result = f.read().strip()
            
            assert saved_result == result
    
    def test_save_result_with_nonexistent_directory(self):
        """Test saving result to non-existent directory."""
        detector = Captcha()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "subdir", "test_output.txt")
            result = "TEST2"
            
            # Should create directory automatically
            detector._save_result(result, output_path)
            
            assert os.path.exists(output_path)
            
            with open(output_path, 'r', encoding='utf-8') as f:
                saved_result = f.read().strip()
            
            assert saved_result == result
