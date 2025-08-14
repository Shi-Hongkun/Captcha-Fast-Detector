"""
Tests for Character Segmentation module.
"""

import numpy as np
import pytest
from captcha_detector.character_segmenter import CharacterSegmenter


class TestCharacterSegmenter:
    """Test cases for CharacterSegmenter class."""
    
    def test_init_default(self):
        """Test default initialization."""
        segmenter = CharacterSegmenter()
        assert segmenter.char_count == 5
    
    def test_init_custom(self):
        """Test custom character count initialization."""
        segmenter = CharacterSegmenter(char_count=3)
        assert segmenter.char_count == 3
    
    def test_segment_chars_basic(self):
        """Test basic character segmentation."""
        segmenter = CharacterSegmenter(char_count=5)
        
        # Create a test array: 30x60 (height x width)
        roi_array = np.random.randint(0, 255, (30, 60), dtype=np.uint8)
        
        chars = segmenter.segment_chars(roi_array)
        
        assert len(chars) == 5
        assert all(char.shape[0] == 30 for char in chars)  # Same height
        
        # Check widths: 60/5 = 12 each
        expected_widths = [12, 12, 12, 12, 12]
        actual_widths = [char.shape[1] for char in chars]
        assert actual_widths == expected_widths
    
    def test_segment_chars_with_remainder(self):
        """Test segmentation when width is not perfectly divisible."""
        segmenter = CharacterSegmenter(char_count=3)
        
        # Create a test array: 20x7 (height x width)
        roi_array = np.random.randint(0, 255, (20, 7), dtype=np.uint8)
        
        chars = segmenter.segment_chars(roi_array)
        
        assert len(chars) == 3
        assert all(char.shape[0] == 20 for char in chars)  # Same height
        
        # Check widths: 7/3 = 2 remainder 1, so [3, 2, 2]
        expected_widths = [3, 2, 2]
        actual_widths = [char.shape[1] for char in chars]
        assert actual_widths == expected_widths
    
    def test_segment_5_chars(self):
        """Test convenience method for 5-character segmentation."""
        segmenter = CharacterSegmenter()
        
        roi_array = np.random.randint(0, 255, (25, 50), dtype=np.uint8)
        chars = segmenter.segment_5_chars(roi_array)
        
        assert len(chars) == 5
        assert all(char.shape[0] == 25 for char in chars)
        
        # 50/5 = 10 each
        expected_widths = [10, 10, 10, 10, 10]
        actual_widths = [char.shape[1] for char in chars]
        assert actual_widths == expected_widths
    
    def test_get_char_dimensions(self):
        """Test character dimension calculation."""
        segmenter = CharacterSegmenter(char_count=4)
        
        roi_array = np.random.randint(0, 255, (40, 80), dtype=np.uint8)
        height, width = segmenter.get_char_dimensions(roi_array)
        
        assert height == 40
        assert width == 20  # 80/4
    
    def test_validate_segmentation_valid(self):
        """Test validation of valid segmentation."""
        segmenter = CharacterSegmenter(char_count=3)
        
        roi_array = np.random.randint(0, 255, (30, 60), dtype=np.uint8)
        chars = segmenter.segment_chars(roi_array)
        
        assert segmenter.validate_segmentation(roi_array, chars) is True
    
    def test_validate_segmentation_invalid(self):
        """Test validation of invalid segmentation."""
        segmenter = CharacterSegmenter(char_count=3)
        
        roi_array = np.random.randint(0, 255, (30, 60), dtype=np.uint8)
        
        # Create invalid chars with wrong total width (60 vs expected 60)
        # But these chars actually have total width 60, so they're valid!
        # Let's create truly invalid chars
        chars = [
            np.random.randint(0, 255, (30, 20), dtype=np.uint8),
            np.random.randint(0, 255, (30, 20), dtype=np.uint8),
            np.random.randint(0, 255, (30, 20), dtype=np.uint8)
        ]
        
        # These chars have total width 60, which matches roi_array width 60
        # So they should actually pass validation
        assert segmenter.validate_segmentation(roi_array, chars) is True
        
        # Now create truly invalid chars with wrong total width
        invalid_chars = [
            np.random.randint(0, 255, (30, 20), dtype=np.uint8),
            np.random.randint(0, 255, (30, 20), dtype=np.uint8),
            np.random.randint(0, 255, (30, 20), dtype=np.uint8),
            np.random.randint(0, 255, (30, 20), dtype=np.uint8)  # Extra char
        ]
        
        assert segmenter.validate_segmentation(roi_array, invalid_chars) is False
    
    def test_error_handling_2d_array(self):
        """Test error handling for non-2D arrays."""
        segmenter = CharacterSegmenter()
        
        # 3D array
        roi_array = np.random.randint(0, 255, (10, 20, 30), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Expected 2D array"):
            segmenter.segment_chars(roi_array)
    
    def test_error_handling_width_too_small(self):
        """Test error handling when width < char_count."""
        segmenter = CharacterSegmenter(char_count=5)
        
        # Width 3 < char_count 5
        roi_array = np.random.randint(0, 255, (10, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Width 3 must be >= char_count 5"):
            segmenter.segment_chars(roi_array)
