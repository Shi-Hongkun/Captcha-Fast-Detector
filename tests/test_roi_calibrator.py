"""
pytest tests for ROI Calibrator module.

Run with: pytest test_roi_calibrator.py -v
"""

import json
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil

from captcha_detector.roi_calibrator import ROICalibrator, ROIBounds


class TestROIBounds:
    """Test ROIBounds dataclass."""
    
    def test_roi_bounds_creation(self):
        bounds = ROIBounds(top=10, bottom=50, left=5, right=25)
        assert bounds.top == 10
        assert bounds.bottom == 50
        assert bounds.left == 5
        assert bounds.right == 25
    
    def test_roi_bounds_as_tuple(self):
        bounds = ROIBounds(top=10, bottom=50, left=5, right=25)
        assert bounds.as_tuple() == (10, 50, 5, 25)
    
    def test_roi_bounds_dimensions(self):
        bounds = ROIBounds(top=10, bottom=50, left=5, right=25)
        assert bounds.height == 40
        assert bounds.width == 20
    
    def test_roi_bounds_zero_dimensions(self):
        bounds = ROIBounds(top=10, bottom=10, left=5, right=5)
        assert bounds.height == 0
        assert bounds.width == 0
    
    def test_roi_bounds_negative_dimensions(self):
        bounds = ROIBounds(top=50, bottom=10, left=25, right=5)
        assert bounds.height == 0  # max(0, negative) = 0
        assert bounds.width == 0


class TestROICalibrator:
    """Test ROI Calibrator main functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_images(self, temp_dir):
        """Create sample captcha-like images for testing."""
        paths = []
        # Create 5 sample images with consistent borders
        for i in range(5):
            # Create 60x30 image (typical captcha size)
            img = Image.new('L', (60, 30), color=255)  # White background
            pixels = img.load()
            
            # Add consistent border (first 5 and last 5 pixels in each dimension)
            # Add some "text" in the center that varies between images
            for y in range(30):
                for x in range(60):
                    if 5 <= y <= 24 and 10 <= x <= 49:
                        # Text region - add some variation
                        if (x + y + i) % 3 == 0:
                            pixels[x, y] = 0  # Black text
                    # Borders remain white (255)
            
            path = temp_dir / f"test_{i:02d}.png"
            img.save(path)
            paths.append(path)
        
        return paths
    
    @pytest.fixture
    def sample_txt_files(self, temp_dir):
        """Create sample txt files for testing."""
        paths = []
        width, height = 30, 20
        
        for i in range(3):
            path = temp_dir / f"test_{i:02d}.txt"
            
            with open(path, 'w') as f:
                f.write(f"{width} {height}\n")
                
                # Create consistent pattern with some variation
                for row in range(height):
                    row_pixels = []
                    for col in range(width):
                        # Border areas (consistent)
                        if row < 3 or row >= height-3 or col < 3 or col >= width-3:
                            r = g = b = 255  # White border
                        else:
                            # Text area (varies by image)
                            base = 100 + i * 50  # Different base for each image
                            r = g = b = base + (row + col) % 50
                        
                        row_pixels.append(f"{r},{g},{b}")
                    
                    f.write(" ".join(row_pixels) + "\n")
            
            paths.append(path)
        
        return paths
    
    def test_calibrator_initialization(self):
        cal = ROICalibrator()
        assert cal.expected_size == (60, 30)
        assert cal.roi_bounds is None
    
    def test_calibrator_custom_size(self):
        cal = ROICalibrator(expected_size=(80, 40))
        assert cal.expected_size == (80, 40)
    
    def test_calibrate_from_images(self, sample_images):
        cal = ROICalibrator()
        bounds = cal.calibrate_from_dir(sample_images[0].parent)
        
        assert isinstance(bounds, ROIBounds)
        assert bounds.top >= 0
        assert bounds.bottom <= 30
        assert bounds.left >= 0
        assert bounds.right <= 60
        assert bounds.height > 0
        assert bounds.width > 0
    
    def test_calibrate_from_txt_files(self, sample_txt_files):
        cal = ROICalibrator()
        bounds = cal.calibrate_from_txt_dir(sample_txt_files[0].parent)
        
        assert isinstance(bounds, ROIBounds)
        assert bounds.top >= 0
        assert bounds.bottom <= 20
        assert bounds.left >= 0
        assert bounds.right <= 30
    
    def test_save_and_load_bounds(self, temp_dir):
        cal = ROICalibrator()
        original_bounds = ROIBounds(top=5, bottom=25, left=10, right=50)
        cal.roi_bounds = original_bounds
        
        json_path = temp_dir / "roi.json"
        cal.save_bounds(json_path)
        
        assert json_path.exists()
        
        # Load with new calibrator
        cal2 = ROICalibrator()
        loaded_bounds = cal2.load_bounds(json_path)
        
        assert loaded_bounds.as_tuple() == original_bounds.as_tuple()
    
    def test_extract_roi_array(self):
        cal = ROICalibrator()
        cal.roi_bounds = ROIBounds(top=5, bottom=15, left=10, right=20)
        
        # Create test array
        img_array = np.random.randint(0, 256, (30, 60), dtype=np.uint8)
        
        cropped = cal.extract_roi_array(img_array)
        
        assert cropped.shape == (10, 10)  # (15-5, 20-10)
        np.testing.assert_array_equal(cropped, img_array[5:15, 10:20])
    
    def test_txt_file_parsing(self, temp_dir):
        """Test parsing of txt file format."""
        cal = ROICalibrator()
        
        # Create a simple test txt file
        txt_path = temp_dir / "test.txt"
        with open(txt_path, 'w') as f:
            f.write("3 2\n")  # 3x2 image
            f.write("255,0,0 0,255,0 0,0,255\n")  # Red, Green, Blue
            f.write("128,128,128 64,64,64 192,192,192\n")  # Gray pixels
        
        img_array = cal._load_txt_as_array(txt_path)
        
        assert img_array.shape == (2, 3, 3)  # height=2, width=3, channels=3
        
        # Check first row
        np.testing.assert_array_equal(img_array[0, 0], [255, 0, 0])    # Red
        np.testing.assert_array_equal(img_array[0, 1], [0, 255, 0])    # Green
        np.testing.assert_array_equal(img_array[0, 2], [0, 0, 255])    # Blue
        
        # Check second row
        np.testing.assert_array_equal(img_array[1, 0], [128, 128, 128])  # Gray
        np.testing.assert_array_equal(img_array[1, 1], [64, 64, 64])     # Dark gray
        np.testing.assert_array_equal(img_array[1, 2], [192, 192, 192])  # Light gray
    
    def test_txt_file_save_load_roundtrip(self, temp_dir):
        """Test saving and loading txt files maintains data integrity."""
        cal = ROICalibrator()
        
        # Create original array
        original = np.random.randint(0, 256, (4, 5, 3), dtype=np.uint8)
        
        # Save as txt
        txt_path = temp_dir / "roundtrip.txt"
        cal._save_array_as_txt(original, txt_path)
        
        # Load back
        loaded = cal._load_txt_as_array(txt_path)
        
        # Should be identical
        np.testing.assert_array_equal(original, loaded)
    
    def test_crop_image_file(self, sample_images, temp_dir):
        """Test cropping individual image files."""
        cal = ROICalibrator()
        cal.roi_bounds = ROIBounds(top=2, bottom=28, left=5, right=55)
        
        input_path = sample_images[0]
        output_path = temp_dir / "cropped.png"
        
        cal.crop_image_file(input_path, output_path)
        
        assert output_path.exists()
        
        # Verify cropped image dimensions
        cropped_img = Image.open(output_path)
        assert cropped_img.size == (50, 26)  # width=55-5, height=28-2
    
    def test_error_no_images_found(self, temp_dir):
        """Test error when no images found in directory."""
        cal = ROICalibrator()
        
        with pytest.raises(ValueError, match="No images found"):
            cal.calibrate_from_dir(temp_dir)
    
    def test_error_roi_not_calibrated(self):
        """Test error when trying to use ROI before calibration."""
        cal = ROICalibrator()
        
        with pytest.raises(ValueError, match="ROI is not calibrated"):
            cal.extract_roi_array(np.zeros((10, 10)))
    
    def test_error_insufficient_images(self, temp_dir):
        """Test error when insufficient images for calibration."""
        cal = ROICalibrator()
        
        # Create only one image
        img = Image.new('L', (60, 30), color=255)
        img.save(temp_dir / "single.png")
        
        with pytest.raises(ValueError, match="At least 2 images are required"):
            cal.calibrate_from_dir(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])