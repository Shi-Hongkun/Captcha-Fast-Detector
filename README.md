# Captcha Fast Detector

A deterministic captcha recognition system using ROI calibration, character segmentation, template-based recognition, and evaluation.

## Performance Results

**System Performance (Leave-One-Captcha-Out Validation):**
- **Overall Captcha Accuracy: 83.33%** (20/24 captchas)
- **Character-Level Accuracy: 96.67%**
- **Position Accuracy:** Position 1: 91.67%, Position 2: 100%, Position 3: 95.83%, Position 4: 95.83%, Position 5: 100%

The system successfully recognizes captchas with high confidence scores (typically >0.95) and demonstrates robust performance across different character positions.

## Key Features

- **Robust ROI Calibration**: Adaptive ROI detection that works with arbitrary image sizes - no fixed dimensions required
- **Character Segmentation**: Divide ROI into 5 equal-width character segments with intelligent remainder distribution
- **Template Building**: Create character templates from segmented training data with background-aware padding
- **Character Recognition**: Recognize characters using template matching (NCC/MAE algorithms)
- **Evaluation Framework**: Comprehensive evaluation using leave-one-captcha-out validation
- **Main Controller**: Orchestrate complete pipeline from ROI to evaluation

## System Architecture

```
Training Phase:
┌────────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐
│ ROI Calibrator │──→ │ Segmentation │──→ │ Organization │──→ │ Template Builder   │
│ (Learn ROI)    │    │ (5 segments) │    │ (Label data) │    │ (Create templates) │
└────────────────┘    └──────────────┘    └──────────────┘    └────────────────────┘

Application Phase:
┌───────────────┐   ┌──────────────┐   ┌─────────────────┐   ┌───────────────┐
│ ROI Calibrator│──→│ Segmentation │──→│ Recognition     │──→│ Result Output │
│ (Apply ROI)   │   │ (5 segments) │   │ (Template match)│   │ (Captcha text)│
└───────────────┘   └──────────────┘   └─────────────────┘   └───────────────┘
```

## Project Structure

```
captcha_detector/
├── roi_calibrator.py      # ROI calibration and cropping
├── character_segmenter.py # Character segmentation
├── dataset_organizer.py   # Dataset organization
├── template_builder.py    # Template building
├── character_recognizer.py # Character recognition
├── evaluation.py          # Evaluation framework
├── captcha_detector.py    # Main controller
└── __main__.py           # CLI interface

data/
├── input/                # Original captcha images and txt files
├── input_cropped/        # ROI-cropped images and txt files
├── input_segmented/      # Segmented character files
├── char_dataset/         # Organized character dataset
└── output/              # Ground truth labels

artifacts/
├── roi/                 # ROI calibration results
├── templates/           # Character templates
└── evaluation/          # Evaluation results
```

## Method Overview

1. **ROI Calibration**: Learn fixed ROI from training images using pixel variance analysis - works with arbitrary image sizes
2. **Character Segmentation**: Divide ROI into 5 equal-width segments with robust remainder distribution
3. **Template Building**: Create templates by averaging canonicalized character samples with background-aware padding
4. **Character Recognition**: Recognize characters using template matching (NCC/MAE)
5. **Evaluation**: Evaluate performance using leave-one-captcha-out validation

## Data Format

All TXT files use consistent `(height, width)` format:
- First line: `height width`
- Subsequent lines: RGB pixel values as `R,G,B R,G,B ...`

## Installation

```bash
uv sync
```

## Usage

### ROI Calibration
```bash
# Calibrate ROI from images
uv run python -m captcha_detector calibrate --input-dir data/input --roi-json artifacts/roi/roi.json

# Calibrate ROI from txt files
uv run python -m captcha_detector calibrate --input-dir data/input --roi-json artifacts/roi/roi.json --use-txt
```

### ROI Application
```bash
# Apply ROI to single image
uv run python -m captcha_detector apply --roi-json artifacts/roi/roi.json --input input.jpg --output cropped.jpg

# Apply ROI to paired jpg+txt files
uv run python -m captcha_detector apply-paired --roi-json artifacts/roi/roi.json --input-jpg input.jpg --input-txt input.txt --output-jpg cropped.jpg --output-txt cropped.txt

# Batch apply ROI to directory
uv run python -m captcha_detector batch --roi-json artifacts/roi/roi.json --input-dir data/input --output-dir data/input_cropped

# Batch apply ROI to paired files
uv run python -m captcha_detector batch --roi-json artifacts/roi/roi.json --input-dir data/input --output-dir data/input_cropped --paired
```

### Character Segmentation
```bash
# Segment ROI-cropped images into characters
uv run python -m captcha_detector segment --input-dir data/input_cropped --output-dir data/input_segmented
```

### Dataset Organization
```bash
# Organize segmented characters into labeled dataset
uv run python -m captcha_detector organize --segmented-dir data/input_segmented --ground-truth-dir data/output --target-dir data/char_dataset
```

### Template Building
```bash
# Build character templates (default 12x9)
uv run python -m captcha_detector build-templates --char-dataset-dir data/char_dataset --templates-dir artifacts/templates

# Build templates with custom size
uv run python -m captcha_detector build-templates --char-dataset-dir data/char_dataset --templates-dir artifacts/templates --target-height 12 --target-width 9
```

### Evaluation
```bash
# Evaluate system performance
uv run python -m captcha_detector evaluate --segmented-dir data/input_segmented --ground-truth-dir data/output --templates-dir artifacts/templates --output-dir artifacts/evaluation
```

### Complete Pipeline Example
```bash
# Step 1: Calibrate ROI
uv run python -m captcha_detector calibrate --input-dir data/input --roi-json artifacts/roi/roi.json --use-txt

# Step 2: Apply ROI cropping
uv run python -m captcha_detector batch --roi-json artifacts/roi/roi.json --input-dir data/input --output-dir data/input_cropped --paired

# Step 3: Segment characters
uv run python -m captcha_detector segment --input-dir data/input_cropped --output-dir data/input_segmented

# Step 4: Organize dataset
uv run python -m captcha_detector organize --segmented-dir data/input_segmented --ground-truth-dir data/output --target-dir data/char_dataset

# Step 5: Build templates
uv run python -m captcha_detector build-templates --char-dataset-dir data/char_dataset --templates-dir artifacts/templates

# Step 6: Evaluate performance
uv run python -m captcha_detector evaluate --segmented-dir data/input_segmented --ground-truth-dir data/output --templates-dir artifacts/templates --output-dir artifacts/evaluation
```

## Development

```bash
# Run tests
uv run pytest

# Run with verbose output
uv run pytest -v
```

## Current Status

- **ROI Calibration**: Implemented and tested - robust to arbitrary image sizes
- **Character Segmentation**: Implemented and tested with equal-width distribution
- **Template Building**: Implemented with background-aware padding
- **Character Recognition**: Implemented with NCC/MAE algorithms
- **Evaluation Framework**: Implemented with leave-one-captcha-out validation
- **Main Controller**: Implemented for complete pipeline orchestration

**Performance**: 83.33% captcha accuracy achieved through systematic dimension consistency fixes and robust segmentation algorithms.