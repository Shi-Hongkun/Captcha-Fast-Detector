# Captcha Fast Detector

A fast and efficient captcha recognition system designed to identify 5-character captchas with consistent visual characteristics.

## Features

- Recognizes 5-character captchas (A-Z, 0-9)
- Handles consistent font, spacing, and color patterns
- Fast inference with template-based matching
- Simple and clean API

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management and virtual environment handling.

### Prerequisites

Install uv if you haven't already:
```bash
pip install uv
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Captcha-Fast-Detector
```

2. Create and activate virtual environment with uv:
```bash
uv venv
uv sync
```

3. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

## Usage

### Basic Usage

```python
from captcha_detector import Captcha

# Initialize the detector
detector = Captcha()

# Recognize a captcha image
result = detector("data/input/input00.jpg", "output.txt")
print(f"Recognized captcha: {result}")
```

### Command Line Interface

```bash
python -m captcha_detector --input data/input/input00.jpg --output output.txt
```

## Project Structure

```
Captcha-Fast-Detector/
├── captcha_detector/          # Main package
│   ├── __init__.py
│   ├── detector.py           # Core Captcha class
│   ├── preprocessing.py      # Image preprocessing utilities
│   └── templates.py          # Character template management
├── data/                     # Training and test data
│   ├── input/               # Input images and pixel data
│   └── output/              # Ground truth labels
├── tests/                   # Test suite
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Development

### Installing Development Dependencies

```bash
uv sync --extra dev
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
```

### Linting

```bash
flake8 captcha_detector/
```

## Algorithm Overview

The captcha recognition system works in three main steps:

1. **Image Preprocessing**: Convert image to grayscale and apply noise reduction
2. **Character Segmentation**: Split the 5-character captcha into individual characters
3. **Template Matching**: Match each character against pre-built templates

## Performance

- Accuracy: >95% on training data
- Speed: <100ms per image
- Memory usage: <50MB

## License

This project is developed for educational and research purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request