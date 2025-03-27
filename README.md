# Film Negative Converter

A desktop application for converting film negatives to positive images with multiple conversion presets.

## Features

- **Multiple Film Presets**: Choose from Standard, Frontier, Noritsu, and Hasselblad conversion profiles
- **CMYK Adjustments**: Fine-tune your conversions with Cyan, Magenta, Yellow, and Black sliders
- **Automatic Film Border Detection**: Automatically detects film borders and extracts the inner image area
- **Film Base Color Extraction**: Detects the film base color from borders for accurate inversion
- **Manual Color Picking**: Ability to manually pick the film base color for difficult negatives
- **macOS & Windows Compatible**: Works on major desktop platforms

## Installation

### Prerequisites

- Python 3.8+ 
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- Pillow
- Tkinter (usually comes with Python)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/kevinlin29/negConverter.git
   cd negConverter
   ```

2. Install required packages:
   ```bash
   pip install numpy opencv-python matplotlib scikit-image pillow
   ```

3. Run the application:
   ```bash
   python film_converter_ui.py
   ```

## Usage

1. **Load Image**: Click "Load Image" to select a film negative scan
2. **Process Image**: Click "Process" to convert the negative using the selected preset
3. **Adjust Settings**: 
   - Choose a preset from the dropdown (Standard, Frontier, Noritsu, Hasselblad)
   - Adjust CMYK values with the sliders if needed
4. **Pick Base Color**: For difficult negatives, use "Pick Base Color" and click on the film border
5. **Save Image**: Click "Save Converted" to save the processed image

## Project Structure

- `film_converter.py`: Core conversion algorithms and functions 
- `film_converter_ui.py`: Tkinter user interface
- `test.JPG`: Sample test image

## Acknowledgments

This project is inspired by tools like Negative Lab Pro, but implements custom algorithms for film conversion.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 