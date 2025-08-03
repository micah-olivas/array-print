<img src="images/array_print.png" alt="Array Print" align="right"/>

# Array Print

Generates optimized print layouts for Scienion liquid-handling robots.

## Overview

Takes a `.csv` library file and generates Scienion-compatible `.fld` files with optimized biological replicate placement. Includes visualization and quality control metrics.

## Structure

- `template.ipynb` - Main workflow notebook
- `utils.py` - Core utility functions (array filling, plate conversion, file I/O)
- `viz.py` - Visualization functions (frequency plots, position maps, heatmaps)

## Usage

1. Install: `pip install -r requirements.txt`
2. Configure project settings in `template.ipynb`
3. Set input CSV path with columns: Plate, Well, Name, Rank, Block
4. Run notebook to generate `.fld` file and analysis plots

## Output

Results saved to `~/my-prints/[project-name]/` containing:
- Timestamped `.fld` print file
- Frequency and position visualization plots
- Print metrics summary
