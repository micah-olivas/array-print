<img src="images/array_print.png" alt="Array Print" align="right"/>

# Array Print

A Python package for generating optimized print layouts for Scienion liquid-handling robots.

## Overview

Takes a `.csv` library file and generates Scienion-compatible `.fld` files with optimized biological replicate placement. Includes visualization and quality control metrics.

## Installation

### Install from source
```bash
# Clone the repository
git clone https://github.com/micaholivas/Array-Print.git
cd Array-Print

# Install the package
pip install -e .

# Or install with Jupyter notebook support
pip install -e ".[jupyter]"
```

### Requirements
- Python ≥3.8
- Core dependencies: numpy, pandas, matplotlib, seaborn
- Optional: Jupyter notebook support

## Package Structure

```
array_print/
├── __init__.py     # Main package with exported functions
├── utils.py        # Core utility functions (array generation, file I/O)
└── viz.py          # Visualization functions (plots, heatmaps)
```

## Usage

### In Python/Jupyter
```python
import array_print as ap
import pandas as pd

# Load your library data
library_df = pd.read_csv('your_library.csv')

# Create project directory
export_dir = ap.create_project_directory('my-project')

# Generate print array
print_array = ap.generate_print_array(
    library_df=library_df,
    columns=32, rows=56,
    using_blocked_device=True,
    n_blocks=4
)

# Generate visualizations and metrics
ap.plot_mutant_frequency(print_array, export_dir)
metrics = ap.get_print_metrics(print_array)
ap.print_metrics_summary(metrics)

# Export FLD file
ap.write_fld('my-project', print_array, 32, 56, export_dir)
```

### Using the Template Notebook
1. Open `template.ipynb`
2. Configure project settings (file paths, device parameters)
3. Set input CSV path with required columns: Plate, Well, Name, Rank, Block
4. Run all cells to generate complete analysis

## Input Format

Your CSV file should contain:
- **Plate**: Plate number (int)
- **Well**: Well position (e.g., 'A1', 'B5')
- **Name**: Variant/mutant name
- **Rank**: Priority ranking for placement
- **Block**: Block assignment for blocked devices

## Output

Results saved to `~/my-prints/[project-name]/` containing:
- Timestamped `.fld` print file for Scienion robots
- Frequency and position visualization plots
- Print metrics summary and analysis

## Available Functions

### Core Functions
- `generate_print_array()` - Create optimized array layout
- `get_print_metrics()` - Calculate array statistics
- `create_project_directory()` - Set up output directories
- `write_fld()` - Export Scienion-compatible files

### Visualization Functions  
- `plot_mutant_frequency()` - Show variant distribution
- `plot_mutant_position()` - Highlight specific variants
- `plot_array_heatmap()` - Array occupancy visualization
- `plot_plate_layouts()` - 96/384-well plate layouts

### Utility Functions
- `get_96_to_384()` - Convert plate formats
- `modify_array_column()` - Array manipulation tools
