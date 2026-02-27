<img src="images/array_print.png" alt="Array Print" align="right"/>

# Array Print

A Python package for generating optimized printing inputs for Scienion liquid-handling robots.

## Overview

Takes a `.csv` library file and generates Scienion-compatible `.fld` files that optimize sample placement for arrayed microfluidics experiments (e.g. [MITOMI](https://www.fordycelab.com/research-2019)). Includes visualization and quality control metrics.

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
- Python ≥3.8 (3.11 recommended)
- Core dependencies: numpy, pandas, matplotlib, seaborn
- CLI dependencies: typer, questionary (installed automatically)
- Optional: Jupyter notebook support

## Package Structure

```
array_print/
├── __init__.py     # Main package with exported functions
├── cli.py          # Command-line interface (array-print command)
├── utils.py        # Core utility functions (array generation, file I/O)
└── viz.py          # Visualization functions (plots, heatmaps)
```

## Usage

### In Python/Jupyter
```python
import array_print as ap
import pandas as pd
import numpy as np

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
print_df = pd.DataFrame(np.flip(print_array, axis=1))
ap.write_fld('my-project', print_df, 32, 56, export_dir)
```

### From the command line

**Interactive wizard** (no arguments — guides you through each step):
```bash
array-print
```

**Non-interactive** (good for scripts):
```bash
array-print my-project library.csv
array-print my-project library.csv --preset mitomi-ps1.8k
array-print my-project library.csv --no-plots --catalytic-binning
array-print my-project library.csv --preset custom --columns 40 --rows 64
```

**All options:**
```
--preset            Device preset [default: mitomi-ps1.8k] | use 'custom' with --columns/--rows
--columns           Number of columns (overrides preset)
--rows              Number of rows (overrides preset)
--skip-rows         Insert blank rows between samples [default: on]
--catalytic-binning Sort by catalytic rank before filling [default: off]
--plots             Generate visualization plots [default: on]
```

### Using the Template Notebook
1. Open `template.ipynb`
2. Configure project settings (file paths, device parameters)
3. Set input CSV path with required columns: `Name`, `Plate`, `Well`
4. Run all cells to generate complete analysis

## Input Format

Your CSV file must contain the following columns:

| Column | Type | Description |
|---|---|---|
| `Name` | string | Variant / mutant identifier (e.g. `K304E`) |
| `Plate` | int | Source plate number |
| `Well` | string | Well position on the source plate (e.g. `A1`, `B12`) |

Optional columns:

| Column | Type | Description |
|---|---|---|
| `Block` | int (1–4) | Block assignment for blocked-device layout (required with `--blocked-device`) |
| `cat_rank` | int | Catalytic rank; lower = higher priority placement (required with `--catalytic-binning`) |

**Example:**
```csv
Name,Plate,Well
WT,1,A1
K304E,1,A2
R267E,2,A1
E345A,2,A2
```

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
