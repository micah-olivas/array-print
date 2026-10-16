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

### Scienion-readable .fld files

A bare list of positions is not read by the Scienion software; the instrument
expects a header describing the device geometry and a footer holding an
encrypted timestamp, with the whole file in CRLF and cp1252 encoding. Runs using
a known preset now write that complete file, and each write is checked for the
bytes the instrument looks for.

The device headers in `array_print/fld_format.py` come from the
[FordyceLab fork](https://github.com/FordyceLab/arrayprint) of this package,
where they were worked out against the instrument. Three are available: `PS1.8K`,
`PS1.8K_Premiere` and `2lagoon`.

### Reproducibility

Placing variants in the array is deterministic: the same library, settings and
`--seed` always produce the same layout, so a `.fld` can be regenerated exactly.
Pass a different `--seed` for a different arrangement.

Every library member is dealt an equal share of chambers, differing by at most
one where the library does not divide the array evenly. With a blocked device
the share is balanced within each block, so members of a block holding fewer
entries get slightly more chambers each.

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
array-print my-project library.csv --category-column scan_type
```

**All options:**
```
--preset            Device preset [default: mitomi-ps1.8k] | use 'custom' with --columns/--rows
--columns           Number of columns (overrides preset)
--rows              Number of rows (overrides preset)
--skip-rows         Insert blank rows between samples [default: on]
--catalytic-binning Sort by catalytic rank before filling [default: off]
--plots             Generate visualization plots [default: on]
--pinlist           Write <project>_pinlist.csv next to the .fld [default: on]
--category-column   Library column grouping variants (e.g. a scan name); each
                    group gets its own colour on the dashboard map
```

The run writes `<project>_dashboard.html`, a sheet for checking the print before
it goes on the robot. The header carries the run's figures — array size, fill,
variant count, blocks, replicate range. Below that sit two charts and a map of
the array.

The first chart is the distribution of replicate counts, on an axis anchored at
zero so it is clear how far the floor sits from nothing. The bar holding the
least-replicated variants is called out, naming them when only a few are tied.
A third chart counts the designed library members in each category, so a
category whose members never reached the array is flagged next to one that
printed in full.

The second chart gives each group's spread on its own horizontal scale — a thin
line for the full range, a thicker one for the middle half, and a dot at the
median — so a scan that came out short is visible against the distribution above
it. Hovering any bar or row gives the underlying numbers.

The fewest-replicated variants are marked in red only when they are an exception.
In a library large enough that most variants land on the same floor, that floor
is the design rather than a fault, and the count is reported without the alarm.

On the map, hovering a spot holds that variant's positions at full colour and
fades the rest, which shows whether its replicates are spread across the device
or clustered in one region. When the library carries a `Block` column the array
is printed in blocks of columns, and each block is labelled and underscored above
the map rather than ruled through it.

Pass `--category-column` when the library splits into groups — an alanine scan,
a phenylalanine scan, controls — and each group is drawn in its own colour,
taken from matplotlib's tab20 palette (without its greys) for the first 18
groups and generated beyond that, so no two groups share a colour. Without the
flag, spots are coloured by replicate count.

Everything is inline SVG and Helvetica, with no external scripts, stylesheets or
fonts, so the file opens with no network connection at all.

Alongside the sheet, a run with `--category-column` and plots enabled writes
`<project>_library_panels.png` and `.pdf`: one panel of the array per sublibrary
or sample type, each holding its own group at full colour with the rest ghosted,
plus a final panel marking the empty chambers. Use these for a figure or a
printout; the sheet is for reading on screen.

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
- `<project>_pinlist.csv` mapping each chamber to its variant (skip with `--no-pinlist`)
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
