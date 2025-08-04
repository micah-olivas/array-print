"""
Array Print

A Python package for generating Scienion-compatible array print files from MITOMI-assayable libraries.

This package takes a CSV containing a MITOMI-assayable library in a tabular format and returns:
- A Scienion-compatible .fld file
- A CSV record of the array layout
- Visualization plots for array analysis
"""

__version__ = "0.1.0"
__author__ = "Micah Olivas"

# Import main functions for easy access
from .utils import (
    generate_print_array,
    get_print_metrics,
    print_metrics_summary,
    create_project_directory,
    write_fld,
    display_fld,
    get_96_to_384,
    fill_array
)

from .viz import (
    plot_mutant_frequency,
    plot_mutant_position,
    plot_plate_layouts,
    plot_array_heatmap,
    modify_array_column
)

# Define what gets imported with "from array_print import *"
__all__ = [
    # Core functionality
    "generate_print_array",
    "get_print_metrics", 
    "print_metrics_summary",
    "create_project_directory",
    "write_fld",
    "display_fld",
    "get_96_to_384",
    "fill_array",
    # Visualization
    "plot_mutant_frequency",
    "plot_mutant_position",
    "plot_plate_layouts",
    "plot_array_heatmap",
    "modify_array_column"
]