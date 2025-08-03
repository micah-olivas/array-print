"""
Array Print Utilities

Common helper functions for array printing and plate format conversions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


def fill_array(df, temp_array, catalytic_binning=False):
    """
    Fill an array uniformly with values from a dataframe.
    
    Args:
        df: DataFrame containing plate well information
        temp_array: Array to fill with values
        catalytic_binning: If True, sort by catalytic rank
    
    Returns:
        Filled array
    """
    # randomize order of df
    df = df.sample(frac=1).reset_index(drop=True)

    # sort df by catalytic rank if 'catalytic_binning' == True
    if catalytic_binning:
        # sort df by cat_rank
        df = df.sort_values(by='cat_rank', ascending=True)

    # iterate over array and fill with values from df
    for i in range(len(temp_array)):
        try:
            # randomly sample a value from df and store the Well value
            value = df.sample(n=1)['plate_well'].values[0]

            # add to temp_array
            temp_array[i] = value
        except:
            temp_array[i] = ''

    return temp_array


def get_96_to_384(well_96, plots=True):
    """
    Convert 96 well plate format to 384 well plate format.
    
    Args:
        well_96: Well position in 96-well format (e.g., '3A10')
        plots: If True, display visual comparison plots
    
    Returns:
        Well position in 384-well format
    """
    # if blank, return blank
    if well_96 == '':
        return ''
    elif well_96 == 'nan':
        return ''
    elif well_96 == None:
        return ''
        
    ## get 96 well plate information
    rows_96 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cols_96 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # first character is number between 1-4, corresponding to 384 well plate quadrant
    quadrant = int(well_96[0])
    # second character is letter between A-H, corresponding to 96 well plate row
    row_96 = well_96[1]
    # third character is number between 1-12, corresponding to 96 well plate column
    col_96 = int(well_96[2:])

    ## now get 384 well plate position
    cols_384 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    rows_384 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    quad_1_array = np.array([ str(r)+str(c) for r in rows_384[0::2] for c in cols_384[0::2] ]).reshape(8, 12) # this is the top left quadrant
    quad_2_array = np.array([ str(r)+str(c) for r in rows_384[0::2] for c in cols_384[1::2] ]).reshape(8, 12) # this is the top right quadrant
    quad_3_array = np.array([ str(r)+str(c) for r in rows_384[1::2] for c in cols_384[0::2] ]).reshape(8, 12) # this is the bottom left quadrant
    quad_4_array = np.array([ str(r)+str(c) for r in rows_384[1::2] for c in cols_384[1::2] ]).reshape(8, 12) # this is the bottom right quadrant
    quad_dict = {1: quad_1_array, 2: quad_2_array, 3: quad_3_array, 4: quad_4_array}
    
    # get 384 well plate position
    well_384 = quad_dict[quadrant][rows_96.index(row_96)][col_96-1]

    # print visual of 96 and 384
    if plots:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        array_96 = np.zeros((8, 12))
        array_96[rows_96.index(row_96)][col_96-1] = 1
        ax[0].matshow(array_96, cmap='binary')
        ax[0].set_title('96 Well Plate (Well: {})'.format(well_96), loc='left')
        ax[0].set_xticks(np.arange(0, 12, 1))
        ax[0].set_xticklabels(cols_96)
        ax[0].set_yticks(np.arange(0, 8, 1))
        ax[0].set_yticklabels(rows_96)

        # add grid to 96 well plate
        ax[0].set_xticks(np.arange(-.5, 12, 1), minor=True)
        ax[0].set_yticks(np.arange(-.5, 8, 1), minor=True)
        ax[0].grid(True, which='minor', color='black', linestyle='-', linewidth=0.5)

        array_384 = np.zeros((16, 24))
        row_384 = rows_384.index(well_384[0])
        col_384 = int(well_384[1:])
        array_384[row_384][col_384-1] = 1
        ax[1].matshow(array_384, cmap='binary')
        ax[1].set_title('384 Well Plate (Well: {})'.format(well_384), loc='left')
        ax[1].set_xticks(np.arange(0, 24, 1))
        ax[1].set_xticklabels(cols_384)
        ax[1].set_yticks(np.arange(0, 16, 1))
        ax[1].set_yticklabels(rows_384)

        # add grid to 384 well plate
        ax[1].set_xticks(np.arange(-.5, 24, 1), minor=True)
        ax[1].set_yticks(np.arange(-.5, 16, 1), minor=True)
        ax[1].grid(True, which='minor', color='black', linestyle='-', linewidth=0.5)
    
    return well_384


def display_fld(print_df, total_columns, total_rows):
    """
    Display FLD file content to console.
    
    Args:
        print_df: DataFrame containing print array
        total_columns: Number of columns in array
        total_rows: Number of rows in array
    """
    for i in range(0, total_columns):
        for j in range(0, total_rows):
            current_fld_loc = str(i + 1) + '/' + str(j + 1) # add ones to change from 0-indexing
            current_array_val = str(print_df.iloc[j][i])

            # Insert blank for NaN values
            if type(current_array_val) != str:
                current_array_val = '\t'

            array_loc_print = current_array_val
            
            # Add a plate number to only the non-blank wells
            if len(array_loc_print) >= 1:
                if array_loc_print[0] != '1':
                 array_loc_print = '1' + array_loc_print

            print(current_fld_loc + '\t' + array_loc_print + ',' + '\t' + '1,')


def write_fld(filename, print_df, total_columns, total_rows, export_dir):
    """
    Write FLD file to disk.
    
    Args:
        filename: Base filename for the FLD file
        print_df: DataFrame containing print array
        total_columns: Number of columns in array
        total_rows: Number of rows in array
        export_dir: Directory to save the file
    """
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    
    with open(os.path.join(export_dir, filename + '_' + timestamp + '.fld'), 'w') as f:
        for i in range(0, total_columns):
            for j in range(0, total_rows):
                current_fld_loc = str(i + 1) + '/' + str(j + 1) # add ones to change from 0-indexing
                current_array_val = str(print_df.iloc[j][i])

                # Insert blank for NaN values
                if type(current_array_val) != str:
                    current_array_val = '\t'

                array_loc_print = current_array_val
                
                # Add a plate number to only the non-blank wells
                if len(array_loc_print) >= 1:
                    if array_loc_print[0] != '1':
                        array_loc_print = '1' + array_loc_print

                f.write(current_fld_loc + '\t' + array_loc_print + ',' + '\t' + '1,' + '\n')


def create_project_directory(project_name, custom_dir=None):
    """
    Create project directory structure for exports.
    
    Args:
        project_name: Name of the project
        custom_dir: Custom directory path (optional)
    
    Returns:
        Path to the created project directory
    """
    if custom_dir:
        base_dir = Path(custom_dir)
    else:
        base_dir = Path.home() / "my-prints"
    
    project_dir = base_dir / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (project_dir / "metrics").mkdir(exist_ok=True)
    (project_dir / "analysis").mkdir(exist_ok=True)
    
    return str(project_dir)


def get_print_metrics(print_array):
    """
    Calculate comprehensive print metrics.
    
    Args:
        print_array: Array containing mutant positions
    
    Returns:
        Dictionary containing metrics summary
    """
    # Get frequency of each mutant in print
    unique, counts = np.unique(print_array, return_counts=True)
    print_counts = dict(zip(unique, counts))
    
    # Remove blanks from print_counts
    print_counts.pop('', None)
    
    if not print_counts:
        return {"error": "No valid entries found in print array"}
    
    count_values = list(print_counts.values())
    
    metrics = {
        "total_variants": len(print_counts),
        "total_positions": len(print_array.flatten()),
        "filled_positions": sum(count_values),
        "blank_positions": len(print_array.flatten()) - sum(count_values),
        "replicates_min": min(count_values),
        "replicates_max": max(count_values),
        "replicates_mean": np.mean(count_values),
        "replicates_std": np.std(count_values),
        "variant_counts": print_counts
    }
    
    return metrics


def print_metrics_summary(metrics):
    """
    Print formatted metrics summary.
    
    Args:
        metrics: Dictionary from get_print_metrics()
    """
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return
    
    print("=" * 50)
    print("PRINT METRICS SUMMARY")
    print("=" * 50)
    print(f"Total Variants: {metrics['total_variants']}")
    print(f"Array Positions: {metrics['total_positions']}")
    print(f"Filled Positions: {metrics['filled_positions']}")
    print(f"Blank Positions: {metrics['blank_positions']}")
    print(f"Fill Rate: {metrics['filled_positions']/metrics['total_positions']:.1%}")
    print()
    print("REPLICATES PER VARIANT:")
    print(f"  Min: {metrics['replicates_min']}")
    print(f"  Max: {metrics['replicates_max']}")
    print(f"  Mean: {metrics['replicates_mean']:.1f}")
    print(f"  Std Dev: {metrics['replicates_std']:.1f}")
    print()
    
    # Show distribution
    unique_counts = list(set(metrics['variant_counts'].values()))
    unique_counts.sort()
    print("REPLICATE DISTRIBUTION:")
    for count in unique_counts:
        variants_with_count = sum(1 for v in metrics['variant_counts'].values() if v == count)
        print(f"  {count} replicates: {variants_with_count} variants")
    print("=" * 50)


def generate_print_array(library_df, columns=32, rows=56, skip_rows=True, 
                        using_blocked_device=True, n_blocks=4, catalytic_binning=False):
    """
    Generate optimized print array from library dataframe.
    
    Args:
        library_df: DataFrame containing library data
        columns: Number of columns in device
        rows: Number of rows in device
        skip_rows: Whether to add blank rows
        using_blocked_device: Whether to use block-based assignment
        n_blocks: Number of blocks to divide array into
        catalytic_binning: Whether to sort by catalytic rank
    
    Returns:
        numpy array containing print layout
    """
    # Set size of device
    if skip_rows:
        rows = int(rows / 2)  # accounts for skipped rows
        total_chambers = int((columns * rows) / 2)  # account for skipped rows
        rows = rows / 2
    else:
        total_chambers = columns * rows

    # Get max replicates for each mutant
    num_sequences = len(library_df['Name'].unique())
    max_replicates = int(total_chambers / num_sequences)  # round down to nearest integer

    # Create print array object of strings
    print_array = np.empty(total_chambers, dtype='object')

    # Reshape print array into 2D array
    print_array = print_array.reshape(print_array.shape[0] // columns, columns)

    if using_blocked_device:
        # Create a dictionary to store the blocks
        block_dict = dict(zip(range(n_blocks), n_blocks * [0]))

        # Iterate through each block
        for i in range(n_blocks):
            # Filter library_df for each block
            block_df = library_df[library_df['Block'] == i + 1]

            # Create block_array object
            block_array = np.empty(int(rows * (columns / n_blocks)), dtype='object')

            # Fill array with values from block_df
            block_array = fill_array(block_df, block_array, catalytic_binning)

            # Reshape block_array into 2D
            block_array = block_array.reshape(int(rows), columns // n_blocks)

            # Add block_array to block_dict
            block_dict[i] = block_array

        # Join the blocks together
        print_array = np.concatenate(list(block_dict.values()), axis=1)
    else:
        print_array = fill_array(library_df, print_array, catalytic_binning)

    # Add blank rows to print array
    if skip_rows:
        # Add a blank row at every other row starting at row 1
        print_array = np.insert(print_array, np.arange(1, print_array.shape[0], 1), '', axis=0)
        print_array = np.insert(print_array, print_array.shape[0], '', axis=0)  # add final blank row

    return print_array