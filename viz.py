"""
Array Print Visualization Functions

Plotting and visualization utilities for array printing analysis.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
from adjustText import adjust_text
import textalloc as ta


def plot_mutant_frequency(print_array, working_dir, save_plot=True):
    """
    Plot frequency of each mutant in the print array.
    
    Args:
        print_array: Array containing mutant positions
        working_dir: Directory to save the plot
        save_plot: Whether to save the plot to disk
    
    Returns:
        fig, ax: Figure and axis objects
    """
    # get frequency of each mutant in print
    unique, counts = np.unique(print_array, return_counts=True)
    print_counts = dict(zip(unique, counts))

    # Remove blanks from print_counts
    print_counts.pop('', None)

    # rank mutants by frequency
    print_counts = dict(sorted(print_counts.items(), key=lambda item: item[1], reverse=True))

    # create text list from print_counts
    text_list = []
    for k, v in print_counts.items():
        text_list.append(f'{k}: {v}')

    # plot
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.scatter(print_counts.values(), print_counts.keys(), color='black')

    # add text
    x = list(print_counts.values())
    y = range(len(print_counts))
    ta.allocate(ax, x, y,
                text_list,
                x_scatter=x, y_scatter=y,
                textsize=10,
                avoid_label_lines_overlap=True
                )

    plt.xticks(rotation=90)
    # no x tick labels
    plt.gca().xaxis.set_ticklabels([])

    plt.title('Print Mutant Frequency')
    plt.xlabel('Mutant')
    plt.ylabel('Count')

    # create legend to show if any sequences are missing
    missing = []
    for k, v in print_counts.items():
        if v == 0:
            missing.append(k)

    if len(missing) > 0:
        handles = []
        labels = []
        for i in range(len(missing)):
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label=missing[i]))
            labels.append(missing[i])
    else:
        handles = []
        labels = []
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='None'))

        plt.legend(handles=handles, title='Missing Mutants', loc='upper right', bbox_to_anchor=(1.3, 1))

    # set facecolor to white
    plt.gcf().patch.set_facecolor('white')

    # save to metrics directory
    if save_plot:
        metrics_dir = os.path.join(working_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        plt.savefig(os.path.join(metrics_dir, 'print_mutant_frequency.png'), dpi=300, bbox_inches='tight')

    return fig, ax


def plot_mutant_position(print_array, selected_mutant, working_dir, save_plot=True):
    """
    Visualize the position of a single mutant in the array.
    
    Args:
        print_array: Array containing mutant positions
        selected_mutant: The mutant to highlight
        working_dir: Directory to save the plot
        save_plot: Whether to save the plot to disk
    
    Returns:
        fig, ax: Figure and axis objects
    """
    # get the indices of selected_mutant in the print array
    indices = np.where(print_array == selected_mutant)

    # generate imshow
    fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.imshow(print_array == selected_mutant, cmap='Greys')
    ax.set_title(f'{selected_mutant} in Print')

    # set facecolor to white
    fig.patch.set_facecolor('white')

    # save to metrics directory
    if save_plot:
        metrics_dir = os.path.join(working_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        plt.savefig(os.path.join(metrics_dir, f'{selected_mutant}_position_in_print.png'), dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_plate_layouts(num_plates=3):
    """
    Visualize 96-well and 384-well plate layouts.
    
    Args:
        num_plates: Number of 96-well plates to show
    
    Returns:
        figs: List of figure objects [fig_96, fig_384]
    """
    # 96-well plate visualization
    fig_96, axs = plt.subplots(1, num_plates, figsize=(15, 5))
    
    # plot matrix of 96-well plates one through three
    colors = matplotlib.colors.to_rgba_array(sns.color_palette('Blues', num_plates+1))
    colors = np.insert(colors, 0, [1, 1, 1, 1], axis=0)  # add white color to beginning of array
    values = np.arange(0, num_plates+1, 1)
    cmap = plt.cm.colors.ListedColormap(colors)

    for i in range(num_plates):
        M = np.ones((8, 12)) * i + 1
        ax = axs[i]
        xticks = np.arange(0, 12, 1)
        yticks = np.arange(0, 8, 1)
        cax = ax.matshow(M, vmin=0, vmax=num_plates, cmap=cmap)
        ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        xlabels = list(np.arange(1, 13, 1))
        ylabels = list('ABCDEFGH')
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
        ax.grid(True, which='minor', color='black', linestyle='-', linewidth=1)
        ax.set_title('96-well Plate ' + str(i+1), loc='left')

    # add colorbar to matrix, shrink to fit matrix, only show min to max values
    cbar = fig_96.colorbar(cax, ax=axs[:], shrink=0.5, aspect=5)
    cbar.ax.set_title('Plate Number', rotation=0, loc='center', pad=10)

    # 384-well plate visualization
    fig_384, ax = plt.subplots(1, 1, figsize=(7, 7))
    M = np.ones((16, 24)) * 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            # if in first iterant, set to 1
            if i % 2 != 1 and j % 2 != 1:
                M[i, j] = 1
            # if in second iterant, set to 2
            if i % 2 == 0 and j % 2 != 0:
                M[i, j] = 2
            # if in third iterant, set to 3
            if i % 2 != 0 and j % 2 == 0:
                M[i, j] = 3
            # if in fourth iterant, set to 4
            if i % 2 == 1 and j % 2 == 1:
                M[i, j] = 4

    ax.matshow(M, cmap=cmap, vmin=0, vmax=num_plates+1)
    ax.set_title('384-well Plate', loc='left')
    xlabels = list(np.arange(1, 25, 1))
    ylabels = list('ABCDEFGHIJKLMNOP')
    ax.set_xticks(np.arange(0, 24, 1))
    ax.set_yticks(np.arange(0, 16, 1))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    return [fig_96, fig_384]


def plot_array_heatmap(print_array, figsize=(10, 6)):
    """
    Plot a heatmap visualization of the print array.
    
    Args:
        print_array: Array containing mutant positions
        figsize: Figure size tuple
    
    Returns:
        fig, ax: Figure and axis objects
    """
    # convert print array to binary matrix
    print_df = pd.DataFrame(print_array)
    print_df_binary = print_df.applymap(lambda x: 1 if x != '' else 0)

    # plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(print_df_binary, cmap='binary', cbar=False, ax=ax, fmt='s')
    
    return fig, ax


def modify_array_column(print_array, column, depth):
    """
    Modify a specific column in the print array by adding empty spaces.
    
    Args:
        print_array: Array to modify
        column: Column index to modify
        depth: Number of rows to make empty from the top
    
    Returns:
        Modified array
    """
    print_array_modified = print_array.copy()
    column_w_empties = print_array_modified[:, column]
    column_w_empties[0:depth] = ''
    print_array_modified[:, column] = column_w_empties
    
    return print_array_modified