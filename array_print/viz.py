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
import string
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.embed import components
from bokeh.resources import CDN


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
    print_df_binary = print_df.map(lambda x: 1 if x != '' else 0)

    # plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(print_df_binary, cmap='binary', cbar=False, ax=ax, fmt='s')
    
    return fig, ax


def plot_replicate_histogram(print_array, working_dir, save_plot=True):
    """
    Plot a histogram of replicate counts per variant.

    Args:
        print_array: Array containing variant positions
        working_dir: Directory to save the plot
        save_plot: Whether to save the plot to disk

    Returns:
        fig, ax: Figure and axis objects
    """
    flat = print_array.flatten()
    variants = flat[flat != '']
    unique, counts = np.unique(variants, return_counts=True)

    # Histogram: x = replicate count value, y = number of variants with that count
    count_values, variant_counts = np.unique(counts, return_counts=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(count_values, variant_counts, color='steelblue', edgecolor='white', width=0.8)

    ax.set_xlabel('Replicates per Variant')
    ax.set_ylabel('Number of Variants')
    ax.set_title('Replicate Count Distribution')
    ax.grid(False)

    min_count = counts.min()
    max_count = counts.max()

    ax.axvline(min_count, color='firebrick', linestyle='--', linewidth=1.2)
    ax.text(min_count, ax.get_ylim()[1] * 0.95, f'min={min_count}',
            color='firebrick', ha='center', va='top', fontsize=9)

    ax.axvline(max_count, color='darkgreen', linestyle='--', linewidth=1.2)
    ax.text(max_count, ax.get_ylim()[1] * 0.95, f'max={max_count}',
            color='darkgreen', ha='center', va='top', fontsize=9)

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    if save_plot:
        metric_dir = os.path.join(working_dir, 'metric')
        os.makedirs(metric_dir, exist_ok=True)
        fig.savefig(os.path.join(metric_dir, 'replicate_histogram.png'), dpi=300, bbox_inches='tight')

    return fig, ax


def generate_dashboard(print_array, metrics, project_name, export_dir):
    """
    Generate a self-contained HTML dashboard with a variant table and heatmap.

    Args:
        print_array: Array containing variant names (post name-mapping)
        metrics: Dict of print metrics from get_print_metrics()
        project_name: Name for the dashboard header
        export_dir: Directory to save the HTML file
    """
    TAB10 = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

    # ── Replicate counts ──────────────────────────────────────────────────────
    flat = print_array.flatten()
    variants = flat[flat != '']
    unique_variants_1d, counts = np.unique(variants, return_counts=True)
    min_count = int(counts.min())
    max_count = int(counts.max())
    mean_count = float(np.mean(counts))

    # ── Color map (shared between table and heatmap) ──────────────────────────
    unique_variants = [v for v in np.unique(print_array) if v != '']
    variant_color = {v: TAB10[i % len(TAB10)] for i, v in enumerate(unique_variants)}

    # ── Left panel: always a scrollable HTML table ────────────────────────────
    # Sort ascending by replicate count so underrepresented variants surface first
    order = np.argsort(counts)
    table_rows = ''.join(
        '<tr>'
        f'<td style="padding:4px 10px">'
        f'<span style="display:inline-block;width:11px;height:11px;border-radius:2px;'
        f'background:{variant_color[v]};vertical-align:middle;margin-right:6px"></span>'
        f'{v}</td>'
        f'<td style="padding:4px 10px;text-align:right">{counts[i]}</td>'
        '</tr>'
        for i, v in ((int(i), unique_variants_1d[i]) for i in order)
    )
    th = ('padding:5px 10px;text-align:left;position:sticky;top:0;'
          'background:#f5f5f5;border-bottom:2px solid #ddd')
    left_panel_html = (
        f'<p style="font-weight:600;margin:0 0 8px">Variant Replicate Counts</p>'
        f'<p style="font-size:0.85em;color:#666;margin:0 0 10px">'
        f'min&nbsp;{min_count} &nbsp;·&nbsp; max&nbsp;{max_count}'
        f' &nbsp;·&nbsp; mean&nbsp;{mean_count:.1f}</p>'
        f'<div style="max-height:480px;overflow-y:auto">'
        f'<table style="border-collapse:collapse;width:100%;font-size:0.88em">'
        f'<thead><tr>'
        f'<th style="{th}">Variant</th>'
        f'<th style="{th};text-align:right">Replicates</th>'
        f'</tr></thead>'
        f'<tbody>{table_rows}</tbody>'
        f'</table></div>'
    )

    # ── Heatmap: Bokeh rect glyph with explicit per-cell colors ──────────────
    n_rows, n_cols = print_array.shape

    # Row labels: A-Z for ≤26 rows, numeric strings beyond
    if n_rows <= 26:
        row_labels = list(string.ascii_uppercase[:n_rows])
    else:
        row_labels = [str(i + 1) for i in range(n_rows)]

    row_list, col_list, color_list, name_list = [], [], [], []
    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            v = print_array[r_idx, c_idx]
            row_list.append(row_labels[r_idx])
            col_list.append(c_idx + 1)
            color_list.append(variant_color.get(v, '#ebebeb'))
            name_list.append(v if v else '(blank)')

    src_hm = ColumnDataSource(dict(
        row=row_list, col=col_list, color=color_list, name=name_list,
    ))

    # sizing_mode='scale_width' preserves the aspect ratio as panel width changes
    # → cells stay square on resize
    CELL_PX = 14
    hm_fig = figure(
        x_range=(0.5, n_cols + 0.5),
        y_range=row_labels[::-1],   # row 1 / 'A' at the top
        width=n_cols * CELL_PX,
        height=n_rows * CELL_PX,
        sizing_mode='scale_width',
        tools='hover,pan,wheel_zoom,reset',
        tooltips=[
            ('Variant', '@name'),
            ('Row',     '@row'),
            ('Column',  '@col'),
        ],
        title='Print Array Layout',
    )
    hm_fig.rect(
        'col', 'row', width=0.9, height=0.9, source=src_hm,
        fill_color='color', line_color='#cccccc', line_width=0.5,
    )
    hm_fig.xaxis.ticker = list(range(1, n_cols + 1))
    hm_fig.xaxis.axis_label = 'Column'
    hm_fig.yaxis.axis_label = 'Row'
    hm_fig.grid.grid_line_color = None
    hm_fig.toolbar.logo = None
    hm_fig.background_fill_color = None
    hm_fig.border_fill_color = None

    # ── Embed ──────────────────────────────────────────────────────────────────
    script, heatmap_div = components(hm_fig)

    variant_count = metrics.get('total_variants', len(unique_variants))
    filled = metrics.get('filled_positions', int((print_array != '').sum()))

    # Detect skipped (all-blank) rows inserted by the skip_rows option
    rows_all_blank = np.all(print_array == '', axis=1)
    n_skipped = int(rows_all_blank.sum())
    n_active = n_rows - n_skipped
    active_positions = n_active * n_cols
    effective_fill = filled / active_positions if active_positions > 0 else 0
    fill_str = f'{effective_fill:.1%} of active rows'

    if n_skipped > 0:
        skip_note = (
            f'&nbsp;|&nbsp; Rows: <strong>{n_active} active</strong>, '
            f'{n_skipped} skipped '
            f'<span style="color:#888;font-size:0.9em">'
            f'({n_skipped}/{n_rows} rows reserved for device spacing)</span>'
        )
    else:
        skip_note = ''

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{project_name} Dashboard</title>
  {CDN.render_css()}
  {CDN.render_js()}
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #555; margin-bottom: 20px; font-size: 0.95em; }}
    .panels {{ display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }}
    .panel {{ background: white; border-radius: 8px; padding: 16px;
              box-shadow: 0 1px 4px rgba(0,0,0,.1); flex: 1 1 360px; min-width: 0; }}
    tbody tr:nth-child(even) {{ background: #f9f9f9; }}
  </style>
</head>
<body>
  <h1>{project_name}</h1>
  <div class="meta">
    Variants: <strong>{variant_count}</strong> &nbsp;|&nbsp;
    Fill rate: <strong>{fill_str}</strong>{skip_note}
  </div>
  <div class="panels">
    <div class="panel">{left_panel_html}</div>
    <div class="panel">{heatmap_div}</div>
  </div>
  {script}
</body>
</html>"""

    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, f'{project_name}_dashboard.html')
    with open(out_path, 'w') as f:
        f.write(html)


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