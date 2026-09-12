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
import json
import string
from datetime import date
from html import escape as html_escape


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


# ── Dashboard palette ─────────────────────────────────────────────────────────
# The array map is drawn as a dark field with luminous spots, echoing the way a
# spotted microarray reads coming off a scanner. Everything around it is quiet
# paper so the map is the only thing competing for attention.
_SURFACE = '#ffffff'      # the paper the array is drawn on
_SKIPPED = '#f1f4f3'      # rows the device reserves for spacing
_EMPTY_WELL = '#e0e7e5'   # addressable position left unfilled

# Sequential, one hue, light to dark, used when no category column is supplied.
_COUNT_RAMP = ['#9ED3CC', '#7BC3BB', '#59B0A8', '#3E9891', '#2C7C77', '#1D605E', '#124746']

# One hue family per category; variants inside a category step from the light end
# to the dark end of their family. The order is set so that the families a run is
# most likely to use -- the first few -- are the ones furthest apart under
# colour-vision deficiency; validated with the palette checker, where the worst
# adjacent pair sits in the floor band and is relieved by the labelled legend.
_CATEGORY_FAMILIES = [
    ('#1F5FA8', '#9ECFFF'),   # blue
    ('#B35F0A', '#FFC27A'),   # orange
    ('#A81E74', '#FBA0DC'),   # magenta
    ('#1B7D38', '#95EB93'),   # green
    ('#6A2FC9', '#CBB0FF'),   # purple
    ('#8A7A0E', '#EFDF7A'),   # gold
    ('#0A7E74', '#6FE8D0'),   # teal
    ('#B32236', '#FFA8AE'),   # red
]

# Grid units per cell in the map's viewBox. The SVG scales to whatever box CSS
# gives it, so this only sets the internal proportions.
_U = 10
_GAP = 0.9


def _hex_to_rgb(value):
    return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))


def _blend(dark, light, t):
    """Linearly interpolate between two hex colors. t=0 → dark, t=1 → light."""
    r0, g0, b0 = _hex_to_rgb(dark)
    r1, g1, b1 = _hex_to_rgb(light)
    return '#%02x%02x%02x' % (
        round(r0 + (r1 - r0) * t),
        round(g0 + (g1 - g0) * t),
        round(b0 + (b1 - b0) * t),
    )


# Hover is the map's one interaction: the spots of the variant under the cursor
# hold their colour while everything else drops back, so a variant's replicates
# can be read as a pattern across the device. One mousemove listener on the SVG
# and one path rewrite per variant, rather than events and class flips on every
# cell.
_MAP_JS = """
(function () {
  var svg = document.getElementById('chip');
  if (!svg) return;
  var hl = document.getElementById('chip-hl');
  var out = document.getElementById('chip-readout');
  var C = __C__, R = __R__, U = __U__, S = __S__;
  var VW = __VW__, VH = __VH__, PL = __PL__, PT = __PT__;
  var GRID = __GRID__, V = __V__, ROWS = __ROWS__;


  var TRACE = [];
  for (var i = 0; i < GRID.length; i++) {
    var g = GRID[i];
    if (g >= 0) (TRACE[g] = TRACE[g] || []).push(i);
  }

  var cur = -2;

  function clear() {
    cur = -1;
    svg.classList.remove('tracing');
    hl.setAttribute('d', '');
    if (out) out.classList.remove('on');
  }

  function move(e) {
    var b = svg.getBoundingClientRect();
    if (!b.width || !b.height) return;
    var c = Math.floor(((e.clientX - b.left) / b.width * VW - PL) / U);
    var r = Math.floor(((e.clientY - b.top) / b.height * VH - PT) / U);
    var vi = (c < 0 || c >= C || r < 0 || r >= R) ? -1 : GRID[r * C + c];

    if (vi !== cur) {
      cur = vi;
      if (vi < 0) {
        clear();
      } else {
        var pts = TRACE[vi], d = '', j, p, x, y;
        for (j = 0; j < pts.length; j++) {
          p = pts[j];
          x = (p % C) * U + (U - S) / 2;
          y = Math.floor(p / C) * U + (U - S) / 2;
          d += 'M' + x + ' ' + y + 'h' + S + 'v' + S + 'h-' + S + 'z';
        }
        hl.setAttribute('d', d);
        hl.setAttribute('fill', V[vi].c);
        svg.classList.add('tracing');
        if (out) {
          out.innerHTML =
            '<span class="ro-sw" style="background:' + V[vi].c + '"></span>' +
            '<strong>' + V[vi].n + '</strong>' +
            (V[vi].g ? '<em>' + V[vi].g + '</em>' : '') +
            '<span class="ro-n">' + pts.length + ' replicates</span>' +
            '<span class="ro-at"></span>';
          out.classList.add('on');
        }
      }
    }

    if (vi >= 0 && out) {
      out.querySelector('.ro-at').textContent =
        'row ' + ROWS[r] + ', column ' + (c + 1);
    }
  }

  svg.addEventListener('mousemove', move);
  svg.addEventListener('mouseleave', clear);

  var mask = document.getElementById('chip-mask');
  var W = C * U, H = R * U;
  svg.querySelectorAll('g.blk').forEach(function (lab) {
    lab.addEventListener('mouseenter', function () {
      var x0 = +lab.getAttribute('data-c0') * U,
          x1 = +lab.getAttribute('data-c1') * U, d = '';
      if (x0 > 0) d += 'M0 0h' + x0 + 'v' + H + 'h-' + x0 + 'z';
      if (x1 < W) d += 'M' + x1 + ' 0h' + (W - x1) + 'v' + H + 'h-' + (W - x1) + 'z';
      mask.setAttribute('d', d);
      svg.classList.add('masked');
    });
    lab.addEventListener('mouseleave', function () {
      svg.classList.remove('masked');
    });
  });
})();
"""


_DASHBOARD_CSS = """
*, *::before, *::after { box-sizing: border-box; }

:root {
  --ground: #eef1f0;
  --surface: #ffffff;
  --ink: #0e2b2b;
  --ink-muted: #4a6360;
  --rule: #d0d9d6;
  --flag: #a81e3f;
  --grid: #eef1ef;
  --avail: 520px;
  --font-ui: Helvetica, 'Helvetica Neue', Arial, 'Liberation Sans', sans-serif;
}

html { -webkit-text-size-adjust: 100%; }

body {
  margin: 0;
  padding: 22px clamp(16px, 3vw, 40px) 24px;
  background: var(--ground);
  color: var(--ink);
  font-family: var(--font-ui);
  font-size: 16px;
  line-height: 1.5;
  font-feature-settings: 'tnum' 1;
}

.sheet { max-width: 1180px; margin: 0 auto; }

/* ── Header reads as a specimen label: name, then the run specification ────── */
h1 {
  margin: 0;
  font-family: var(--font-ui);
  font-size: clamp(1.25rem, 1rem + 0.8vw, 1.55rem);
  font-weight: 700;
  letter-spacing: -0.018em;
  line-height: 1.15;
}

.byline {
  margin: 3px 0 0;
  font-family: var(--font-ui);
  font-size: 0.75rem;
  color: var(--ink-muted);
}

.figures {
  display: flex;
  flex-wrap: wrap;
  gap: 7px 17px;
  margin: 11px 0 0;
  padding: 0;
  list-style: none;
}

.figures li { padding-left: 17px; border-left: 1px solid var(--rule); }

.figures li:first-child { padding-left: 0; border-left: 0; }

.fig-v {
  display: block;
  font-size: 0.875rem;
  font-weight: 700;
  line-height: 1.2;
  letter-spacing: -0.01em;
  font-variant-numeric: tabular-nums;
}

.fig-l {
  display: block;
  margin-top: 1px;
  font-size: 0.6875rem;
  line-height: 1.25;
  color: var(--ink-muted);
}



.layout {
  display: grid;
  grid-template-columns: minmax(260px, 520px) minmax(250px, 1fr);
  gap: clamp(16px, 3vw, 44px);
  align-items: start;
  margin-top: 18px;
}

/* ── The array map: the one dark object on the page ───────────────────────── */
.map {
  max-height: var(--avail);
  display: flex;
  align-items: stretch;
  justify-content: start;
  min-width: 0;
  gap: 12px;
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 3px;
  padding: 10px 12px;
}

/* Label tracks are CSS grids over the same row and column counts as the SVG, so
   they stay registered to the cells at any size while the text itself keeps a
   fixed pixel size instead of shrinking with the map. */
.chip {
  min-width: 0;
  --plot-h: max(220px, calc(min(var(--avail), 900px) - 56px));
  --plot-w: calc(var(--plot-h) * var(--chip-ratio));
}

.plot { position: relative; width: var(--plot-w); max-width: 100%; }

#chip { display: block; width: 100%; height: auto; }

#chip .cells { transition: opacity 170ms ease-out; }

#chip.tracing .cells { opacity: 0.18; }

#chip-hl, #chip-mask { pointer-events: none; }

#chip-mask { opacity: 0; transition: opacity 170ms ease-out; }

#chip.masked #chip-mask { opacity: 0.78; }

.tick { stroke: var(--ink); stroke-width: 1; }

.axlab {
  fill: var(--ink-muted);
  font-size: var(--axfs, 12px);
  font-family: var(--font-ui);
}

.blk-hit { fill: transparent; }

.blk-lab {
  fill: var(--ink);
  font-size: var(--blkfs, 13px);
  font-weight: 500;
  font-family: var(--font-ui);
}

.blk-rule { stroke: var(--ink); stroke-width: 1.4; }

g.blk { cursor: default; }

g.blk:hover .blk-lab { fill: var(--flag); }

g.blk:hover .blk-rule { stroke: var(--flag); }

.chart-tip {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 2;
  pointer-events: none;
  max-width: 240px;
  padding: 7px 9px;
  border: 1px solid var(--rule);
  border-radius: 3px;
  background: var(--surface);
  color: var(--ink);
  font-size: 0.75rem;
  line-height: 1.45;
  box-shadow: 0 3px 12px rgba(14, 43, 43, 0.14);
}

.chart-tip b { display: block; font-size: 0.8125rem; font-weight: 600; }

.chart-tip em { display: block; color: var(--ink-muted); font-style: normal; }

/* The array's readout sits under the legend rather than following the cursor,
   so it can never cover the spots being inspected. Its height is fixed and every
   line is clipped rather than wrapped, so moving between squares never resizes
   it. */
.readout {
  height: 54px;
  margin-top: 8px;
  font-size: 0.75rem;
  line-height: 1.4;
  color: var(--ink-muted);
  opacity: 0;
  transition: opacity 130ms ease-out;
}

.readout.on { opacity: 1; }

.readout > * {
  display: block;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.readout strong {
  font-size: 0.875rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--ink);
}

.readout em { font-style: normal; }

.readout .ro-n { color: var(--ink); font-variant-numeric: tabular-nums; }

.readout .ro-at { font-variant-numeric: tabular-nums; }

.readout .ro-sw {
  display: block;
  float: right;
  width: 13px;
  height: 13px;
  margin-left: 6px;
  border: 1px solid rgba(14, 43, 43, 0.55);
}

.map-key {
  flex: none;
  align-self: start;
  width: 190px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 11px;
  border: 1px solid var(--rule);
  border-radius: 3px;
  font-size: 0.8125rem;
  line-height: 1.4;
  color: var(--ink-muted);
}

.key { display: flex; align-items: center; gap: 9px; }

.key.start-singles { margin-top: 7px; }

.key-steps { display: flex; gap: 2px; }

.swatch, .key-ramp {
  height: 13px;
  flex: none;
  border: 1px solid rgba(14, 43, 43, 0.55);
}

.swatch { width: 13px; margin-right: 17px; }

.map-key .key > .swatch { margin-right: 17px; }

.key-ramp { width: 30px; }

.key-name { color: var(--ink); }

.key-label {
  font-variant-numeric: tabular-nums;
  color: var(--ink);
}

/* ── Replicate panel ──────────────────────────────────────────────────────── */
.counts {
  position: relative;
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  max-height: var(--avail);
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 3px;
  padding: 16px clamp(14px, 1.6vw, 22px) 12px;
}

h2 {
  margin: 0;
  font-family: var(--font-ui);
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: -0.012em;
}

.figures li.flag { border-left-color: var(--flag); }

.figures li.flag .fig-v, .figures li.flag .fig-l { color: var(--flag); }

.chart {
  display: block;
  flex: 0 1 auto;
  width: 100%;
  height: auto;
  min-height: 0;
  margin: 10px 0 0;
  overflow: visible;
}

.chart text { font-family: var(--font-ui); }

.bar { fill: var(--ink); transition: fill-opacity 140ms ease-out; }

.bar.flag { fill: var(--flag); }

.bar:hover { fill-opacity: 0.78; }

.grid { stroke: var(--grid); stroke-width: 1; }

.ax { fill: var(--ink-muted); font-size: 10.5px; font-variant-numeric: tabular-nums; }

.axtitle { fill: var(--ink-muted); font-size: 10px; }

.callout { fill: var(--flag); font-size: 11px; font-weight: 700; }

.lead { stroke: var(--flag); stroke-width: 1; stroke-dasharray: 2 3; }

.lead.plain { stroke: var(--ink-muted); }

.callout.plain { fill: var(--ink-muted); font-weight: 600; }

.rowlab { fill: var(--ink); font-size: 11px; }

.span { stroke: #ccd6d3; stroke-width: 2; stroke-linecap: round; }

/* Round caps so a degenerate middle half -- every variant on the same count --
   still reads as a mark rather than vanishing. */
.iqr { stroke: #9fb0ad; stroke-width: 6; stroke-linecap: round; }

.mid { fill: var(--ink); stroke: #fff; stroke-width: 1.2; }

.rowval {
  fill: var(--ink-muted);
  font-size: 10.5px;
  font-variant-numeric: tabular-nums;
  paint-order: stroke;
  stroke: var(--surface);
  stroke-width: 3px;
  stroke-linejoin: round;
}

.rowval.flag { fill: var(--flag); font-weight: 600; }

.hit { fill: transparent; }

g[data-tip]:hover .iqr { stroke: var(--ink-muted); }

g[data-tip]:hover .rowlab { font-weight: 600; }

h3 {
  margin: 14px 0 0;
  font-family: var(--font-ui);
  font-size: 0.875rem;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.empty { margin: 0; padding: 40px 0; color: var(--ink-muted); max-width: 52ch; }

@media (max-width: 1200px) {
  .map { flex-direction: column; }
  .map-key {
    width: auto;
    align-self: stretch;
    flex-direction: row;
    flex-wrap: wrap;
    align-items: flex-start;
    gap: 7px 15px;
    margin-top: 10px;
  }
  .readout { flex: 1 1 190px; height: 52px; margin-top: 0; }
  .key.start-singles { margin-top: 0; }
}

@media (max-width: 820px) {
  .layout { grid-template-columns: minmax(0, 1fr); }
  .counts, .map { max-height: none; }
  .map { flex-direction: column; gap: 0; }
  .map-key {
    width: auto;
    align-self: stretch;
    flex-direction: row;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 16px;
    margin-top: 11px;
  }
  .counts { max-width: 100%; }
}

@media (prefers-reduced-motion: reduce) {
  #chip .cells, .bar, #chip-mask, .readout { transition: none; }
}
"""


# ── Charts ────────────────────────────────────────────────────────────────────
# Both charts are inline SVG on the paper panel and share one horizontal scale,
# so a group's range in the lower chart lines up with the distribution above it.
_CHART_W = 520
_CHART_L = 34      # left gutter, just the y labels
_CHART_R = 14
_HIST_H = 84       # plotting height of the distribution
_HIST_TOP = 30     # headroom for the callout naming the fewest-replicated variant
_ROW_H = 30        # one group per row in the range chart


def _nice_axis(value):
    """Pick an axis maximum and tick step that land on whole numbers."""
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000):
        if value <= step * 5:
            top = int(np.ceil(value / step) * step)
            return max(top, step), step
    return int(value), int(value)


def _replicate_histogram(counts, min_count, max_count, low_names, flag_low):
    """Distribution of replicate counts, with the fewest-replicated variant named."""
    span = max_count + 1
    if span <= 48:
        edges = list(range(0, max_count + 2))
        labels = [str(c) for c in range(0, max_count + 1)]
    else:
        n_bins = 24
        width = span / n_bins
        edges = [i * width for i in range(n_bins + 1)]
        labels = [f'{edges[i]:.0f}' for i in range(n_bins)]

    step = edges[1] - edges[0]
    freq = [0] * (len(edges) - 1)
    for c in counts:
        freq[min(int((c - edges[0]) / step), len(freq) - 1)] += 1
    low_bin = min(int((min_count - edges[0]) / step), len(freq) - 1)

    plot_w = _CHART_W - _CHART_L - _CHART_R
    top, tick = _nice_axis(max(freq))
    base = _HIST_TOP + _HIST_H
    slot = plot_w / len(freq)
    # 2px surface gap between adjacent bars, capped so a two-bin chart does not
    # turn into a pair of slabs.
    bar_w = min(max(2.0, slot - 2), 46)

    def y_of(v):
        return base - (v / top) * _HIST_H

    grid, yl = [], []
    for v in range(0, top + 1, tick):
        y = y_of(v)
        grid.append(f'<line class="grid" x1="{_CHART_L}" y1="{y:.1f}" '
                    f'x2="{_CHART_L + plot_w}" y2="{y:.1f}"/>')
        yl.append(f'<text class="ax" x="{_CHART_L - 8}" y="{y + 4:.1f}" '
                  f'text-anchor="end">{v}</text>')

    bars, xl = [], []
    label_every = max(1, len(freq) // 12)
    for i, n in enumerate(freq):
        x = _CHART_L + i * slot + (slot - bar_w) / 2
        flagged = flag_low and i == low_bin
        if n:
            h = max(base - y_of(n), 2.4)
            r = min(3, bar_w / 2, h)
            tip = (f'<b>{n}</b> variant{"s" if n != 1 else ""} at '
                   f'{labels[i]} replicates')
            if flagged and low_names:
                shown = ', '.join(low_names[:6])
                if len(low_names) > 6:
                    shown += f' +{len(low_names) - 6} more'
                tip += '<em>' + html_escape(shown) + '</em>'
            bars.append(
                f'<path class="bar{" flag" if flagged else ""}" data-tip="{tip}" '
                f'd="M{x:.1f} {base:.1f}V{base - h + r:.1f}'
                f'a{r:.1f} {r:.1f} 0 0 1 {r:.1f} -{r:.1f}'
                f'h{bar_w - 2 * r:.1f}'
                f'a{r:.1f} {r:.1f} 0 0 1 {r:.1f} {r:.1f}'
                f'V{base:.1f}z"/>'
            )
        # No forced final tick: its distance from the previous one is always
        # less than the step, so it can only ever collide. The header strip
        # carries the range instead.
        if i % label_every == 0:
            xl.append(f'<text class="ax" x="{_CHART_L + i * slot + slot / 2:.1f}" '
                      f'y="{base + 16:.0f}" text-anchor="middle">{labels[i]}</text>')

    # The point of the chart is which variant is short, so it gets named on the mark.
    callout = ''
    if low_names and freq[low_bin]:
        cx = _CHART_L + low_bin * slot + slot / 2
        if len(low_names) <= 3:
            named = ', '.join(html_escape(n) for n in low_names)
            text = f'fewest: {named} at {min_count}'
        else:
            text = f'{len(low_names):,} variants at {min_count}, the fewest'
        flip = cx + len(text) * 5.8 > _CHART_L + plot_w
        anchor = ' text-anchor="end"' if flip else ''
        tx = cx - 6 if flip else cx + 6
        cls = 'callout' if flag_low else 'callout plain'
        callout = (
            f'<line class="lead{"" if flag_low else " plain"}" x1="{cx:.1f}" '
            f'y1="{min(y_of(freq[low_bin]), base - 2.4) - 5:.1f}" '
            f'x2="{cx:.1f}" y2="{_HIST_TOP - 16:.0f}"/>'
            f'<text class="{cls}" x="{tx:.1f}" y="{_HIST_TOP - 12:.0f}"{anchor}>{text}</text>'
        )

    def x_of(v):
        return _CHART_L + (v - edges[0]) / (edges[1] - edges[0]) * slot + slot / 2

    height = base + 28
    svg = (
        f'<svg class="chart" viewBox="0 0 {_CHART_W} {height}" '
        f'role="img" aria-label="Distribution of replicate counts across variants. '
        f'Replicates range from {min_count} to {max_count}.">'
        f'{"".join(grid)}{"".join(bars)}{"".join(yl)}{"".join(xl)}{callout}'
        f'<text class="axtitle" x="{_CHART_L + plot_w / 2:.0f}" y="{height - 3}" '
        f'text-anchor="middle">replicates per variant</text>'
        f'<text class="axtitle" x="1" y="13">variants</text>'
        f'</svg>'
    )
    return svg, x_of


def _group_ranges(groups, min_count, heading, x_of, flag_low):
    """Replicate range per group, on the same scale as the histogram above it."""
    plot_w = _CHART_W - _CHART_L - _CHART_R
    rows = []
    for i, (name, vals) in enumerate(groups):
        y = 22 + i * _ROW_H
        g_lo, g_hi = min(vals), max(vals)
        mid = float(np.median(vals))
        q1, q3 = (float(np.percentile(vals, 25)), float(np.percentile(vals, 75)))
        flagged = g_lo == min_count and flag_low
        # Both labels ride with the range and are pulled back inside the plot
        # when they would overrun its right edge.
        # Name and range read as one label above the row, so they cannot drift
        # apart or land on the mark however the range happens to sit.
        label = str(g_lo) if g_lo == g_hi else f'{g_lo}\u2013{g_hi}'
        head_w = (len(name) + len(label) + 2) * 6.2
        name_x = min(x_of(g_lo) - 1, _CHART_W - _CHART_R - head_w)
        name_x = max(name_x, _CHART_L)
        tip = (f'<b>{html_escape(name)}</b><em>{len(vals)} variants, median '
               f'{mid:.0f}, middle half {q1:.0f}&#8211;{q3:.0f}, '
               f'range {g_lo}&#8211;{g_hi}</em>')
        rows.append(
            f'<g data-tip="{tip}">'
            f'<rect class="hit" x="0" y="{y - 20:.0f}" '
            f'width="{_CHART_W}" height="{_ROW_H}"/>'
            f'<text class="rowlab" x="{name_x:.1f}" y="{y - 8:.0f}">'
            f'{html_escape(name)}'
            f'<tspan class="rowval{" flag" if flagged else ""}" dx="7">{label}</tspan>'
            f'</text>'
            f'<line class="span" x1="{x_of(g_lo):.1f}" y1="{y:.0f}" '
            f'x2="{x_of(g_hi):.1f}" y2="{y:.0f}"/>'
            f'<line class="iqr" x1="{x_of(q1):.1f}" y1="{y:.0f}" '
            f'x2="{x_of(q3):.1f}" y2="{y:.0f}"/>'
            f'<circle class="mid" cx="{x_of(mid):.1f}" cy="{y:.0f}" r="3.2"/>'
            f'</g>'
        )

    height = 22 + len(groups) * _ROW_H - 8
    return (
        f'<svg class="chart" viewBox="0 0 {_CHART_W} {height}" role="img" '
        f'aria-label="Replicate range for each {heading}.">{"".join(rows)}</svg>'
    )


# Sizes the sheet to the window. The header's figure strip wraps at narrow
# widths, so no fixed offset holds -- the room below it is measured instead.
# The map is then given a height the column can actually accommodate: if the
# column is narrower than that height's width, the height comes back down, so a
# spot stays square whatever the window's aspect ratio.
_FIT_JS = """
(function () {
  var layout = document.querySelector('.layout');
  var chip = document.querySelector('.chip');
  var plot = document.querySelector('.plot');
  var map = document.querySelector('.map');
  var mkey = document.querySelector('.map-key');
  if (!layout) return;
  var RATIO = __RATIO__, VW = __FITVW__, root = document.documentElement;

  function fit() {
    var top = layout.getBoundingClientRect().top + (window.scrollY || 0);
    var room = Math.max(260, window.innerHeight - top - 28);
    root.style.setProperty('--avail', room + 'px');
    if (!chip || !plot) return;

    // When the legend drops below the map it takes height the array can no
    // longer use, so it is measured rather than assumed.
    var chrome = 56;
    if (map && mkey && getComputedStyle(map).flexDirection === 'column') {
      chrome += mkey.offsetHeight + 12;
    }
    var h = Math.max(200, Math.min(room - chrome, 900));
    chip.style.setProperty('--plot-h', h + 'px');
    var w = plot.getBoundingClientRect().width;
    if (w > 0 && w < h * RATIO - 0.5) {
      chip.style.setProperty('--plot-h', (w / RATIO).toFixed(2) + 'px');
      w = plot.getBoundingClientRect().width;
    }

    // Everything in the map scales with it, so the axis text is sized in grid
    // units that cancel that scale out. Without this a small window renders the
    // row and column labels at a few pixels.
    var svg = document.getElementById('chip');
    if (svg && w > 0) {
      var k = w / VW;
      var clamp = function (px) { return Math.min(Math.max(px / k, px), px * 1.6); };
      svg.style.setProperty('--axfs', clamp(11).toFixed(2) + 'px');
      svg.style.setProperty('--blkfs', clamp(12).toFixed(2) + 'px');
    }
  }

  fit();
  window.addEventListener('resize', fit);
  setTimeout(fit, 160);
})();
"""


# Reloads the page so a regenerated sheet appears without a manual refresh. It
# waits for the pointer to settle and for the tab to be visible, so it never
# interrupts someone reading the map.
_REFRESH_JS = """
(function () {
  var every = __SECS__;
  if (!every) return;
  var moved = 0;
  document.addEventListener('mousemove', function () { moved = Date.now(); });
  setInterval(function () {
    if (document.hidden) return;
    if (Date.now() - moved < 4000) return;
    location.reload();
  }, every * 1000);
})();
"""


# One tooltip shared by both charts; hover is delegated off the data-tip marks.
_CHART_JS = """
(function () {
  var tip = document.getElementById('chart-tip');
  if (!tip) return;
  var host = tip.parentNode;
  function place(e) {
    var b = host.getBoundingClientRect();
    var x = e.clientX - b.left + 13, y = e.clientY - b.top + 13;
    if (x + tip.offsetWidth > b.width) x = e.clientX - b.left - tip.offsetWidth - 13;
    if (y + tip.offsetHeight > b.height) y = e.clientY - b.top - tip.offsetHeight - 13;
    tip.style.transform = 'translate(' + Math.max(0, x) + 'px,' + Math.max(0, y) + 'px)';
  }
  host.addEventListener('mousemove', function (e) {
    var m = e.target.closest('[data-tip]');
    if (!m) { tip.hidden = true; return; }
    tip.innerHTML = m.getAttribute('data-tip');
    tip.hidden = false;
    place(e);
  });
  host.addEventListener('mouseleave', function () { tip.hidden = true; });
})();
"""


def generate_dashboard(print_array, metrics, project_name, export_dir,
                       categories=None, n_blocks=None, refresh=0):
    """
    Write a self-contained HTML sheet for checking a print before it runs.

    The page pairs a map of the physical array with the replicate count for every
    variant. The least-replicated variant is named on the distribution chart, and
    hovering a spot on the map holds that variant's positions at full colour while
    everything else drops back, which shows whether its replicates are spread
    across the device or clustered in one region.

    The map is inline SVG with no external dependencies, so the file opens
    without a network connection.

    Args:
        print_array: Array containing variant names (post name-mapping)
        metrics: Dict of print metrics from get_print_metrics()
        project_name: Name for the sheet header
        export_dir: Directory to save the HTML file
        categories: Optional dict mapping variant name → category label. When
            given, each category is drawn in its own hue family and variants
            step through the shades of their family. Without it, spots are
            colored by replicate count.
        n_blocks: Optional number of blocks the device is divided into. Blocks
            partition the array by column, and are labelled above the map, each
            under a rule spanning its own columns.
        refresh: Seconds between automatic page reloads, so a regenerated sheet
            appears without a manual refresh. 0 turns it off. Reloading waits for
            the pointer to settle and for the tab to be visible.

    Returns:
        Path to the written HTML file.
    """
    n_rows, n_cols = print_array.shape

    flat = print_array.flatten()
    present = flat[flat != '']
    unique_variants, counts = np.unique(present, return_counts=True)

    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, f'{project_name}_dashboard.html')

    if unique_variants.size == 0:
        with open(out_path, 'w') as f:
            f.write(_empty_dashboard(project_name, n_rows, n_cols))
        return out_path

    count_of = {str(v): int(c) for v, c in zip(unique_variants, counts)}
    min_count = int(counts.min())
    max_count = int(counts.max())
    mean_count = float(counts.mean())
    n_at_min = int((counts == min_count).sum())
    has_spread = (max_count - min_count) > 1
    # When most of the library sits on the floor -- which is what happens
    # whenever the library is large relative to the array -- that floor is the
    # design, not a defect, and colouring it as a fault buries the real signal.
    flag_low = has_spread and n_at_min <= max(1, round(0.25 * len(count_of)))

    # ── Assign a color to every variant ───────────────────────────────────────
    cat_of = {}
    if categories:
        for v in count_of:
            label = categories.get(v)
            cat_of[v] = str(label) if label not in (None, '') else 'Uncategorized'

    legend_blocks = []
    variant_color = {}

    if cat_of:
        # Hue families are assigned in fixed order and never cycled: past the
        # eighth category the tail folds into one neutral group rather than
        # reusing a hue that already means something else.
        by_size = sorted(
            set(cat_of.values()),
            key=lambda c: (-sum(1 for v in cat_of if cat_of[v] == c), c),
        )
        if len(by_size) > len(_CATEGORY_FAMILIES):
            keep = set(by_size[:len(_CATEGORY_FAMILIES)])
            cat_of = {v: (c if c in keep else 'Other') for v, c in cat_of.items()}

        # Categories holding one variant are listed together after the scans,
        # rather than interleaved alphabetically among them.
        size_of = {c: sum(1 for v in cat_of if cat_of[v] == c) for c in set(cat_of.values())}
        ordered_cats = sorted(
            size_of,
            key=lambda c: (c in ('Other', 'Uncategorized'), size_of[c] == 1, c),
        )
        first_single = next(
            (i for i, c in enumerate(ordered_cats) if size_of[c] == 1), None
        )
        for i, cat in enumerate(ordered_cats):
            dark, light = (
                ('#5A6B69', '#B9C4C2') if cat == 'Other'
                else _CATEGORY_FAMILIES[i % len(_CATEGORY_FAMILIES)]
            )
            members = sorted(v for v in count_of if cat_of[v] == cat)
            if len(members) == 1:
                stops = [0.3]
            else:
                stops = [0.62 * (j / (len(members) - 1)) for j in range(len(members))]
            shades = [_blend(dark, light, t) for t in stops]
            for v, shade in zip(members, shades):
                variant_color[v] = shade

            # The bar is built from the shades actually assigned, so what the
            # legend shows is exactly what the map draws -- not the family's
            # full range, only the part of it this category uses.
            if len(shades) == 1:
                mark = f'<span class="swatch" style="background:{shades[0]}"></span>'
            else:
                step = 100 / (len(shades) - 1)
                bar = 'linear-gradient(90deg,' + ','.join(
                    f'{c} {i * step:.2f}%' for i, c in enumerate(shades)
                ) + ')'
                mark = f'<span class="key-ramp" style="background:{bar}"></span>'
            gap = ' start-singles' if first_single is not None and i == first_single else ''
            legend_blocks.append(
                f'<span class="key{gap}">{mark}'
                f'<span class="key-name">{html_escape(cat)}</span>'
                '</span>'
            )
    else:
        distinct = sorted({int(c) for c in counts})
        if len(distinct) == 1:
            step_of = {distinct[0]: _COUNT_RAMP[3]}
        else:
            positions = np.linspace(0, len(_COUNT_RAMP) - 1, len(distinct)).round().astype(int)
            step_of = {c: _COUNT_RAMP[p] for c, p in zip(distinct, positions)}
        for v, c in count_of.items():
            variant_color[v] = step_of[c]
        shown = distinct if len(distinct) <= 7 else [distinct[0], distinct[len(distinct) // 2], distinct[-1]]
        steps = ''.join(
            f'<span class="swatch" style="background:{step_of[c]}" title="{c} replicates"></span>'
            for c in shown
        )
        legend_blocks.append(
            '<span class="key">'
            f'<span class="key-label">{min_count}</span>'
            f'<span class="key-steps">{steps}</span>'
            f'<span class="key-label">{max_count}</span>'
            '<span>replicates</span>'
            '</span>'
        )

    # ── Map geometry ──────────────────────────────────────────────────────────
    if n_rows <= 26:
        row_labels = list(string.ascii_uppercase[:n_rows])
    else:
        row_labels = [str(i + 1) for i in range(n_rows)]

    rows_all_blank = np.all(print_array == '', axis=1)
    side = _U - _GAP
    idx_of = {v: i for i, v in enumerate(sorted(count_of))}

    # One path per colour keeps the map to a few dozen nodes instead of one per
    # cell, which matters at a couple of thousand positions.
    runs = {}
    grid = []
    for r in range(n_rows):
        for c in range(n_cols):
            v = str(print_array[r, c])
            x = c * _U + _GAP / 2
            y = r * _U + _GAP / 2
            if v == '':
                color = _SKIPPED if rows_all_blank[r] else _EMPTY_WELL
                grid.append(-1)
            else:
                color = variant_color[v]
                grid.append(idx_of[v])
            runs.setdefault(color, []).append(
                f'M{x:g} {y:g}h{side:g}v{side:g}h-{side:g}z'
            )

    cells = ''.join(
        f'<path fill="{color}" d="{"".join(d)}"/>' for color, d in runs.items()
    )
    # ── One SVG for the whole map ─────────────────────────────────────────────
    # Cells, ticks, axis labels and block marks share a single uniformly scaled
    # coordinate system, so a tick sits on its cell by construction rather than
    # by two layout engines agreeing to the pixel. Uniform scaling also keeps the
    # cells square and the text undistorted at any size.
    blocked = bool(n_blocks) and n_blocks > 1 and n_cols % n_blocks == 0
    pad_l, pad_r, pad_b = 42.0, 6.0, 26.0
    pad_t = 30.0 if blocked else 6.0
    grid_w, grid_h = n_cols * _U, n_rows * _U
    view_w, view_h = pad_l + grid_w + pad_r, pad_t + grid_h + pad_b

    x_step = 1 if n_cols <= 12 else (5 if n_cols <= 60 else 10)
    y_step = 1 if n_rows <= 12 else (5 if n_rows <= 60 else 10)

    col_marks = ''.join(
        f'<line class="tick" x1="{pad_l + c * _U + _U / 2:g}" y1="{pad_t + grid_h + 1.5:g}" '
        f'x2="{pad_l + c * _U + _U / 2:g}" y2="{pad_t + grid_h + 6:g}"/>'
        f'<text class="axlab" x="{pad_l + c * _U + _U / 2:g}" '
        f'y="{pad_t + grid_h + 19:g}" text-anchor="middle">{c + 1}</text>'
        for c in range(0, n_cols, x_step)
    )
    row_marks = ''.join(
        f'<line class="tick" x1="{pad_l - 6:g}" y1="{pad_t + r * _U + _U / 2:g}" '
        f'x2="{pad_l - 1.5:g}" y2="{pad_t + r * _U + _U / 2:g}"/>'
        f'<text class="axlab" x="{pad_l - 9:g}" y="{pad_t + r * _U + _U / 2 + 4.2:g}" '
        f'text-anchor="end">{row_labels[r]}</text>'
        for r in range(0, n_rows, y_step)
    )

    block_marks = ''
    if blocked:
        span = n_cols // n_blocks
        parts = []
        for k in range(n_blocks):
            x0 = pad_l + k * span * _U + 1.2
            x1 = pad_l + (k + 1) * span * _U - 1.2
            parts.append(
                f'<g class="blk" data-c0="{k * span}" data-c1="{(k + 1) * span}">'
                f'<rect class="blk-hit" x="{x0 - 1.2:g}" y="0" '
                f'width="{(x1 - x0) + 2.4:g}" height="{pad_t:g}"/>'
                f'<text class="blk-lab" x="{(x0 + x1) / 2:g}" y="{pad_t - 9:g}" '
                f'text-anchor="middle">Block {k + 1}</text>'
                f'<line class="blk-rule" x1="{x0:g}" y1="{pad_t - 5:g}" '
                f'x2="{x1:g}" y2="{pad_t - 5:g}"/>'
                f'</g>'
            )
        block_marks = ''.join(parts)

    svg = (
        f'<svg id="chip" viewBox="0 0 {view_w:g} {view_h:g}" '
        f'role="img" aria-label="Layout of the {n_rows} by {n_cols} print array">'
        f'<g transform="translate({pad_l:g},{pad_t:g})">'
        f'<g class="cells">{cells}</g>'
        f'<path id="chip-hl" d=""/>'
        f'<path id="chip-mask" d="" fill="{_SURFACE}"/>'
        f'</g>'
        f'{row_marks}{col_marks}{block_marks}'
        f'</svg>'
    )

    # ── Hover data ────────────────────────────────────────────────────────────
    variants_js = [None] * len(idx_of)
    for v, i in idx_of.items():
        variants_js[i] = {
            'c': variant_color[v],
            'n': html_escape(v),
            'g': html_escape(cat_of[v]) if cat_of else '',
        }
    fit_js = (_FIT_JS.replace('__RATIO__', f'{view_w / view_h:.6f}')
              .replace('__FITVW__', f'{view_w:g}'))
    refresh_js = _REFRESH_JS.replace('__SECS__', str(int(refresh or 0)))
    map_js = (
        _MAP_JS.replace('__C__', str(n_cols))
        .replace('__R__', str(n_rows))
        .replace('__U__', str(_U))
        .replace('__S__', f'{side:g}')
        .replace('__GRID__', json.dumps(grid, separators=(',', ':')))
        .replace('__V__', json.dumps(variants_js, separators=(',', ':')))
        .replace('__ROWS__', json.dumps(row_labels, separators=(',', ':')))
        .replace('__VW__', f'{view_w:g}').replace('__VH__', f'{view_h:g}')
        .replace('__PL__', f'{pad_l:g}').replace('__PT__', f'{pad_t:g}')
    )

    # ── Copy ──────────────────────────────────────────────────────────────────
    filled = int(metrics.get('filled_positions', len(present)))
    n_skipped = int(rows_all_blank.sum())
    n_active = n_rows - n_skipped
    active_positions = n_active * n_cols
    fill_rate = filled / active_positions if active_positions else 0.0

    figures = [(f'{n_rows} &#215; {n_cols}', 'rows &#215; columns')]  # (value, label[, 'flag'])
    if filled == active_positions:
        figures.append((f'{filled:,}', 'positions, all filled'))
    else:
        figures.append((f'{fill_rate:.0%}',
                        f'filled, {filled:,} of {active_positions:,}'))
    figures.append((f'{len(count_of):,}', 'variants'))
    if blocked:
        figures.append((str(n_blocks), f'blocks of {n_cols // n_blocks} columns'))
    if n_skipped:
        figures.append((str(n_skipped), 'rows blank for spacing'))
    figures.append((
        f'{min_count}' if min_count == max_count else f'{min_count}&#8211;{max_count}',
        'replicates per variant',
    ))
    figures.append((f'{mean_count:.1f}', 'mean replicates'))
    if has_spread:
        fig = [f'{n_at_min:,}',
               f'{"variant" if n_at_min == 1 else "variants"} at {min_count}, the fewest']
        figures.append(tuple(fig + ['flag']) if flag_low else tuple(fig))

    figure_items = []
    for fig in figures:
        cls = ' class="flag"' if len(fig) > 2 else ''
        figure_items.append(
            f'<li{cls}><span class="fig-v">{fig[0]}</span>'
            f'<span class="fig-l">{fig[1]}</span></li>'
        )
    figures_html = ''.join(figure_items)
    byline = f'Generated {date.today().strftime("%-d %B %Y")}'

    # ── Charts ────────────────────────────────────────────────────────────────
    low_names = sorted(v for v in count_of if count_of[v] == min_count)
    hist, hist_x = _replicate_histogram(
        list(count_of.values()), min_count, max_count, low_names, flag_low
    )

    # The second chart groups by whatever the run actually distinguishes: the
    # library's own categories if it has them, otherwise the device's blocks.
    group_chart = ''
    buckets = {}
    group_heading = ''
    if cat_of:
        for v, c in count_of.items():
            buckets.setdefault(cat_of[v], []).append(c)
        group_heading = 'category'
    elif blocked:
        span = n_cols // n_blocks
        block_of = {}
        for r in range(n_rows):
            for c in range(n_cols):
                v = str(print_array[r, c])
                if v and v not in block_of:
                    block_of[v] = f'Block {c // span + 1}'
        for v, c in count_of.items():
            if v in block_of:
                buckets.setdefault(block_of[v], []).append(c)
        group_heading = 'block'

    if len(buckets) > 1:
        ordered = sorted(buckets.items(), key=lambda kv: (min(kv[1]), kv[0]))
        if len(ordered) > 8:
            rest = [c for _, vals in ordered[8:] for c in vals]
            ordered = ordered[:8] + [('Other', rest)]
        group_chart = (
            f'<h3>Replicates by {group_heading}</h3>'
            + _group_ranges(ordered, min_count, group_heading, hist_x, flag_low)
        )

    legend_html = ''.join(legend_blocks)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html_escape(project_name)} print check</title>
  <style>{_DASHBOARD_CSS}</style>
</head>
<body style="--chip-ratio:{view_w / view_h:.6f}">
  <div class="sheet">
    <header>
      <h1>{html_escape(project_name)}</h1>
      <p class="byline">{byline}</p>
      <ul class="figures">{figures_html}</ul>
    </header>

    <div class="layout">
      <section class="counts">
        <h2>Replicates per variant</h2>
        {hist}
        {group_chart}
        <div class="chart-tip" id="chart-tip" hidden></div>
      </section>

      <section class="map" aria-label="Array layout">
        <div class="chip"><div class="plot">{svg}</div></div>
        <div class="map-key">
          {legend_html}
          <div class="readout" id="chip-readout" aria-live="polite"></div>
        </div>
      </section>
    </div>
  </div>
  <script>{fit_js}{map_js}{_CHART_JS}{refresh_js}</script>
</body>
</html>"""

    with open(out_path, 'w') as f:
        f.write(page)
    return out_path


def _empty_dashboard(project_name, n_rows, n_cols):
    """Sheet shown when the print array holds no samples at all."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html_escape(project_name)} print check</title>
  <style>{_DASHBOARD_CSS}</style>
</head>
<body>
  <div class="sheet">
    <h1>{html_escape(project_name)}</h1>
    <p class="empty">The {n_rows} by {n_cols} array came back empty, so there is nothing to
    check yet. Confirm the library CSV has a <strong>Name</strong> column with values in it,
    then generate the array again.</p>
  </div>
</body>
</html>"""


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