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
import matplotlib.transforms
import matplotlib.patches
from adjustText import adjust_text
import textalloc as ta
import colorsys
import json
import re
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
_EMPTY = '#dde4e2'        # any chamber with no sample in it

# Sequential, one hue, light to dark, used when no category column is supplied.
_COUNT_RAMP = ['#9ED3CC', '#7BC3BB', '#59B0A8', '#3E9891', '#2C7C77', '#1D605E', '#124746']

# One colour per category: matplotlib's tab20, saturated half first, then the
# light half, with its grey pair dropped because grey already means an empty
# chamber or an uncategorized variant.
_CATEGORY_COLOURS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5',
]
_UNGROUPED = '#8C9A98'    # 'Other' and 'Uncategorized'

# Grid units per cell in the map's viewBox. The SVG scales to whatever box CSS
# gives it, so this only sets the internal proportions.
_U = 10
_GAP = 0.9


def _category_colour(i):
    """
    The colour for the i-th category.

    The first eighteen come from tab20. Past those, hues are generated by
    golden-angle rotation, which keeps every category distinct in name though
    no longer reliably by eye; the legend and hover tracing carry the rest.
    """
    if i < len(_CATEGORY_COLOURS):
        return _CATEGORY_COLOURS[i]
    h = (0.07 + (i - len(_CATEGORY_COLOURS)) * 0.381966) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, 0.55, 0.60)
    return '#%02x%02x%02x' % (round(r * 255), round(g * 255), round(b * 255))


def _hex_to_rgb(value):
    return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))


def _natural_key(name):
    """Sort key that orders digit runs by value, so scan2 precedes scan10."""
    return [(0, int(part), '') if part.isdigit() else (1, 0, part.lower())
            for part in re.split(r'(\d+)', str(name)) if part]


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
  var catg = document.getElementById('chip-cat');
  var out = document.getElementById('chip-readout');
  var C = __C__, R = __R__, U = __U__, S = __S__;
  var VW = __VW__, VH = __VH__, PL = __PL__, PT = __PT__;
  var GRID = __GRID__, V = __V__, ROWS = __ROWS__;

  // The source plate, when there is one, shares the chip's variant indices, so
  // a hover on either side can light the other.
  var PLATE = __PLATE__;
  var psvg = document.getElementById('plate');
  var phl = document.getElementById('plate-hl');
  var PTRACE = [];
  if (PLATE) {
    for (var w = 0; w < PLATE.W.length; w++) {
      if (PLATE.W[w] >= 0) (PTRACE[PLATE.W[w]] = PTRACE[PLATE.W[w]] || []).push(w);
    }
  }

  // pairs: [[well index, colour], ...]; empty or null clears the plate.
  function plateMark(pairs) {
    if (!psvg) return;
    if (!pairs || !pairs.length) {
      phl.innerHTML = '';
      psvg.classList.remove('tracing');
      return;
    }
    var html = '', j, w;
    for (j = 0; j < pairs.length; j++) {
      w = pairs[j][0];
      html += '<circle cx="' + ((w % PLATE.C) + 0.5) * PLATE.U +
              '" cy="' + (Math.floor(w / PLATE.C) + 0.5) * PLATE.U +
              '" r="' + PLATE.r + '" fill="' + pairs[j][1] + '"/>';
    }
    phl.innerHTML = html;
    psvg.classList.add('tracing');
  }

  function sourcePairs(vi) {
    return (PTRACE[vi] || []).map(function (w) { return [w, V[vi].c]; });
  }

  function sourceWells(vi) {
    var list = (PTRACE[vi] || []).map(function (w) {
      return PLATE.L[w] + (PLATE.M[w] ? ' (by hand)' : '');
    });
    if (!list.length) return '';
    var shown = list.slice(0, 3).join(', ') + (list.length > 3 ? ' +' + (list.length - 3) : '');
    return ' · well' + (list.length > 1 ? 's ' : ' ') + shown;
  }

  var TRACE = [];
  for (var i = 0; i < GRID.length; i++) {
    var g = GRID[i];
    if (g >= 0) (TRACE[g] = TRACE[g] || []).push(i);
  }

  var OUTSIDE = -9;
  var cur = OUTSIDE;
  // Set while a legend key or the empty-chambers row is held, so moving the
  // pointer over the array does not tear down that highlight.
  var locked = false;

  function cellPath(list) {
    var d = '', j, p, x, y;
    for (j = 0; j < list.length; j++) {
      p = list[j];
      x = (p % C) * U + (U - S) / 2;
      y = Math.floor(p / C) * U + (U - S) / 2;
      d += 'M' + x + ' ' + y + 'h' + S + 'v' + S + 'h-' + S + 'z';
    }
    return d;
  }

  function matching(code) {
    var list = [];
    for (var i = 0; i < GRID.length; i++) if (GRID[i] === code) list.push(i);
    return list;
  }

  function paintBlank(code) {
    var d = cellPath(matching(code));
    catg.innerHTML = d ? '<path fill="#0e2b2b" d="' + d + '"/>' : '';
    svg.classList.add('tracing');
  }

  function say(html) {
    if (!out) return;
    out.innerHTML = html + '<span class="ro-at"></span>';
    out.classList.add('on');
  }

  function clear() {
    cur = OUTSIDE;
    hl.setAttribute('d', '');
    if (locked) return;
    plateMark(null);
    catg.innerHTML = '';
    svg.classList.remove('tracing');
    if (out) out.classList.remove('on');
  }

  function move(e) {
    var b = svg.getBoundingClientRect();
    if (!b.width || !b.height) return;
    var c = Math.floor(((e.clientX - b.left) / b.width * VW - PL) / U);
    var r = Math.floor(((e.clientY - b.top) / b.height * VH - PT) / U);
    var inside = c >= 0 && c < C && r >= 0 && r < R;
    var vi = inside ? GRID[r * C + c] : OUTSIDE;

    if (vi !== cur) {
      cur = vi;
      if (vi === OUTSIDE) {
        clear();
      } else if (vi < 0) {
        hl.setAttribute('d', '');
        if (!locked) { paintBlank(vi); plateMark(null); }
        say('<strong>Empty chamber</strong><em>no sample printed</em>');
      } else {
        if (!locked) {
          catg.innerHTML = '';
          svg.classList.add('tracing');
        }
        hl.setAttribute('d', cellPath(TRACE[vi]));
        hl.setAttribute('fill', V[vi].c);
        if (!locked) plateMark(sourcePairs(vi));
        say('<span class="ro-sw" style="background:' + V[vi].c + '"></span>' +
            '<strong>' + V[vi].n + '</strong>' +
            (V[vi].g ? '<em>' + V[vi].g + '</em>' : '') +
            '<span class="ro-n">' + TRACE[vi].length + ' replicates' +
            (PLATE ? sourceWells(vi) : '') + '</span>');
      }
    }

    if (inside && out) {
      var at = out.querySelector('.ro-at');
      if (at) at.textContent = 'row ' + ROWS[r] + ', column ' + (c + 1);
    }
  }

  svg.addEventListener('mousemove', move);
  svg.addEventListener('mouseleave', clear);

  // Hovering a legend key holds every spot in that category and drops the rest
  // back, the same read as hovering a single variant but one group wider.
  document.querySelectorAll('.map-key .key[data-cat]').forEach(function (key) {
    key.addEventListener('mouseenter', function () {
      var want = +key.getAttribute('data-cat'), byColour = {};
      for (var i = 0; i < GRID.length; i++) {
        var v = GRID[i];
        if (v < 0 || V[v].k !== want) continue;
        (byColour[V[v].c] = byColour[V[v].c] || []).push(i);
      }
      var html = '';
      for (var col in byColour) {
        html += '<path fill="' + col + '" d="' + cellPath(byColour[col]) + '"/>';
      }
      catg.innerHTML = html;
      svg.classList.add('tracing');
      if (PLATE) {
        var pairs = [];
        for (var w = 0; w < PLATE.W.length; w++) {
          var pv = PLATE.W[w];
          if (pv >= 0 && V[pv].k === want) pairs.push([w, V[pv].c]);
        }
        plateMark(pairs);
      }
      locked = true;
    });
    key.addEventListener('mouseleave', function () { locked = false; clear(); });
  });

  // The blank legend keys and the empty-chambers row light up the cells they
  // describe, drawn dark so they read against the faded field.
  function holdBlank(el, code) {
    el.addEventListener('mouseenter', function () {
      paintBlank(code);
      locked = true;
    });
    el.addEventListener('mouseleave', function () { locked = false; clear(); });
  }
  document.querySelectorAll('[data-empty]').forEach(function (row) {
    holdBlank(row, -1);
  });
  document.querySelectorAll('.map-key .key[data-blank]').forEach(function (key) {
    holdBlank(key, +key.getAttribute('data-blank'));
  });

  // A block label on the map, or a row of the region table, veils everything
  // outside the columns it names.
  var mask = document.getElementById('chip-mask');
  var W = C * U, H = R * U;
  function holdColumns(el) {
    el.addEventListener('mouseenter', function () {
      var x0 = +el.getAttribute('data-c0') * U,
          x1 = +el.getAttribute('data-c1') * U, d = '';
      if (x0 > 0) d += 'M0 0h' + x0 + 'v' + H + 'h-' + x0 + 'z';
      if (x1 < W) d += 'M' + x1 + ' 0h' + (W - x1) + 'v' + H + 'h-' + (W - x1) + 'z';
      if (!d) return;
      mask.setAttribute('d', d);
      svg.classList.add('masked');
    });
    el.addEventListener('mouseleave', function () {
      svg.classList.remove('masked');
    });
  }
  svg.querySelectorAll('g.blk').forEach(holdColumns);
  document.querySelectorAll('.regions tr[data-c0]').forEach(holdColumns);

  if (psvg && PLATE) {
    var pcur = OUTSIDE;
    var plateLeave = function () { pcur = OUTSIDE; clear(); };
    psvg.addEventListener('mousemove', function (e) {
      var b = psvg.getBoundingClientRect();
      if (!b.width || !b.height) return;
      var c = Math.floor(((e.clientX - b.left) / b.width * PLATE.VW - PLATE.PL) / PLATE.U);
      var r = Math.floor(((e.clientY - b.top) / b.height * PLATE.VH - PLATE.PT) / PLATE.U);
      var w = (c >= 0 && c < PLATE.C && r >= 0 && r < PLATE.R) ? r * PLATE.C + c : OUTSIDE;
      if (w === pcur) return;
      pcur = w;
      if (w === OUTSIDE) { clear(); return; }
      var vi = PLATE.W[w];
      catg.innerHTML = '';
      cur = OUTSIDE;
      if (vi >= 0) {
        svg.classList.add('tracing');
        hl.setAttribute('d', cellPath(TRACE[vi] || []));
        hl.setAttribute('fill', V[vi].c);
        plateMark(sourcePairs(vi));
        say('<span class="ro-sw" style="background:' + V[vi].c + '"></span>' +
            '<strong>' + V[vi].n + '</strong>' +
            (V[vi].g ? '<em>' + V[vi].g + '</em>' : '') +
            '<span class="ro-n">' + (TRACE[vi] || []).length + ' replicates</span>' +
            (PLATE.M[w] ? '<span class="ro-hand">load ' + PLATE.M[w] + ' by hand</span>' : ''));
      } else {
        hl.setAttribute('d', '');
        svg.classList.remove('tracing');
        plateMark([[w, 'none']]);
        say(PLATE.N[w]
            ? '<strong>' + PLATE.N[w] + '</strong><em>in the plate, not printed on this chip</em>'
            : '<strong>Empty well</strong><em>no sample recorded</em>');
      }
      var at = out && out.querySelector('.ro-at');
      if (at) at.textContent = 'well ' + PLATE.L[w];
    });
    psvg.addEventListener('mouseleave', plateLeave);
  }
})();
"""


_DASHBOARD_CSS = """
*, *::before, *::after { box-sizing: border-box; }

:root {
  --ground: #ffffff;
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
  padding: 24px clamp(18px, 3vw, 44px) 26px;
  background: var(--ground);
  color: var(--ink);
  font-family: var(--font-ui);
  font-size: 16px;
  line-height: 1.5;
  font-feature-settings: 'tnum' 1;
}

.sheet { max-width: 1520px; margin: 0 auto; }

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
  font-size: 0.8125rem;
  font-weight: 700;
  line-height: 1.2;
  letter-spacing: -0.01em;
  font-variant-numeric: tabular-nums;
}

.fig-l {
  display: block;
  margin-top: 1px;
  font-size: 0.625rem;
  line-height: 1.25;
  color: var(--ink-muted);
}



/* Two columns for the life of the page: the array is a column of its own and
   runs the full height, so it is on screen whatever the window is doing. The
   header and the charts share the other column and wrap inside it. */
.page {
  display: grid;
  grid-template-columns: minmax(240px, 1fr) auto;
  gap: clamp(14px, 2.5vw, 40px);
  align-items: start;
}

.col-main { display: flex; flex-direction: column; min-width: 0; }

/* The array is bound by height, so spare width goes to the charts: they sit
   side by side and the column stays short enough not to scroll. */
.counts { margin-top: 18px; }

/* ── The array map: the one dark object on the page ───────────────────────── */
.map {
  max-height: var(--avail);
  display: flex;
  align-items: stretch;
  justify-content: start;
  min-width: 0;
  gap: 16px;
}

/* Label tracks are CSS grids over the same row and column counts as the SVG, so
   they stay registered to the cells at any size while the text itself keeps a
   fixed pixel size instead of shrinking with the map. */
.chip {
  min-width: 0;
  --plot-h: max(220px, calc(min(var(--avail), 1100px) - 12px));
  --plot-w: calc(var(--plot-h) * var(--chip-ratio));
}

.plot { position: relative; width: var(--plot-w); max-width: 100%; }

#chip { display: block; width: 100%; height: auto; }

#chip .cells { transition: opacity 170ms ease-out; }

#chip.tracing .cells { opacity: 0.18; }

#chip-hl, #chip-mask, #chip-cat { pointer-events: none; }

/* ── Source plate: in the chip's colours, hover-linked to it ──────────────── */
/* The plate and the array are what setup is read from, so the plate is placed
   first and pinned to the grid's last column: top right, against the array,
   whenever there are two or more columns, and straight under the header when
   there is one. Either way it is in view without scrolling the chart panel.
   It is drawn in the charts' coordinates, so it lines up with them edge for
   edge with no margins to tune. */
.plate { order: -1; grid-column: -2 / -1; min-width: 0; container-type: inline-size; }
/* Centred under the frame rather than the whole cell: the frame spans x 30 to
   286 of 300, so its centre sits 8/300 right of the cell's. */
/* Sized like the plate's axis title, 10 of the 300 units across, so the key
   scales with the drawing instead of looming over it in a narrow cell. */
.plate-key {
  margin: 2px 0 0;
  padding-left: 5.33%;
  font-size: 11px;
  font-size: clamp(8px, 3.4cqw, 12px);
  color: var(--ink-muted);
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 2px 14px;
}
.plate-key .ci { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
.plate-key .ring {
  display: inline-block; width: 0.75em; height: 0.75em; border-radius: 50%;
  box-sizing: border-box; border: 1px solid #b9c4c2;
}
#plate { display: block; width: 100%; height: auto; }
#plate .wells { transition: opacity 170ms ease-out; }
#plate.tracing .wells { opacity: 0.18; }
#plate-hl { pointer-events: none; }
#plate-hl circle { stroke: var(--ink); stroke-width: 0.9; }
#plate .unused { fill: none; stroke: #b9c4c2; stroke-width: 0.7; }
#plate .none { fill: #e4eae8; }
#plate .pframe { fill: none; stroke: var(--ink); stroke-width: 0.8; }
#plate .hand { pointer-events: none; }
#plate .hand circle { fill: none; stroke: var(--flag); stroke-width: 1.3; }
.plate-key .ci.hand { color: var(--flag); font-weight: 600; }
.plate-key .hring {
  display: inline-block; width: 0.8em; height: 0.8em; border-radius: 50%;
  box-sizing: border-box; border: 2px solid var(--flag);
}
.readout .ro-hand { display: block; color: var(--flag); font-weight: 600; }
#plate .plab {
  fill: var(--ink-muted);
  font-family: var(--font-ui);
  font-size: 9.5px;
  font-variant-numeric: tabular-nums;
}

#chip-mask { opacity: 0; transition: opacity 170ms ease-out; }

#chip.masked #chip-mask { opacity: 0.78; }

/* Block boundaries surface only while something is highlighted -- useful for
   reading where a variant falls, but not worth ruling across a resting map. */
.spine { fill: none; stroke: var(--ink); stroke-width: 0.8; }

.blk-sep {
  stroke: var(--ink);
  stroke-width: 1.4;
  opacity: 0;
  transition: opacity 170ms ease-out;
}

#chip.tracing .blk-sep, #chip.masked .blk-sep { opacity: 0.9; }

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
/* Zero intrinsic width, stretched to the column: the legend alone sets how wide
   the side column is. Sized by its own text, a long readout widened the
   column, which narrowed the charts and moved the plate under the pointer --
   onto another well, whose shorter text narrowed it back, and so on. */
.readout {
  width: 0;
  min-width: 100%;
  box-sizing: border-box;
  overflow: hidden;
  overflow-wrap: anywhere;
  height: 78px;
  padding: 6px 7px;
  border: 1px solid var(--rule);
  border-radius: 2px;
  background: var(--surface);
  font-size: 0.72rem;
  line-height: 1.35;
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
  font-size: 0.8125rem;
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
  width: 10px;
  height: 10px;
  margin-left: 5px;
  border: 1px solid rgba(14, 43, 43, 0.5);
}

.side {
  flex: 0 1 auto;
  align-self: start;
  margin-top: var(--grid-top, 0px);
  width: max-content;
  max-width: min(200px, 26vw);
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 7px;
}

/* Compact, in the manner of a matplotlib legend: a hairline frame, small
   markers, tight leading, no more room than the labels need. */
.map-key {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 5px 7px 6px;
  border: 1px solid var(--rule);
  border-radius: 2px;
  font-size: 0.75rem;
  line-height: 1.3;
  color: var(--ink);
}

.key { display: flex; align-items: center; gap: 6px; min-width: 0; }

.map-key .key[data-cat],
.map-key .key[data-blank] { cursor: default; transition: color 130ms ease-out; }

.map-key .key[data-cat]:hover .key-name,
.map-key .key[data-blank]:hover .key-name { color: var(--flag); }

.key-head {
  margin: 5px 0 0;
  font-size: 0.6875rem;
  font-weight: 700;
  color: var(--ink);
}

.map-key > .key-head:first-child { margin-top: 0; }

.key-sub {
  margin: 4px 0 0;
  font-size: 0.6875rem;
  color: var(--ink-muted);
}

.map-key > .key-sub:first-child { margin-top: 0; }

.key.spaced { margin-top: 5px; }

.key.nested { padding-left: 8px; }

.key-steps { display: flex; gap: 2px; }

.swatch {
  height: 10px;
  flex: none;
  border: 1px solid rgba(14, 43, 43, 0.5);
}

.swatch { width: 10px; margin-right: 12px; }

.key-name { color: var(--ink); min-width: 0; overflow-wrap: anywhere; }

.key-n { color: var(--ink-muted); font-variant-numeric: tabular-nums; }

.key-n::before { content: " "; }

.key-label {
  font-variant-numeric: tabular-nums;
  color: var(--ink);
}

/* ── Replicate panel ──────────────────────────────────────────────────────── */
.counts {
  position: relative;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(225px, 1fr));
  /* Rows take their content's full height and never less: when the panel is
     short the panel scrolls. Left to shrink (Safari does, under max-height),
     a row squeezed below its chart let the SVG spill over the next one. Dense
     flow back-fills the cells beside the plate, which is placed first. */
  grid-auto-rows: max-content;
  grid-auto-flow: row dense;
  gap: 0 20px;
  align-content: start;
  min-width: 0;
  min-height: 0;
  max-height: var(--charts-avail, var(--avail));
  overflow-y: auto;
  overflow-x: hidden;
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
  /* Held at its natural size: the panel scrolls rather than squeezing the
     charts when something else is added below them. */
  flex: 0 0 auto;
  width: 100%;
  height: auto;
  aspect-ratio: var(--chart-ratio);
  min-height: 0;
  align-self: start;
  margin: 0;
  overflow: visible;
}

.chart text { font-family: var(--font-ui); }

.bar { fill: var(--ink); transition: fill-opacity 140ms ease-out; }

.bar.flag { fill: var(--flag); }

.bar:hover { fill-opacity: 0.78; }

.axis { fill: none; stroke: var(--ink); stroke-width: 1; }

.ax { fill: var(--ink-muted); font-size: 10.5px; font-variant-numeric: tabular-nums; }

.axtitle { fill: var(--ink-muted); font-size: 10px; }

.callout { fill: var(--flag); font-size: 11px; font-weight: 700; }

.lead { stroke: var(--flag); stroke-width: 1; stroke-dasharray: 2 3; }

.lead.plain { stroke: var(--ink-muted); }

.callout.plain { fill: var(--ink-muted); font-weight: 600; }

.rowlab { fill: var(--ink); font-size: 10.5px; }

.rowlab.dim { fill: var(--ink-muted); }

.swarm { fill: var(--ink); fill-opacity: 0.78; }

.swarm-over { fill: var(--ink); fill-opacity: 0.5; }

g[data-tip]:hover .swarm { fill: var(--flag); fill-opacity: 1; }

/* Round caps so a degenerate middle half -- every variant on the same count --
   still reads as a mark rather than vanishing. */
.iqr { stroke: #9fb0ad; stroke-width: 6; stroke-linecap: round; }

.mid { fill: var(--ink); stroke: #fff; stroke-width: 1.2; }

.rowval {
  fill: var(--ink-muted);
  font-size: 10px;
  font-variant-numeric: tabular-nums;
  paint-order: stroke;
  stroke: var(--surface);
  stroke-width: 3px;
  stroke-linejoin: round;
}

.rowval.flag { fill: var(--flag); font-weight: 600; }

.hit { fill: transparent; }

/* Per-region statistics: a handful of exact numbers, so a table rather than a
   chart. Hovering a row holds that region on the map. */
.rtoggle { position: absolute; width: 0; height: 0; opacity: 0; }

.rmore {
  grid-column: 1 / -1;
  justify-self: start;
  flex: none;
  margin-top: 14px;
  padding-bottom: 1px;
  font-size: 0.75rem;
  color: var(--ink-muted);
  cursor: pointer;
  border-bottom: 1px solid var(--rule);
}

.rmore:hover { color: var(--ink); border-bottom-color: var(--ink); }

.rtoggle:focus-visible ~ .rmore { outline: 2px solid var(--ink); outline-offset: 3px; }

.rmore .shut { display: none; }

.rtoggle:checked ~ .rmore .open { display: none; }

.rtoggle:checked ~ .rmore .shut { display: inline; }

.rwrap { display: none; grid-column: 1 / -1; }

.rtoggle:checked ~ .rwrap { display: block; }

.regions {
  width: 100%;
  margin-top: 10px;
  border-collapse: collapse;
  font-size: 0.75rem;
}

.regions th {
  padding: 0 0 3px;
  text-align: left;
  font-weight: 500;
  font-size: 0.6875rem;
  color: var(--ink-muted);
  border-bottom: 1px solid var(--rule);
}

.regions td { padding: 3px 0; font-variant-numeric: tabular-nums; }

.regions th.num, .regions td.num { text-align: right; }

.regions tr.half td { border-top: 1px solid var(--rule); padding-top: 5px; }

.regions tbody tr { cursor: default; transition: color 130ms ease-out; }

.regions tbody tr:hover { color: var(--flag); }

.cbar { rx: 1; }

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

/* Only a genuinely tiny window stacks, and then the array goes first so it is
   still the thing you see. */
@media (max-width: 520px) {
  .page { grid-template-columns: minmax(0, 1fr); }
  .map { order: -1; }
  .counts, .map { max-height: none; }
}

@media (prefers-reduced-motion: reduce) {
  #chip .cells, .bar, #chip-mask, .readout, .blk-sep { transition: none; }
}
"""


# ── Charts ────────────────────────────────────────────────────────────────────
# Both charts are inline SVG on the paper panel and share one horizontal scale,
# so a group's range in the lower chart lines up with the distribution above it.
_CHART_W = 300
_PLOT_TOP = 28     # top of every chart's plotting area; room above for the callout
# Height of every chart's plotting area, the plate's frame included. The plate
# is 3:2 (96, 384 and 1536 wells alike) and spans the plot's full width, so the
# box is as tall as that makes it: 24 columns across the 256-unit plot, less a
# 1.5-unit inset each side, gives 16 rows of 10.54 = 168.7, plus the insets.
# One box for every cell is what lets charts and plate share edges.
_PLOT_H = 172
_AXIS_Y = _PLOT_TOP + _PLOT_H
_CHART_H = _AXIS_Y + 30
_CHART_L = 30      # left gutter, just the y labels
_CHART_R = 14
_HIST_H = _PLOT_H  # the distribution fills the shared plotting area
_HIST_TOP = _PLOT_TOP
_ROW_H = 26        # one group per row in the range chart
_LAB_IN = 4        # row labels start this far in from the axis rule


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
    bar_w = min(max(1.6, slot - 1.4), 26)

    def y_of(v):
        return base - (v / top) * _HIST_H

    grid, yl = [], []
    for v in range(0, top + 1, tick):
        y = y_of(v)
        grid.append(f'<line class="tick" x1="{_CHART_L - 4}" y1="{y:.1f}" '
                    f'x2="{_CHART_L}" y2="{y:.1f}"/>')
        yl.append(f'<text class="ax" x="{_CHART_L - 7}" y="{y + 3.5:.1f}" '
                  f'text-anchor="end">{v}</text>')
    grid.append(
        f'<path class="axis" d="M{_CHART_L} {y_of(top):.1f}V{base:.1f}'
        f'H{_CHART_L + plot_w:.1f}"/>'
    )

    bars, xl = [], []
    # The charts sit three to a row now, so labels need room: about ten across
    # the axis at most.
    label_every = max(1, -(-len(freq) // 10))
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
            cx = _CHART_L + i * slot + slot / 2
            xl.append(f'<line class="tick" x1="{cx:.1f}" y1="{base:.1f}" '
                      f'x2="{cx:.1f}" y2="{base + 3.5:.1f}"/>'
                      f'<text class="ax" x="{cx:.1f}" y="{_AXIS_Y + 13:.0f}" '
                      f'text-anchor="middle">{labels[i]}</text>')

    # The point of the chart is which variant is short, so it gets named on the mark.
    callout = ''
    if low_names and freq[low_bin]:
        cx = _CHART_L + low_bin * slot + slot / 2
        names = ', '.join(html_escape(n) for n in low_names)
        wordy = f'fewest: {names} at {min_count}'
        brief = (f'fewest at {min_count}' if len(low_names) == 1
                 else f'{len(low_names):,} at {min_count}, the fewest')

        # Try the full label on either side of the mark, then the short form; a
        # narrow chart cannot always hold the names, and running off an edge is
        # worse than not naming them.
        text, tx, anchor = brief, cx + 6, ''
        for candidate in ((wordy, True), (wordy, False), (brief, True), (brief, False)):
            label, right = candidate
            width = len(label) * 5.9
            x = cx + 6 if right else cx - 6
            lo, hi = (x, x + width) if right else (x - width, x)
            if lo >= 2 and hi <= _CHART_W - 2:
                text, tx, anchor = label, x, '' if right else ' text-anchor="end"'
                break
        else:
            tx = min(max(cx + 6, 2), _CHART_W - 2 - len(brief) * 5.9)

        cls = 'callout' if flag_low else 'callout plain'
        callout = (
            f'<line class="lead{"" if flag_low else " plain"}" x1="{cx:.1f}" '
            f'y1="{min(y_of(freq[low_bin]), base - 2.4) - 5:.1f}" '
            f'x2="{cx:.1f}" y2="{_HIST_TOP - 12:.0f}"/>'
            f'<text class="{cls}" x="{tx:.1f}" y="{_HIST_TOP - 8:.0f}"{anchor}>{text}</text>'
        )

    def x_of(v):
        return _CHART_L + (v - edges[0]) / (edges[1] - edges[0]) * slot + slot / 2

    height = _CHART_H
    svg = (
        f'<svg class="chart" viewBox="0 0 {_CHART_W} {height}" '
        f'role="img" aria-label="Distribution of replicate counts across variants. '
        f'Replicates range from {min_count} to {max_count}.">'
        f'{"".join(grid)}{"".join(bars)}{"".join(yl)}{"".join(xl)}{callout}'
        f'<text class="axtitle" x="{_CHART_L + plot_w / 2:.0f}" y="{_AXIS_Y + 24:.0f}" '
        f'text-anchor="middle">replicates per variant</text>'
        f'<text class="axtitle" x="1" y="8">variants</text>'
        f'</svg>'
    )
    return svg, x_of


def _group_ranges(groups, min_count, heading, x_of, flag_low, empty=0):
    """
    Replicate count for every member of each group, as a dot swarm.

    Counts are whole numbers, so the swarm is really a dot histogram: members
    sharing a count stack symmetrically about their row's centre line. Dot size
    comes from the tallest stack so no row spills into its neighbour, and where
    a group is large enough that the dots would shrink below legibility the
    stack is capped and the remainder drawn as a bar.

    The plotting area matches the other charts exactly, so a row of them shares
    tick and title lines and a column of them shares its left edge.
    """
    x0, x1 = float(_CHART_L), float(_CHART_W - _CHART_R)
    lo = min_count
    hi = max(max(v for _, vals in groups for v in vals), lo + 1)

    slots = hi - lo + 1

    def xg(v):
        return x0 + (v - lo + 0.5) / slots * (x1 - x0)

    stacks = []
    for _, vals in groups:
        seen = {}
        for v in vals:
            seen[v] = seen.get(v, 0) + 1
        stacks.append(seen)

    n_rows = len(groups) + (1 if empty else 0)
    row_h = _PLOT_H / max(n_rows, 1)
    band = max(6.0, row_h - 12)

    # Members sharing a count are packed into a small cluster rather than one
    # tall column: the gap between neighbouring counts is usually far wider than
    # a row is tall, so spreading sideways keeps the dots legible.
    dot = 2.5
    gap = (x1 - x0) / slots
    max_cols = max(1, int(gap / dot) - 1)
    max_rows = max(1, int(band / dot))
    cap = max_cols * max_rows

    rows = []
    for i, (name, vals) in enumerate(groups):
        top = _PLOT_TOP + i * row_h
        centre = top + 9 + band / 2
        g_lo, g_hi = min(vals), max(vals)
        mid = float(np.median(vals))
        flagged = g_lo == min_count and flag_low
        label = str(g_lo) if g_lo == g_hi else f'{g_lo}\u2013{g_hi}'
        tip = (f'<b>{html_escape(name)}</b><em>{len(vals)} variants, median '
               f'{mid:.0f}, range {g_lo}&#8211;{g_hi}</em>')

        marks = []
        for value, n in sorted(stacks[i].items()):
            cx = xg(value)
            shown = min(n, cap)
            cols = min(max_cols, -(-shown // max_rows))
            for k in range(shown):
                col, row = k % cols, k // cols
                n_in_col = len(range(col, shown, cols))
                dx = (col - (cols - 1) / 2) * dot
                dy = (row - (n_in_col - 1) / 2) * dot
                marks.append(f'<circle class="swarm" cx="{cx + dx:.1f}" '
                             f'cy="{centre + dy:.1f}" r="{dot * 0.4:.2f}"/>')
            if n > shown:
                # Past what the row can hold; the bar carries the remainder.
                marks.append(
                    f'<rect class="swarm-over" x="{cx - dot * 0.4:.1f}" '
                    f'y="{centre - band / 2:.1f}" width="{dot * 0.8:.2f}" '
                    f'height="{band:.1f}"/>'
                )

        rows.append(
            f'<g data-tip="{tip}">'
            f'<rect class="hit" x="0" y="{top:.1f}" width="{_CHART_W}" '
            f'height="{row_h:.1f}"/>'
            f'<text class="rowlab" x="{x0 + _LAB_IN:.0f}" y="{top + 6:.1f}">'
            f'{html_escape(name)}</text>'
            f'<text class="rowval{" flag" if flagged else ""}" x="{x1:.0f}" '
            f'y="{top + 6:.1f}" text-anchor="end">{label}</text>'
            + ''.join(marks) +
            '</g>'
        )

    # Empty chambers share the plot rather than trailing below it. They carry no
    # replicate count, so the row states the number instead of placing a mark.
    if empty:
        top = _PLOT_TOP + len(groups) * row_h
        rows.append(
            f'<g data-empty="1" data-tip="<b>{empty:,} empty chamber'
            f'{"" if empty == 1 else "s"}</b>'
            f'<em>addressable positions left unfilled</em>">'
            f'<rect class="hit" x="0" y="{top:.1f}" width="{_CHART_W}" '
            f'height="{row_h:.1f}"/>'
            f'<text class="rowlab dim" x="{x0 + _LAB_IN:.0f}" y="{top + 6:.1f}">'
            f'Empty chambers</text>'
            f'<text class="rowval" x="{x1:.0f}" y="{top + 6:.1f}" '
            f'text-anchor="end">{empty:,}</text>'
            '</g>'
        )

    marks = []
    for v in sorted({lo, (lo + hi) // 2, hi}):
        vx = xg(v)
        marks.append(
            f'<line class="tick" x1="{vx:.1f}" y1="{_AXIS_Y}" '
            f'x2="{vx:.1f}" y2="{_AXIS_Y + 3.5}"/>'
            f'<text class="ax" x="{vx:.1f}" y="{_AXIS_Y + 13}" '
            f'text-anchor="middle">{v}</text>'
        )
    rows.append(
        f'<path class="axis" d="M{x0:.0f} {_PLOT_TOP}V{_AXIS_Y}H{x1:.0f}"/>'
        + ''.join(marks)
        + f'<text class="axtitle" x="{(x0 + x1) / 2:.0f}" y="{_AXIS_Y + 24}" '
          f'text-anchor="middle">replicates by {heading}</text>'
    )

    return (
        f'<svg class="chart" viewBox="0 0 {_CHART_W} {_CHART_H}" role="img" '
        f'aria-label="Replicate count for every member of each {heading}, and '
        f'the number of empty chambers.">{"".join(rows)}</svg>'
    )


def legend_order(ordered_cats, size_of, category_groups=None):
    """
    Categories in the order the legend lists them.

    Sublibraries first, grouped so that everything sharing a second designation
    is contiguous, then the single sample types. Shared with the exported panel
    figure so its panels run in the same order as the key beside the map.
    """
    groups = category_groups or {}
    subs = [c for c in ordered_cats if size_of[c] > 1]
    singles = [c for c in ordered_cats if size_of[c] == 1]

    kinds = []
    for c in subs:
        kind = str(groups.get(c, '')) or 'Other'
        if kind not in kinds:
            kinds.append(kind)

    if groups and len(kinds) > 1:
        ranked = []
        for kind in kinds:
            ranked += [c for c in subs if (str(groups.get(c, '')) or 'Other') == kind]
        subs = ranked
    return subs + singles


def assign_variant_colours(count_of, categories=None, category_families=None):
    """
    Work out which colour every variant is drawn in.

    Shared by the HTML sheet and the static figures so they always agree.
    Returns (cat_of, variant_colour, cat_colour, shades_of, ordered_cats,
    size_of); every mapping is empty when no categories are supplied.

    Every variant in a category shares that category's colour. Colours are
    assigned in fixed order and never cycled, so no two categories share one.
    `category_families` pins named categories to a palette index, so a category
    keeps its colour as a library grows. `shades_of` maps each category to a
    one-item list holding that colour.
    """
    cat_of = {}
    if categories:
        for v in count_of:
            label = categories.get(v)
            cat_of[v] = str(label) if label not in (None, '') else 'Uncategorized'
    if not cat_of:
        return {}, {}, {}, {}, [], {}

    # Categories holding one variant are listed after the sublibraries rather
    # than interleaved alphabetically among them.
    size_of = {c: sum(1 for v in cat_of if cat_of[v] == c) for c in set(cat_of.values())}
    ordered_cats = sorted(
        size_of,
        key=lambda c: (c in ('Other', 'Uncategorized'), size_of[c] == 1, _natural_key(c)),
    )

    pins = dict(category_families or {})
    variant_colour, cat_colour, shades_of = {}, {}, {}
    for i, cat in enumerate(ordered_cats):
        if cat in ('Other', 'Uncategorized'):
            colour = _UNGROUPED
        elif cat in pins:
            colour = _category_colour(pins[cat])
        else:
            colour = _category_colour(i)
        shades_of[cat] = [colour]
        cat_colour[cat] = colour
        for v in count_of:
            if cat_of[v] == cat:
                variant_colour[v] = colour

    return cat_of, variant_colour, cat_colour, shades_of, ordered_cats, size_of


_COMP_H = 22       # one designed category per row


def _composition(rows):
    """
    Designed library members per category.

    Counts what the library set out to contain, so a category whose members
    never reached the array is visible next to one that printed in full. Shares
    the other charts' plotting area so rows and columns of charts line up.
    """
    x0, x1 = float(_CHART_L), float(_CHART_W - _CHART_R)
    top_n = max(n for _, n, _, _ in rows)
    row_h = _PLOT_H / max(len(rows), 1)
    bar_h = min(6.0, max(3.0, row_h - 14))

    out = []
    for i, (label, n, absent, colour) in enumerate(rows):
        y = _PLOT_TOP + i * row_h
        w = (x1 - x0) * n / top_n
        tip = (f'<b>{html_escape(label)}</b><em>{n:,} designed member'
               f'{"" if n == 1 else "s"}'
               + (f', {absent:,} not printed' if absent else ', all printed')
               + '</em>')
        out.append(
            f'<g data-tip="{tip}">'
            f'<rect class="hit" x="0" y="{y:.1f}" width="{_CHART_W}" '
            f'height="{row_h:.1f}"/>'
            f'<rect class="cbar" x="{x0:.0f}" y="{y + 10:.1f}" '
            f'width="{w:.1f}" height="{bar_h:.1f}" fill="{colour}"/>'
            f'<text class="rowlab" x="{x0 + _LAB_IN:.0f}" y="{y + 6:.1f}">'
            f'{html_escape(label)}</text>'
            f'<text class="rowval{" flag" if absent else ""}" x="{x1:.0f}" '
            f'y="{y + 6:.1f}" text-anchor="end">{n:,}'
            + (f' ({absent:,} unprinted)' if absent else '')
            + '</text>'
            '</g>'
        )

    marks = []
    for v in sorted({0, top_n}):
        vx = x0 + (x1 - x0) * v / top_n
        marks.append(
            f'<line class="tick" x1="{vx:.1f}" y1="{_AXIS_Y}" '
            f'x2="{vx:.1f}" y2="{_AXIS_Y + 3.5}"/>'
            f'<text class="ax" x="{vx:.1f}" y="{_AXIS_Y + 13}" '
            f'text-anchor="{"start" if v == 0 else "end"}">{v:,}</text>'
        )
    out.append(
        f'<path class="axis" d="M{x0:.0f} {_PLOT_TOP}V{_AXIS_Y}H{x1:.0f}"/>'
        + ''.join(marks)
        + f'<text class="axtitle" x="{(x0 + x1) / 2:.0f}" y="{_AXIS_Y + 24}" '
          f'text-anchor="middle">designed library members</text>'
    )
    return (
        f'<svg class="chart" viewBox="0 0 {_CHART_W} {_CHART_H}" role="img" '
        f'aria-label="Designed library members per category.">{"".join(out)}</svg>'
    )


_FIT_JS = """
(function () {
  var layout = document.querySelector('.page');
  var head = document.querySelector('.col-main header');
  var chip = document.querySelector('.chip');
  var plot = document.querySelector('.plot');
  if (!layout) return;
  var RATIO = __RATIO__, VW = __FITVW__, PT = __FITPT__;
  var root = document.documentElement;

  function fit() {
    var sy = window.scrollY || 0;
    var top = layout.getBoundingClientRect().top + sy;
    var room = Math.max(260, window.innerHeight - top - 28);
    root.style.setProperty('--avail', room + 'px');
    // The charts sit below the header inside the other column, so they get what
    // the header leaves; the array gets the whole height.
    var headH = head ? head.getBoundingClientRect().height + 18 : 0;
    root.style.setProperty('--charts-avail', Math.max(180, room - headH) + 'px');
    if (!chip || !plot) return;

    var h = Math.max(200, Math.min(room - 12, 1100));
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
      // The block labels sit above the grid inside the SVG, so the legend is
      // dropped by that band to line up with the array's own top edge.
      root.style.setProperty('--grid-top', (PT * k).toFixed(2) + 'px');
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


def _region_stats(print_array, n_blocks=None):
    """
    Variant statistics for each block and each half of the device.

    Blocks partition the array by column, so the halves are the column halves
    too -- named for the blocks they hold when the block count divides evenly.
    Counts are local to the region: a variant's spots inside that region, not
    its total across the device.

    Returns a list of (label, first_col, last_col, n_variants, lo, hi, n_empty),
    with the halves last.
    """
    n_rows, n_cols = print_array.shape
    regions = []

    blocked = bool(n_blocks) and n_blocks > 1 and n_cols % n_blocks == 0
    if blocked:
        span = n_cols // n_blocks
        for k in range(n_blocks):
            regions.append((f'Block {k + 1}', k * span, (k + 1) * span))

    mid = n_cols // 2
    if n_cols >= 2:
        if blocked and n_blocks % 2 == 0 and n_blocks > 2:
            half = n_blocks // 2
            left = f'Blocks 1&#8211;{half}'
            right = f'Blocks {half + 1}&#8211;{n_blocks}'
        else:
            left = f'Columns 1&#8211;{mid}'
            right = f'Columns {mid + 1}&#8211;{n_cols}'
        regions.append((left, 0, mid))
        regions.append((right, mid, n_cols))

    out = []
    for label, c0, c1 in regions:
        chunk = print_array[:, c0:c1]
        counts = {}
        for v in chunk.flatten():
            v = str(v)
            if v:
                counts[v] = counts.get(v, 0) + 1
        empty = int((chunk == '').sum())
        if counts:
            out.append((label, c0, c1, len(counts),
                        min(counts.values()), max(counts.values()), empty))
        else:
            out.append((label, c0, c1, 0, 0, 0, empty))
    return out


def _region_table(rows, n_blocks=None):
    """Compact table of the per-region statistics, hover-linked to the map."""
    if not rows:
        return ''
    blocked = bool(n_blocks) and n_blocks > 1
    body = []
    for i, (label, c0, c1, n_var, lo, hi, empty) in enumerate(rows):
        # The halves follow the blocks; a rule sets them apart.
        half = ' class="half"' if blocked and i == len(rows) - 2 else ''
        spots = f'{lo}' if lo == hi else f'{lo}&#8211;{hi}'
        body.append(
            f'<tr{half} data-c0="{c0}" data-c1="{c1}">'
            f'<td>{label}</td>'
            f'<td class="num">{n_var:,}</td>'
            f'<td class="num">{spots}</td>'
            f'<td class="num">{empty:,}</td>'
            '</tr>'
        )
    return (
        '<table class="regions">'
        '<thead><tr><th>Region</th><th class="num">Variants</th>'
        '<th class="num">Spots each</th><th class="num">Empty</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )


def plot_array_map(print_array, categories=None, n_blocks=None,
                   export_dir='.', project_name='print', save=True,
                   category_families=None, category_groups=None, dpi=300):
    """
    The whole array in one static figure, with its legend.

    A print-ready counterpart to the dashboard's map: every chamber in the
    colour the sheet gives it, blocks ruled and labelled, and a key beside it in
    the legend's own order.

    Args:
        print_array: Array containing variant names (post name-mapping)
        categories: Optional dict mapping variant name -> category label
        n_blocks: Optional number of blocks, ruled and labelled on the map
        export_dir: Directory to write into
        project_name: Stem for the written files
        save: Write <project>_array.png and .pdf
        category_families: Optional dict pinning a category to a palette index
        category_groups: Optional dict giving each category a second designation
        dpi: Raster resolution

    Returns:
        fig, ax
    """
    n_rows, n_cols = print_array.shape
    flat = print_array.flatten()
    present = flat[flat != '']
    unique, counts = np.unique(present, return_counts=True)
    count_of = {str(v): int(c) for v, c in zip(unique, counts)}

    cat_of, variant_colour, cat_colour, _, ordered_cats, size_of = (
        assign_variant_colours(count_of, categories, category_families)
    )

    img = np.ones((n_rows, n_cols, 3))
    empty_rgb = np.array(_hex_to_rgb(_EMPTY)) / 255.0
    for r in range(n_rows):
        for c in range(n_cols):
            v = str(print_array[r, c])
            img[r, c] = (np.array(_hex_to_rgb(variant_colour[v])) / 255.0
                         if v and v in variant_colour else empty_rgb)

    keys = legend_order(ordered_cats, size_of, category_groups)
    fig, ax = plt.subplots(figsize=(4.4 + 0.02 * max(len(k) for k in keys or ['x']), 7.4))
    ax.imshow(img, extent=(0.5, n_cols + 0.5, n_rows + 0.5, 0.5),
              interpolation='nearest', aspect='equal')

    x_step = 1 if n_cols <= 12 else 5
    y_step = 1 if n_rows <= 12 else 5
    ax.set_xticks(range(1, n_cols + 1, x_step))
    ax.set_yticks(range(1, n_rows + 1, y_step))
    ax.tick_params(labelsize=7, length=2.5, width=0.6, pad=4)
    for side in ax.spines.values():
        side.set_linewidth(0.8)

    blocked = bool(n_blocks) and n_blocks > 1 and n_cols % n_blocks == 0
    if blocked:
        span = n_cols // n_blocks
        trans = matplotlib.transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        for b in range(1, n_blocks):
            ax.axvline(b * span + 0.5, color='#0e2b2b', lw=1.1)
        for b in range(n_blocks):
            ax.plot([b * span + 0.7, (b + 1) * span + 0.3], [1.008, 1.008],
                    transform=trans, color='#0e2b2b', lw=1.2,
                    clip_on=False, solid_capstyle='butt')
            ax.text((b + 0.5) * span + 0.5, 1.018, f'Block {b + 1}',
                    transform=trans, ha='center', va='bottom',
                    fontsize=7.5, color='#0e2b2b')

    handles = [
        matplotlib.patches.Patch(facecolor=cat_colour.get(k, '#9fb0ad'),
                                 edgecolor='#0e2b2b', linewidth=0.5, label=k)
        for k in keys
    ]
    n_empty = int((print_array == '').sum())
    if n_empty:
        handles.append(matplotlib.patches.Patch(
            facecolor=_EMPTY, edgecolor='#0e2b2b', linewidth=0.5,
            label=f'Empty chambers ({n_empty:,})'))
    if handles:
        leg = ax.legend(
            handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0),
            frameon=True, fontsize=8, handlelength=1.5, handleheight=1.0,
            borderpad=0.6, labelspacing=0.5,
            edgecolor='#d0d9d6', facecolor='white',
        )
        leg.get_frame().set_linewidth(0.6)

    fig.patch.set_facecolor('white')
    fig.tight_layout()

    if save:
        os.makedirs(export_dir, exist_ok=True)
        stem = os.path.join(export_dir, f'{project_name}_array')
        fig.savefig(f'{stem}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
        fig.savefig(f'{stem}.pdf', bbox_inches='tight', facecolor='white')

    return fig, ax


def plot_library_panels(print_array, categories=None, n_blocks=None,
                        export_dir='.', project_name='print', save=True,
                        max_cols=8, dpi=300, category_families=None,
                        category_groups=None):
    """
    Small multiples of the array, one panel per sublibrary or sample type.

    Every panel shows the whole array with one group held at full colour and
    everything else ghosted, so a group's spread across the device can be read
    at a glance and the panels compared side by side. A final panel marks the
    empty chambers. Colours come from the same assignment the HTML sheet uses.

    Args:
        print_array: Array containing variant names (post name-mapping)
        categories: Optional dict mapping variant name -> category label
        n_blocks: Optional number of blocks, ruled and labelled on each panel
        export_dir: Directory to write into
        project_name: Stem for the written files
        save: Write <project>_library_panels.png and .pdf
        max_cols: Panels per row before wrapping
        dpi: Raster resolution
        category_groups: Optional dict mapping a category to a second
            designation; panels then run in the dashboard legend's order, with
            everything sharing a designation contiguous.
        category_families: Optional dict pinning a category to a palette index,
            so it keeps its colour across prints with different category sets

    Returns:
        fig, axes
    """
    n_rows, n_cols = print_array.shape
    flat = print_array.flatten()
    present = flat[flat != '']
    unique, counts = np.unique(present, return_counts=True)
    count_of = {str(v): int(c) for v, c in zip(unique, counts)}

    cat_of, variant_colour, cat_colour, _, ordered_cats, size_of = (
        assign_variant_colours(count_of, categories, category_families)
    )

    # One panel per category, in the order the dashboard legend lists them, so
    # sublibraries sharing a designation sit together; empties last.
    panels = []
    for cat in legend_order(ordered_cats, size_of, category_groups):
        members = [v for v in count_of if cat_of[v] == cat]
        spots = sum(count_of[v] for v in members)
        title = cat if size_of[cat] > 1 else f'{cat}\n(n={spots})'
        panels.append((title, set(members), None))
    printed_rows = ~np.all(print_array == '', axis=1)
    n_gaps = int((print_array[printed_rows] == '').sum())
    if n_gaps:
        panels.append((f'Empties\n(n={n_gaps:,})', None, 'gaps'))
    if not panels:
        raise ValueError('nothing to plot: the array holds no samples')

    # Ghosted base: every cell in its own colour, mostly washed out to white.
    ghost = np.ones((n_rows, n_cols, 3))
    ink = np.array(_hex_to_rgb('#0e2b2b')) / 255.0
    for r in range(n_rows):
        for c in range(n_cols):
            v = str(print_array[r, c])
            hexcol = variant_colour.get(v, _EMPTY) if v else _EMPTY
            rgb = np.array(_hex_to_rgb(hexcol)) / 255.0
            ghost[r, c] = 1.0 - (1.0 - rgb) * 0.16

    ncol = min(max_cols, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(1.95 * ncol, 3.6 * nrow), squeeze=False,
    )

    blocked = bool(n_blocks) and n_blocks > 1 and n_cols % n_blocks == 0
    span = n_cols // n_blocks if blocked else 0
    x_step = 1 if n_cols <= 12 else 5
    y_step = 1 if n_rows <= 12 else 5

    for k, ax in enumerate(axes.flat):
        if k >= len(panels):
            ax.axis('off')
            continue
        title, members, mode = panels[k]
        img = ghost.copy()
        for r in range(n_rows):
            for c in range(n_cols):
                v = str(print_array[r, c])
                if mode == 'gaps':
                    if v == '' and printed_rows[r]:
                        img[r, c] = ink
                elif v and v in members:
                    img[r, c] = np.array(_hex_to_rgb(variant_colour[v])) / 255.0

        ax.imshow(img, extent=(0.5, n_cols + 0.5, n_rows + 0.5, 0.5),
                  interpolation='nearest', aspect='equal')
        ax.set_title(title, fontsize=9, pad=16, linespacing=1.35)
        ax.set_xticks(range(1, n_cols + 1, x_step))
        ax.set_yticks(range(1, n_rows + 1, y_step))
        ax.tick_params(labelsize=5.5, length=2, width=0.5, pad=3)
        for side in ax.spines.values():
            side.set_linewidth(0.6)

        if blocked:
            for b in range(1, n_blocks):
                ax.axvline(b * span + 0.5, color='#0e2b2b', lw=0.9)
            trans = matplotlib.transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
            for b in range(n_blocks):
                ax.plot([b * span + 0.7, (b + 1) * span + 0.3], [1.012, 1.012],
                        transform=trans, color='#0e2b2b', lw=0.9,
                        clip_on=False, solid_capstyle='butt')
                ax.text((b + 0.5) * span + 0.5, 1.025, f'Block {b + 1}',
                        transform=trans, ha='center', va='bottom',
                        fontsize=5.4, color='#0e2b2b')

    fig.patch.set_facecolor('white')
    fig.tight_layout(w_pad=0.9, h_pad=1.8)

    # A shared key centred under the panels, so the figure stands on its own.
    keys = [
        matplotlib.patches.Patch(
            facecolor=cat_colour.get(cat, '#9fb0ad'), edgecolor='#0e2b2b',
            linewidth=0.5, label=cat)
        for cat in legend_order(ordered_cats, size_of, category_groups)
    ]
    if n_gaps:
        keys.append(matplotlib.patches.Patch(
            facecolor='#0e2b2b', edgecolor='#0e2b2b', linewidth=0.5,
            label='Empty chambers'))
    leg = fig.legend(
        handles=keys, loc='upper center', bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(keys), 6), frameon=True, fontsize=7.5,
        handlelength=1.5, handleheight=1.0, columnspacing=1.4,
        borderpad=0.5, edgecolor='#d0d9d6', facecolor='white',
    )
    leg.get_frame().set_linewidth(0.6)

    if save:
        os.makedirs(export_dir, exist_ok=True)
        stem = os.path.join(export_dir, f'{project_name}_library_panels')
        fig.savefig(f'{stem}.png', dpi=dpi, bbox_inches='tight',
                    facecolor='white')
        fig.savefig(f'{stem}.pdf', bbox_inches='tight', facecolor='white')

    return fig, axes

_PLATE_SIZES = [(8, 12), (16, 24), (32, 48)]   # 96, 384 and 1536-well plates


def _row_name(i):
    """Plate row label: A..Z, then AA, AB... for 1536-well plates."""
    return chr(65 + i) if i < 26 else 'A' + chr(65 + i - 26)


def _plate_map(plate, printed, idx_of, variant_color, manual=None):
    """
    Source plate drawn well by well, in the chip's colours.

    Returns (html, data) for the page and its hover script, or ('', None) when
    no plate was given. A well is filled in its variant's colour only when it
    was actually dispensed; wells that hold a sample but do not print on this
    chip are drawn as open rings, so a plate shared between chips reads
    correctly for each.

    `manual` maps wells loaded by hand at setup, rather than carried over from
    the library plate, to what goes in them. They get a ring in the flag colour
    that stays up through every hover state: they are the wells that get
    forgotten.
    """
    if not plate:
        return '', None
    manual = {str(w).strip().upper(): str(v) for w, v in (manual or {}).items()}

    cells = {}
    for well, name in plate.items():
        m = re.match(r'^([A-Z]{1,2})(\d+)$', str(well).strip().upper())
        if not m:
            raise ValueError(f'plate well {well!r} is not a well ID like A1 or P24')
        letters, col = m.groups()
        row = (ord(letters[0]) - 65 if len(letters) == 1
               else 26 + ord(letters[1]) - 65)
        cells[(row, int(col) - 1)] = (str(well).strip().upper(), str(name))

    need_r = max(r for r, _ in cells) + 1
    need_c = max(c for _, c in cells) + 1
    n_rows, n_cols = next(
        ((r, c) for r, c in _PLATE_SIZES if need_r <= r and need_c <= c),
        (need_r, need_c),
    )

    printed = ({str(w).strip().upper() for w in printed} if printed is not None
               else {w for w, n in cells.values() if n in idx_of})

    # Drawn in the charts' own coordinates: the frame is their plotting box --
    # left edge on the y-axis line, right edge at the plot's end, top at the
    # plot top, bottom on the x-axis line -- so in the grid the plate shares
    # every edge with the charts beside and above it.
    x0, x1 = float(_CHART_L), float(_CHART_W - _CHART_R)
    y0, y1 = float(_PLOT_TOP), float(_AXIS_Y)
    inset = 1.5
    pitch = min((x1 - x0 - 2 * inset) / n_cols, (y1 - y0 - 2 * inset) / n_rows)
    grid_w, grid_h = pitch * n_cols, pitch * n_rows
    gx = x0 + inset + (x1 - x0 - 2 * inset - grid_w) / 2
    gy = y0 + inset + (y1 - y0 - 2 * inset - grid_h) / 2

    radius = pitch * 0.41
    wells_js, names_js, labels_js, marks = [], [], [], []
    hand_js, hand_rings = [], []
    n_printed = n_held = 0
    for r in range(n_rows):
        for c in range(n_cols):
            cx, cy = round(c * pitch + pitch / 2, 2), round(r * pitch + pitch / 2, 2)
            label = f'{_row_name(r)}{c + 1}'
            well, name = cells.get((r, c), (label, ''))
            vi = idx_of.get(name, -1) if (name and well in printed) else -1
            wells_js.append(vi)
            names_js.append(html_escape(name))
            labels_js.append(label)
            hand_js.append(html_escape(manual.get(label, '')))
            if label in manual:
                hand_rings.append(f'<circle cx="{cx:g}" cy="{cy:g}" r="{radius + 1.4:.2f}"/>')
            if vi >= 0:
                n_printed += 1
                marks.append(f'<circle cx="{cx:g}" cy="{cy:g}" r="{radius:.2f}" '
                             f'fill="{variant_color[name]}"/>')
            elif name:
                n_held += 1
                marks.append(f'<circle class="unused" cx="{cx:g}" cy="{cy:g}" '
                             f'r="{radius - 0.4:.2f}"/>')
            else:
                marks.append(f'<circle class="none" cx="{cx:g}" cy="{cy:g}" '
                             f'r="{radius * 0.45:.2f}"/>')

    # Row letters where the charts put y tick labels; column numbers above the
    # frame. At this pitch a two-digit number is wider than a column, so the
    # odd columns are numbered, as on a narrow plate diagram.
    col_labs = ''.join(
        f'<text class="plab" x="{gx + c * pitch + pitch / 2:.2f}" y="{y0 - 6:g}" '
        f'text-anchor="middle">{c + 1}</text>'
        for c in range(0, n_cols, 2)
    )
    row_labs = ''.join(
        f'<text class="plab" x="{x0 - 7:g}" y="{gy + r * pitch + pitch / 2 + 3.3:.2f}" '
        f'text-anchor="end">{_row_name(r)}</text>'
        for r in range(n_rows)
    )
    svg = (
        f'<svg id="plate" class="chart" viewBox="0 0 {_CHART_W} {_CHART_H}" '
        f'role="img" aria-label="Source plate, {n_rows} by {n_cols} wells">'
        f'{col_labs}{row_labs}'
        f'<rect class="pframe" x="{x0:g}" y="{y0:g}" width="{x1 - x0:g}" '
        f'height="{y1 - y0:g}" rx="3"/>'
        f'<g transform="translate({gx:.2f},{gy:.2f})">'
        f'<g class="wells">{"".join(marks)}</g>'
        f'<g id="plate-hl"></g>'
        f'<g class="hand">{"".join(hand_rings)}</g>'
        f'</g>'
        f'<text class="axtitle" x="{(x0 + x1) / 2:g}" y="{_AXIS_Y + 24}" '
        f'text-anchor="middle">source plate · {n_printed:,} of '
        f'{n_rows * n_cols:,} wells printed</text>'
        f'</svg>'
    )
    # The key sits under the plate's axis title. Filled wells need no key --
    # they are drawn in the legend's colours -- so it only explains the rings.
    # Each item is one unwrappable span: a narrow cell breaks between items,
    # never inside one.
    key = []
    if n_held:
        key.append(f'<span class="ci"><span class="ring"></span>{n_held:,} not '
                   f'printed here</span>')
    if manual:
        listed = ', '.join(f'{w} ({html_escape(v)})' for w, v in sorted(
            manual.items(), key=lambda kv: (len(kv[0].rstrip('0123456789')), kv[0].rstrip('0123456789'),
                                            int(kv[0][len(kv[0].rstrip('0123456789')):] or 0))))
        key.append(f'<span class="ci hand"><span class="hring"></span>load by hand: '
                   f'{listed}</span>')
    html = (
        f'<section class="plate" aria-label="Source plate">{svg}'
        + (f'<p class="plate-key">{"".join(key)}</p>' if key else '')
        + '</section>'
    )
    data = {
        'C': n_cols, 'R': n_rows, 'U': round(pitch, 4), 'r': round(radius, 3),
        'PL': round(gx, 3), 'PT': round(gy, 3), 'VW': _CHART_W, 'VH': _CHART_H,
        'W': wells_js, 'N': names_js, 'L': labels_js, 'M': hand_js,
    }
    return html, data


def generate_dashboard(print_array, metrics, project_name, export_dir,
                       categories=None, n_blocks=None, refresh=0,
                       category_groups=None, device=None,
                       category_families=None, plate=None, plate_printed=None,
                       plate_manual=None):
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
            given, each category is drawn in its own colour. Without it, spots
            are colored by replicate count.
        n_blocks: Optional number of blocks the device is divided into. Blocks
            partition the array by column, and are labelled above the map, each
            under a rule spanning its own columns.
        device: Optional short name for the device, shown in the header (for
            example "MITOMI PS1.8k").
        category_groups: Optional dict mapping a category label to a second
            designation, used to sub-head the sublibraries in the legend -- two
            kinds of scan, say. Categories left out fall under a final group.
        category_families: Optional dict pinning a category to a palette index, so
            it keeps its colour across prints with different category sets.
        plate: Optional dict mapping source-plate well ID ("A1") to the name
            printed from it, drawn as a plate map beside the charts. Hovering a
            well lights its chambers on the array and the reverse. The plate
            size is the smallest standard format holding every well given.
        plate_printed: Optional set of well IDs actually dispensed on this chip.
            Defaults to every well whose name appears on the array; pass it when
            two wells share a name but only one is used (a WT control, say).
        plate_manual: Optional dict of wells loaded by hand at setup, rather
            than carried over from the library plate, to what goes in them
            ({"P23": "WT"}). Flagged in the header and ringed on the plate map.
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
    n_at_min = int((counts == min_count).sum())
    has_spread = (max_count - min_count) > 1
    # When most of the library sits on the floor -- which is what happens
    # whenever the library is large relative to the array -- that floor is the
    # design, not a defect, and colouring it as a fault buries the real signal.
    flag_low = has_spread and n_at_min <= max(1, round(0.25 * len(count_of)))

    rows_all_blank = np.all(print_array == '', axis=1)
    n_skipped = int(rows_all_blank.sum())
    n_empty = int((print_array == '').sum())

    # ── Assign a color to every variant ───────────────────────────────────────
    (cat_of, variant_color, cat_colour, shades_of,
     ordered_cats, size_of) = assign_variant_colours(
         count_of, categories, category_families)

    legend_blocks = []
    cat_mark = {}
    cat_index = {c: i for i, c in enumerate(ordered_cats)}

    if cat_of:
        for i, cat in enumerate(ordered_cats):
            mark = f'<span class="swatch" style="background:{cat_colour[cat]}"></span>'
            cat_mark[cat] = (
                f'<span class="key" data-cat="{i}">{mark}'
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
                color = _EMPTY
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
    pad_l, pad_r, pad_b = 47.0, 6.0, 30.0
    pad_t = 30.0 if blocked else 6.0
    grid_w, grid_h = n_cols * _U, n_rows * _U
    view_w, view_h = pad_l + grid_w + pad_r, pad_t + grid_h + pad_b

    x_step = 1 if n_cols <= 12 else (5 if n_cols <= 60 else 10)
    y_step = 1 if n_rows <= 12 else (5 if n_rows <= 60 else 10)

    col_marks = ''.join(
        f'<line class="tick" x1="{pad_l + c * _U + _U / 2:g}" y1="{pad_t + grid_h + 1.5:g}" '
        f'x2="{pad_l + c * _U + _U / 2:g}" y2="{pad_t + grid_h + 6:g}"/>'
        f'<text class="axlab" x="{pad_l + c * _U + _U / 2:g}" '
        f'y="{pad_t + grid_h + 23:g}" text-anchor="middle">{c + 1}</text>'
        for c in range(0, n_cols, x_step)
    )
    row_marks = ''.join(
        f'<line class="tick" x1="{pad_l - 6:g}" y1="{pad_t + r * _U + _U / 2:g}" '
        f'x2="{pad_l - 1.5:g}" y2="{pad_t + r * _U + _U / 2:g}"/>'
        f'<text class="axlab" x="{pad_l - 13:g}" y="{pad_t + r * _U + _U / 2 + 4.2:g}" '
        f'text-anchor="end">{row_labels[r]}</text>'
        for r in range(0, n_rows, y_step)
    )

    block_marks = ''
    block_seps = ''
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
        block_seps = ''.join(
            f'<line class="blk-sep" vector-effect="non-scaling-stroke" '
            f'x1="{k * span * _U:g}" y1="0" '
            f'x2="{k * span * _U:g}" y2="{grid_h:g}"/>'
            for k in range(1, n_blocks)
        )

    svg = (
        f'<svg id="chip" viewBox="0 0 {view_w:g} {view_h:g}" '
        f'role="img" aria-label="Layout of the {n_rows} by {n_cols} print array">'
        f'<g transform="translate({pad_l:g},{pad_t:g})">'
        f'<g class="cells">{cells}</g>'
        f'<path id="chip-hl" d=""/>'
        f'<g id="chip-cat"></g>'
        f'<path id="chip-mask" d="" fill="{_SURFACE}"/>'
        f'<g class="seps">{block_seps}</g>'
        f'<rect class="spine" x="0" y="0" width="{grid_w:g}" height="{grid_h:g}" '
        f'vector-effect="non-scaling-stroke"/>'
        f'</g>'
        f'{row_marks}{col_marks}{block_marks}'
        f'</svg>'
    )

    plate_html, plate_data = _plate_map(plate, plate_printed, idx_of, variant_color,
                                        plate_manual)

    # ── Hover data ────────────────────────────────────────────────────────────
    cat_index = cat_index if cat_of else {}
    variants_js = [None] * len(idx_of)
    for v, i in idx_of.items():
        variants_js[i] = {
            'c': variant_color[v],
            'n': html_escape(v),
            'g': html_escape(cat_of[v]) if cat_of else '',
            'k': cat_index[cat_of[v]] if cat_of else -1,
        }
    fit_js = (_FIT_JS.replace('__RATIO__', f'{view_w / view_h:.6f}')
              .replace('__FITVW__', f'{view_w:g}')
              .replace('__FITPT__', f'{pad_t:g}'))
    # Omitted entirely when off, rather than shipped inert.
    refresh_js = (_REFRESH_JS.replace('__SECS__', str(int(refresh)))
                  if refresh else '')
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
        .replace('__PLATE__', json.dumps(plate_data, separators=(',', ':')))
    )

    # ── Copy ──────────────────────────────────────────────────────────────────

    # Kept to what a reader needs at a glance: the device and how much of it is
    # in play. Replication lives in the charts, empties in the legend.
    used_blocks = None
    if blocked:
        span = n_cols // n_blocks
        used_blocks = sum(
            1 for k in range(n_blocks)
            if (print_array[:, k * span:(k + 1) * span] != '').any()
        )

    figures = [(f'{n_rows} &#215; {n_cols}', 'rows &#215; columns')]
    if device:
        figures.append((
            html_escape(str(device)),
            f'{n_blocks}-block device' if blocked else 'device',
        ))
    figures.append((f'{len(count_of):,}', 'variants'))
    if used_blocks is not None:
        figures.append((f'{used_blocks} of {n_blocks}', 'blocks used'))
    if plate_manual:
        # Flagged: these wells are not on the library plate and have to be
        # loaded by hand at setup.
        hand_wells = sorted(str(w).strip().upper() for w in plate_manual)
        hand_what = sorted({str(v) for v in plate_manual.values()})
        figures.append((
            html_escape(', '.join(hand_wells)),
            f'load {html_escape(", ".join(hand_what))} by hand',
            'flag',
        ))

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
        ordered = sorted(buckets.items(), key=lambda kv: (min(kv[1]), _natural_key(kv[0])))
        if len(ordered) > 8:
            rest = [c for _, vals in ordered[8:] for c in vals]
            ordered = ordered[:8] + [('Other', rest)]
        group_chart = (
            _group_ranges(ordered, min_count, group_heading, hist_x, flag_low,
                            empty=n_empty)
        )

    if cat_of:
        subs = [c for c in ordered_cats if size_of[c] > 1]
        singles = [c for c in ordered_cats if size_of[c] == 1]
        groups = category_groups or {}

        def kind_of(cat):
            return str(groups.get(cat, '')) or 'Other'

        kinds = []
        for c in subs:
            if kind_of(c) not in kinds:
                kinds.append(kind_of(c))
        subs = [c for c in legend_order(ordered_cats, size_of, groups)
                if size_of[c] > 1]
        split = bool(subs and singles)          # two kinds of entry to tell apart
        nest = bool(groups) and len(kinds) > 1  # more than one kind of sublibrary

        def section(title, cats, headed):
            if not cats:
                return
            if headed:
                legend_blocks.append(f'<span class="key-head">{html_escape(title)}</span>')
            for c in cats:
                legend_blocks.append(cat_mark[c])

        if nest:
            if split:
                legend_blocks.append('<span class="key-head">Sublibraries</span>')
            for g in kinds:
                legend_blocks.append(f'<span class="key-sub">{html_escape(g)}</span>')
                for c in subs:
                    if kind_of(c) == g:
                        legend_blocks.append(
                            cat_mark[c].replace('class="key"', 'class="key nested"'))
        else:
            section('Sublibraries', subs, split)

        section('Single samples', singles, split)

    # The map's two pale shades had no key at all: one is a chamber left
    # unfilled, the other a whole row the device reserves. Both are empty, and
    # the distinction is worth stating.
    if n_empty:
        spaced = ' spaced' if legend_blocks else ''
        legend_blocks.append(
            f'<span class="key{spaced}" data-blank="-1">'
            f'<span class="swatch" style="background:{_EMPTY}"></span>'
            f'<span class="key-name">Empty chamber{"" if n_empty == 1 else "s"}'
            f'<span class="key-n">{n_empty:,}</span></span>'
            '</span>'
        )

    # What the library was designed to hold, against what reached the array.
    # A designed member that never got printed is the point of this chart.
    composition = ''
    if categories:
        designed = {}
        for name, raw in categories.items():
            label = str(raw) if raw not in (None, '') else 'Uncategorized'
            designed.setdefault(label, []).append(str(name))
        known = [c for c in ordered_cats if c in designed]
        extra = sorted((c for c in designed if c not in ordered_cats), key=_natural_key)
        rows = []
        for label in known + extra:
            members = designed[label]
            absent = sum(1 for m in members if m not in count_of)
            rows.append((label, len(members), absent,
                         cat_colour.get(label, '#9fb0ad')))
        if rows:
            composition = _composition(rows)

    regions_html = _region_table(_region_stats(print_array, n_blocks), n_blocks)
    regions_block = (
        '<input type="checkbox" id="show-regions" class="rtoggle">'
        '<label class="rmore" for="show-regions">'
        '<span class="open">Statistics by block and half</span>'
        '<span class="shut">Hide block statistics</span>'
        '</label>'
        f'<div class="rwrap">{regions_html}</div>'
    ) if regions_html else ''

    legend_html = ''.join(legend_blocks)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html_escape(project_name)} print check</title>
  <style>{_DASHBOARD_CSS}</style>
</head>
<body style="--chip-ratio:{view_w / view_h:.6f};--chart-ratio:{_CHART_W} / {_CHART_H}">
  <div class="sheet">
    <div class="page">
      <div class="col-main">
        <header>
          <h1>{html_escape(project_name)}</h1>
          <p class="byline">{byline}</p>
          <ul class="figures">{figures_html}</ul>
        </header>

        <section class="counts">
        {hist}
        {group_chart}
        {composition}
        {plate_html}
        {regions_block}
          <div class="chart-tip" id="chart-tip" hidden></div>
        </section>
      </div>

      <section class="map" aria-label="Array layout">
        <div class="chip"><div class="plot">{svg}</div></div>
        <div class="side">
          <div class="map-key">{legend_html}</div>
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