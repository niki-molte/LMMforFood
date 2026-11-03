"""
dot_plot.py
===========
GENERIC generator of the "few-shot dot-plot" chart: it is the function reused
by all the charts of the package (NLP few-shot, Macro few-shot, comparison of
Optuna strategies, YOLO/YOLO-Optuna comparison). It receives only the data (via
a callback) and the style from style.py: no plotting logic is
duplicated across the various charts.
"""
import os
import matplotlib.pyplot as plt

from . import style as st


def plot_dot_grid(
    fields,
    value_fn,
    strategies,
    colors,
    markers,
    out_path_combined,
    out_path_single_template,
    title_fn,
    ylabel_fn,
    baseline_fn=None,
    log_scale=False,
    connect_lines=False,
    linestyles=None,
    extra_legend_handles=None,
    legend_strategies=None,
    y_lim_fn=None,
    y_ticks_fn=None,
    grid_shape=(2, 3),
    figsize_combined=(18, 12),
    figsize_single=(10, 8),
    hide_axes=(5,),
    legend_title="Strategies (Left=1 ex., Center=5 ex., Right=10 ex.)",
    suptitle=None,
    legend_ncol=None,
):
    """
    Draw a grid of dot-plots (one per 'field') plus the individual panels.

    value_fn(group, example, strategy, field) -> float | None
        Returns the value to plot, or None/0.0 if missing.
    baseline_fn(field) -> float | None
        If provided, draws the dashed baseline line (MAE% style).
    linestyles: dict strategy -> linestyle (es. {'Standard': '-', 'Standard Optuna': '--'}).
        If provided, always draws the connection lines with the given style.
    extra_legend_handles: list of matplotlib handles to prepend to the marker legend.
    title_fn(field) -> str
    ylabel_fn(field) -> str
    """

    def draw_panel(ax, field):
        base_val = baseline_fn(field) if baseline_fn else None
        if base_val is not None:
            st.draw_baseline(ax, base_val, log_scale=log_scale)

        for grp_idx, g in enumerate(st.GROUPS):
            if grp_idx > 0:
                ax.axvline(x=grp_idx - 0.5, color='lightgray', linestyle=':', linewidth=1.5, zorder=0)

            # Connection lines: with linestyles use a per-strategy style; otherwise
            # use the default behavior (all solid) if connect_lines=True.
            if linestyles is not None or connect_lines:
                for strat in strategies:
                    ls = linestyles.get(strat, '-') if linestyles else '-'
                    line_x, line_y = [], []
                    for ex in st.EXAMPLES:
                        val = value_fn(g, ex, strat, field)
                        if val:
                            line_x.append(grp_idx + st.EX_OFFSETS[ex])
                            line_y.append(val)
                    if len(line_x) >= 2:
                        ax.plot(line_x, line_y, color=colors[strat], linestyle=ls,
                                linewidth=1.8, alpha=0.45, zorder=2, solid_capstyle='round')

            for ex in st.EXAMPLES:
                x_pos = grp_idx + st.EX_OFFSETS[ex]
                for strat in strategies:
                    val = value_fn(g, ex, strat, field)
                    if val is None:
                        continue
                    plot_val = max(val, 0.01) if log_scale else val
                    if plot_val <= 0.01 and log_scale and val == 0:
                        continue
                    is_below = base_val is not None and val < base_val
                    edgecolor = 'black' if is_below else 'white'
                    lw = 1.5 if is_below else 0.6
                    scale = st.MARKER_SIZE_SCALE.get(markers[strat], 1.0)
                    size = (110 if is_below else 80) * scale
                    ax.scatter(x_pos, plot_val, color=colors[strat], marker=markers[strat],
                                s=size, edgecolors=edgecolor, linewidth=lw, zorder=4)

        y_lim = y_lim_fn(field) if y_lim_fn else None
        y_ticks = y_ticks_fn(field) if y_ticks_fn else None
        st.style_dot_axis(ax, ylabel_fn(field), title_fn(field), log_scale=log_scale,
                           y_lim=y_lim, y_ticks=y_ticks)

    _legend_strats = legend_strategies if legend_strategies is not None else strategies
    strategy_handles = st.strategy_legend_handles(_legend_strats, colors, markers)
    all_legend_handles = (list(extra_legend_handles) if extra_legend_handles else []) + strategy_handles

    # ---- combined panel ----
    rows, cols = grid_shape
    fig, axes = plt.subplots(rows, cols, figsize=figsize_combined)
    import numpy as _np
    axes = _np.atleast_1d(axes).flatten()
    for idx, field in enumerate(fields):
        draw_panel(axes[idx], field)
    for h in hide_axes:
        if h < len(axes):
            axes[h].axis('off')

    st.add_bottom_legend(fig, all_legend_handles, legend_title, ncol=legend_ncol)
    if suptitle:
        plt.suptitle(suptitle, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    os.makedirs(os.path.dirname(out_path_combined), exist_ok=True)
    plt.savefig(out_path_combined, bbox_inches='tight', dpi=300)
    plt.close()

    # ---- single panels ----
    for field in fields:
        fig_s, ax_s = plt.subplots(figsize=figsize_single)
        draw_panel(ax_s, field)
        ax_s.legend(handles=all_legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    ncol=min(len(all_legend_handles), 4), fontsize=11, frameon=False,
                    title=legend_title, title_fontsize=12)
        plt.tight_layout(rect=[0, 0.15, 1, 1])
        out_path = out_path_single_template.format(field=field)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close()