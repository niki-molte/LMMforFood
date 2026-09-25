"""
dot_plot.py
===========
Generatore GENERICO del grafico "dot-plot few-shot": è la funzione riusata
da tutti i grafici del pacchetto (NLP few-shot, Macro few-shot, confronto
strategie Optuna, confronto YOLO/YOLO-Optuna). Riceve solo i dati (tramite
una callback) e lo stile da style.py: nessuna logica di plotting viene
duplicata tra i vari grafici.
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
    y_lim_fn=None,
    y_ticks_fn=None,
    grid_shape=(2, 3),
    figsize_combined=(18, 12),
    figsize_single=(10, 8),
    hide_axes=(5,),
    legend_title="Strategie (Sinistra=1 es., Centro=5 es., Destra=10 es.)",
    suptitle=None,
    legend_ncol=None,
):
    """
    Disegna una griglia di dot-plot (uno per 'field') + i singoli pannelli.

    value_fn(group, example, strategy, field) -> float | None
        Restituisce il valore da plottare, o None/0.0 se assente.
    baseline_fn(field) -> float | None
        Se fornito, disegna la linea di baseline tratteggiata (stile MAE%).
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

            if connect_lines:
                for strat in strategies:
                    line_x, line_y = [], []
                    for ex in st.EXAMPLES:
                        val = value_fn(g, ex, strat, field)
                        if val:
                            line_x.append(grp_idx + st.EX_OFFSETS[ex])
                            line_y.append(val)
                    if len(line_x) >= 2:
                        ax.plot(line_x, line_y, color=colors[strat], linewidth=1.8,
                                 alpha=0.45, zorder=2, solid_capstyle='round')

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
                    size = 110 if is_below else 80
                    ax.scatter(x_pos, plot_val, color=colors[strat], marker=markers[strat],
                                s=size, edgecolors=edgecolor, linewidth=lw, zorder=4)

        y_lim = y_lim_fn(field) if y_lim_fn else None
        y_ticks = y_ticks_fn(field) if y_ticks_fn else None
        st.style_dot_axis(ax, ylabel_fn(field), title_fn(field), log_scale=log_scale,
                           y_lim=y_lim, y_ticks=y_ticks)

    legend_handles = st.strategy_legend_handles(strategies, colors, markers)

    # ---- pannello combinato ----
    rows, cols = grid_shape
    fig, axes = plt.subplots(rows, cols, figsize=figsize_combined)
    import numpy as _np
    axes = _np.atleast_1d(axes).flatten()
    for idx, field in enumerate(fields):
        draw_panel(axes[idx], field)
    for h in hide_axes:
        if h < len(axes):
            axes[h].axis('off')

    st.add_bottom_legend(fig, legend_handles, legend_title, ncol=legend_ncol)
    if suptitle:
        plt.suptitle(suptitle, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    os.makedirs(os.path.dirname(out_path_combined), exist_ok=True)
    plt.savefig(out_path_combined, bbox_inches='tight', dpi=300)
    plt.close()

    # ---- pannelli singoli ----
    for field in fields:
        fig_s, ax_s = plt.subplots(figsize=figsize_single)
        draw_panel(ax_s, field)
        ax_s.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    ncol=min(len(legend_handles), 4), fontsize=11, frameon=False,
                    title=legend_title, title_fontsize=12)
        plt.tight_layout(rect=[0, 0.15, 1, 1])
        out_path = out_path_single_template.format(field=field)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close()