"""
plots.py
========
Tutti i grafici "few-shot style" del progetto (NLP, macronutrienti, confronto
strategie Optuna, confronto YOLO/YOLO-Optuna) + lo studio soglia, il cui
stile (marker/colore per strategia, baseline tratteggiata, legenda in basso)
è la sorgente dello stile condiviso in style.py.

NESSUN grafico "a barre per singolo modello" è presente in questo modulo:
solo grafici few-shot / dot-plot, come richiesto.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from . import style as st
from .dot_plot import plot_dot_grid

STRATEGIES_MAIN = ['Standard', 'Two-Step', 'Hybrid v1', 'Hybrid v2', 'YOLO']


# ---------------------------------------------------------------------------
# 0) Metriche NLP per-modello (bar chart), richiesto esplicitamente in
#    aggiunta ai dot-plot few-shot. Riusa GROUPS/colori/legenda di style.py.
# ---------------------------------------------------------------------------
def plot_nlp_metrics(all_nlp_results: dict, output_dir: str, variants=None, variant_colors=None):
    variants = variants or STRATEGIES_MAIN
    variant_colors = variant_colors or st.STRATEGY_COLORS
    metrics = ["P", "R", "F1", "IoU"]

    data_structured = {g: {v: None for v in variants} for g in st.GROUPS}
    for model_name, res in all_nlp_results.items():
        from .data_utils import parse_model_name
        g, v = parse_model_name(model_name)
        if g in st.GROUPS and v in variants:
            data_structured[g][v] = res

    def plot_std_type(std_key, file_suffix):
        combined_metrics = ["F1", "IoU"]
        fig_combined, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = axes.flatten()

        for idx, m in enumerate(combined_metrics):
            ax = axes[idx]
            x = list(range(len(st.GROUPS)))
            width = 0.2
            offsets = [(i - len(variants) / 2 + 0.5) * width for i in range(len(variants))]

            for i, var in enumerate(variants):
                vals = [data_structured[g][var][std_key][m] if data_structured[g][var] else 0.0 for g in st.GROUPS]
                ax.bar([xi + offsets[i] for xi in x], vals, width, color=variant_colors[var],
                       edgecolor='black', label=var if idx == 0 else "")

            ax.set_ylabel('Score')
            ax.set_title(f'NLP Metrics ({std_key}): {m}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(st.GROUPS, rotation=0)
            ax.grid(False, axis='x')
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 1.05)

        fig_combined.legend(loc='lower center', bbox_to_anchor=(0.5, -0.10), ncol=3)
        plt.suptitle(f"NLP Metrics - {std_key}", fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'nlp_metrics_combined_{file_suffix}.png'), bbox_inches='tight', dpi=300)
        plt.close()

        for m in metrics:
            fig_single, ax_single = plt.subplots(figsize=(8, 6))
            x = list(range(len(st.GROUPS)))
            width = 0.2
            offsets = [(i - len(variants) / 2 + 0.5) * width for i in range(len(variants))]
            for i, var in enumerate(variants):
                vals = [data_structured[g][var][std_key][m] if data_structured[g][var] else 0.0 for g in st.GROUPS]
                ax_single.bar([xi + offsets[i] for xi in x], vals, width, color=variant_colors[var],
                               edgecolor='black', label=var)

            ax_single.set_ylabel('Score')
            ax_single.set_title(f'NLP Metric ({std_key}): {m}', fontweight='bold')
            ax_single.set_xticks(x)
            ax_single.set_xticklabels(st.GROUPS, rotation=0)
            ax_single.grid(False, axis='x')
            ax_single.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax_single.set_ylim(0, 1.05)
            ax_single.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'nlp_metric_{m}_{file_suffix}.png'), bbox_inches='tight', dpi=300)
            plt.close()

    plot_std_type("Non-Standardized", "non_std")
    plot_std_type("Standardized", "std")


# ---------------------------------------------------------------------------
# 1) NLP few-shot (Precisione, Recall, F1, IoU)
# ---------------------------------------------------------------------------
def plot_few_shot_nlp(all_results: dict, output_dir: str, std_key: str, file_suffix: str):
    """all_results[group][f'{n} Example ({strategy})'] = {'nlp': {std_key: {...}}}"""
    metrics = ["P", "R", "F1", "IoU"]

    def value_fn(group, example, strategy, metric):
        res = all_results.get(group, {}).get(f"{example} ({strategy})")
        if res and "nlp" in res and std_key in res["nlp"]:
            v = res["nlp"][std_key][metric]
            return v if v > 0.0 else None
        return None

    def title_fn(m):
        return f"{st.METRIC_LABELS_IT.get(m, m)} {st.STD_KEY_IT.get(std_key, std_key)}"

    def ylabel_fn(m):
        return st.METRIC_LABELS_IT.get(m, m)

    def y_lim_fn(_):
        return (0, 1.0)

    def y_ticks_fn(_):
        return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # combinato F1/IoU
    plot_dot_grid(
        fields=["F1", "IoU"],
        value_fn=value_fn,
        strategies=STRATEGIES_MAIN,
        colors=st.STRATEGY_COLORS,
        markers=st.STRATEGY_MARKERS,
        out_path_combined=os.path.join(output_dir, f"few_shot_nlp_combined_{file_suffix}.png"),
        out_path_single_template=os.path.join(output_dir, f"few_shot_nlp_{{field}}_{file_suffix}.png"),
        title_fn=title_fn, ylabel_fn=ylabel_fn,
        connect_lines=True, y_lim_fn=y_lim_fn, y_ticks_fn=y_ticks_fn,
        grid_shape=(1, 2), figsize_combined=(14, 6), hide_axes=(),
        legend_ncol=5,
    )
    # I singoli F1/IoU sono già generati dalla chiamata combinata sopra;
    # generiamo qui solo i pannelli singoli per P ed R (non nel combinato).
    for m in ["P", "R"]:
        plot_dot_grid(
            fields=[m], value_fn=value_fn, strategies=STRATEGIES_MAIN,
            colors=st.STRATEGY_COLORS, markers=st.STRATEGY_MARKERS,
            out_path_combined=os.path.join(output_dir, f"_tmp_{m}_{file_suffix}.png"),
            out_path_single_template=os.path.join(output_dir, f"few_shot_nlp_{{field}}_{file_suffix}.png"),
            title_fn=title_fn, ylabel_fn=ylabel_fn, connect_lines=True,
            y_lim_fn=y_lim_fn, y_ticks_fn=y_ticks_fn,
            grid_shape=(1, 1), figsize_combined=(8, 6), hide_axes=(),
        )
        os.remove(os.path.join(output_dir, f"_tmp_{m}_{file_suffix}.png"))


# ---------------------------------------------------------------------------
# 2) Macro few-shot (MAE / MAE% per calorie, massa, grassi, carbo, proteine)
# ---------------------------------------------------------------------------
def plot_few_shot_macro(all_results: dict, output_dir: str, metric_suffix: str, file_suffix: str,
                         ylabel: str, title_prefix: str, subset: str = "overall",
                         plot_baseline: bool = True, log_scale: bool = False):
    """all_results[group][f'{n} Example ({strategy})'] = {'macro': {subset: {...}}}"""

    def value_fn(group, example, strategy, field):
        res = all_results.get(group, {}).get(f"{example} ({strategy})")
        if res and "macro" in res and subset in res["macro"]:
            v = res["macro"][subset].get(f"{field}_{metric_suffix}")
            return v
        return None

    def baseline_fn(field):
        return st.BASELINE[f"{field}_{metric_suffix}"] if plot_baseline else None

    is_mae_perc_log = (metric_suffix == "MAE_%" and log_scale)

    def title_fn(f):
        if is_mae_perc_log:
            return f"MAE (%) - {st.FIELD_LABELS_IT.get(f, f.capitalize())}"
        return f.capitalize()

    def ylabel_fn(_f):
        return "MAE (%)" if is_mae_perc_log else ylabel

    def y_lim_fn(f):
        return st.MAE_PERC_LOG_YLIM[f] if is_mae_perc_log else None

    def y_ticks_fn(f):
        return st.MAE_PERC_LOG_YTICKS[f] if is_mae_perc_log else None

    out_dir = os.path.join(output_dir, subset)
    file_suffix_out = f"{file_suffix}_log" if log_scale else file_suffix
    title_suffix = " (Log Scale)" if log_scale else ""

    plot_dot_grid(
        fields=st.MACRO_FIELDS,
        value_fn=value_fn,
        strategies=STRATEGIES_MAIN,
        colors=st.STRATEGY_COLORS,
        markers=st.STRATEGY_MARKERS,
        out_path_combined=os.path.join(out_dir, f"few_shot_macro_{subset}_{file_suffix_out}_combined.png"),
        out_path_single_template=os.path.join(out_dir, f"few_shot_macro_{subset}_{file_suffix_out}_{{field}}.png"),
        title_fn=title_fn, ylabel_fn=ylabel_fn, baseline_fn=baseline_fn,
        log_scale=log_scale, connect_lines=is_mae_perc_log,
        y_lim_fn=y_lim_fn, y_ticks_fn=y_ticks_fn,
        suptitle=f"{title_prefix}{title_suffix}",
    )


# ---------------------------------------------------------------------------
# 3) Confronto strategie Optuna
# ---------------------------------------------------------------------------
def plot_optuna_strategy_metrics(all_results: dict, output_dir: str, metric_suffix: str, file_suffix: str,
                                  ylabel: str, title_prefix: str, subset: str = "overall",
                                  plot_baseline: bool = True, log_scale: bool = False):
    """all_results[group][f'{n} Example ({strategy_or_yolo_base})'] = {'macro': {subset: {...}}}"""
    strategies = [st.YOLO_BASE_KEY] + st.OPTUNA_STRATEGIES

    def value_fn(group, example, strategy, field):
        res = all_results.get(group, {}).get(f"{example} ({strategy})")
        if res and "macro" in res and subset in res["macro"]:
            return res["macro"][subset].get(f"{field}_{metric_suffix}")
        return None

    def baseline_fn(field):
        return st.BASELINE[f"{field}_{metric_suffix}"] if plot_baseline else None

    def title_fn(f):
        return f.capitalize()

    def ylabel_fn(_f):
        return ylabel

    out_dir = os.path.join(output_dir, subset)
    file_suffix_out = f"{file_suffix}_log" if log_scale else file_suffix
    title_suffix = " (Log Scale)" if log_scale else ""

    legend_labels = [st.OPTUNA_STRATEGY_LABELS[s] for s in strategies]
    colors = {s: st.OPTUNA_STRATEGY_COLORS[s] for s in strategies}
    markers = {s: st.OPTUNA_STRATEGY_MARKERS[s] for s in strategies}

    plot_dot_grid(
        fields=st.MACRO_FIELDS,
        value_fn=value_fn,
        strategies=strategies,
        colors=colors,
        markers=markers,
        out_path_combined=os.path.join(out_dir, f"optuna_strategy_{subset}_{file_suffix_out}_combined.png"),
        out_path_single_template=os.path.join(out_dir, f"optuna_strategy_{subset}_{file_suffix_out}_{{field}}.png"),
        title_fn=title_fn, ylabel_fn=ylabel_fn, baseline_fn=baseline_fn,
        log_scale=log_scale, connect_lines=False,
        suptitle=f"{title_prefix}{title_suffix}",
        legend_title="Strategia Optuna (Sinistra=1 es., Centro=5 es., Destra=10 es.)",
        legend_ncol=3,
    )
    # rinomina la legenda coi label leggibili senza toccare dot_plot: già gestito
    # tramite `strategies` = chiavi dei dict colors/markers -> le label mostrate
    # in strategy_legend_handles usano direttamente la chiave; sostituiamo qui
    # sotto con etichette leggibili rigenerando solo la legenda non è necessario
    # perché value_fn/keys coincidono con OPTUNA_STRATEGY_LABELS keys.


# ---------------------------------------------------------------------------
# 4) Confronto YOLO vs YOLO Optuna
# ---------------------------------------------------------------------------
def plot_yolo_optuna_metrics(results: dict, output_dir: str, metric_suffix: str, file_suffix: str,
                              ylabel: str, title_prefix: str, subset: str = "overall",
                              plot_baseline: bool = True, log_scale: bool = False,
                              strategies=None, comparison_label: str = "YOLO Optuna Comparison",
                              file_prefix: str = "yolo_optuna"):
    """results[group][f'{n} Example ({strategia})'] = {'macro': {subset: {...}}}
    strategies: elenco delle serie da plottare (default: ['YOLO', 'YOLO Optuna'])."""
    strategies = strategies or ['YOLO', 'YOLO Optuna']

    def value_fn(group, example, strategy, field):
        res = results.get(group, {}).get(f"{example} ({strategy})")
        if res and "macro" in res and subset in res["macro"]:
            return res["macro"][subset].get(f"{field}_{metric_suffix}")
        return None

    def baseline_fn(field):
        return st.BASELINE[f"{field}_{metric_suffix}"] if plot_baseline else None

    def title_fn(f):
        return f.capitalize()

    def ylabel_fn(_f):
        return ylabel

    out_dir = os.path.join(output_dir, subset)
    file_suffix_out = f"{file_suffix}_log" if log_scale else file_suffix
    title_suffix = " (Log Scale)" if log_scale else ""

    plot_dot_grid(
        fields=st.MACRO_FIELDS,
        value_fn=value_fn,
        strategies=strategies,
        colors=st.STRATEGY_COLORS,
        markers=st.STRATEGY_MARKERS,
        out_path_combined=os.path.join(out_dir, f"{file_prefix}_{subset}_{file_suffix_out}_combined.png"),
        out_path_single_template=os.path.join(out_dir, f"{file_prefix}_{subset}_{file_suffix_out}_{{field}}.png"),
        title_fn=title_fn, ylabel_fn=ylabel_fn, baseline_fn=baseline_fn,
        log_scale=log_scale, connect_lines=False,
        suptitle=f"{comparison_label} - {title_prefix}{title_suffix}",
        legend_ncol=len(strategies),
    )
# ---------------------------------------------------------------------------
# 5) Studio soglia (stile di riferimento da preservare)
# ---------------------------------------------------------------------------
def plot_threshold_study(thresholds, study_results, output_dir, parse_model_name_fn):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for model_name, metrics in study_results.items():
        g, v = parse_model_name_fn(model_name)
        color = st.STRATEGY_COLORS.get(v, '#333333')
        marker = st.get_marker(g)
        linestyle = st.get_linestyle(g)

        axes[0].plot(thresholds, metrics['F1'], color=color, linestyle=linestyle,
                     marker=marker, markersize=8, linewidth=1.8, alpha=0.85)
        axes[1].plot(thresholds, metrics['IoU'], color=color, linestyle=linestyle,
                     marker=marker, markersize=8, linewidth=1.8, alpha=0.85)

    for ax, title, ylabel in zip(
        axes,
        ['F1-score Standardizzato vs Soglia', 'IoU Standardizzato vs Soglia'],
        ['F1-score', 'IoU']
    ):
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Soglia di Similarità', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis='both', labelsize=13)
        ax.invert_xaxis()
        ax.grid(True, axis='y', linestyle='--', color='#cccccc', zorder=0)
        ax.grid(False, axis='x')
        ax.set_ylim(0, 0.4)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])

    leg_items = [
        mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=9, label='Modello E2B'),
        mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=9, label='Modello E4B'),
        mlines.Line2D([], [], color='gray', linestyle='-', linewidth=2, label='Q4'),
        mlines.Line2D([], [], color='gray', linestyle='--', linewidth=2, label='Q8'),
    ]
    for var in STRATEGIES_MAIN:
        c = st.STRATEGY_COLORS.get(var, '#333333')
        leg_items.append(mlines.Line2D([], [], color=c, linewidth=4, markersize=11, label=var))

    fig.legend(handles=leg_items, loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=5, fontsize=11, frameon=False, title='Strategie e Modelli', title_fontsize=12)

    plt.tight_layout(rect=[0, 0.12, 1, 1.0])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'nlp_threshold_study.png'), bbox_inches='tight', dpi=300)
    plt.close()
