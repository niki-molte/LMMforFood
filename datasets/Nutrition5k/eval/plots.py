"""
plots.py
========
All the "few-shot style" charts of the project (NLP, macronutrients, comparison of
Optuna strategies, YOLO/YOLO-Optuna comparison) + the threshold study, whose
style (per-strategy marker/color, dashed baseline, legend at the bottom)
is the source of the shared style in style.py.

The only per-model bar chart in this module is the NLP metrics one (section 0);
all the others are few-shot / dot-plot charts.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from . import style as st
from .dot_plot import plot_dot_grid

STRATEGIES_MAIN = ['Standard', 'Two-Step', 'Hybrid v1', 'Hybrid v2', 'YOLO']


# ---------------------------------------------------------------------------
# 0) Per-model NLP metrics (bar chart), in
#    addition to the few-shot dot-plots. Reuses GROUPS/colors/legend from style.py.
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
# 1) NLP few-shot (Precision, Recall, F1, IoU)
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
    # The single F1/IoU panels are already generated by the combined call above;
    # only the single panels for P and R (not in the combined plot) are generated here.
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
# 2) Macro few-shot (MAE / MAE% for calories, mass, fat, carbs, protein)
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

    metric_label = "MAE (%)" if metric_suffix == "MAE_%" else "MAE (Value)"

    def title_fn(f):
        return f"{metric_label} - {st.FIELD_LABELS_IT.get(f, f.capitalize())}"

    def ylabel_fn(_f):
        return metric_label

    def y_lim_fn(f):
        if not log_scale:
            return None
        if metric_suffix == "MAE_%":
            return st.MAE_PERC_LOG_YLIM.get(f)
        return st.get_mae_abs_log_ylim(f, subset)

    def y_ticks_fn(f):
        if not log_scale:
            return None
        if metric_suffix == "MAE_%":
            return st.MAE_PERC_LOG_YTICKS.get(f)
        return st.get_mae_abs_log_yticks(f, subset)

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
        log_scale=log_scale, connect_lines=True,
        y_lim_fn=y_lim_fn, y_ticks_fn=y_ticks_fn,
        suptitle=f"{title_prefix}{title_suffix}",
    )


# ---------------------------------------------------------------------------
# 3) Optuna strategy comparison
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

    metric_label = "MAE (%)" if metric_suffix == "MAE_%" else "MAE (Value)"

    def title_fn(f):
        return f"{metric_label} - {st.FIELD_LABELS_IT.get(f, f.capitalize())}"

    def ylabel_fn(_f):
        return metric_label

    def y_lim_fn(f):
        if not log_scale:
            return None
        if metric_suffix == "MAE_%":
            return st.MAE_PERC_LOG_YLIM.get(f)
        return st.get_mae_abs_log_ylim(f, subset)

    def y_ticks_fn(f):
        if not log_scale:
            return None
        if metric_suffix == "MAE_%":
            return st.MAE_PERC_LOG_YTICKS.get(f)
        return st.get_mae_abs_log_yticks(f, subset)

    out_dir = os.path.join(output_dir, subset)
    file_suffix_out = f"{file_suffix}_log" if log_scale else file_suffix
    title_suffix = " (Log Scale)" if log_scale else ""

    colors = {s: st.OPTUNA_STRATEGY_COLORS[s] for s in strategies}
    markers = {s: st.OPTUNA_STRATEGY_MARKERS[s] for s in strategies}
    linestyles = {st.YOLO_BASE_KEY: '-', **{s: '--' for s in st.OPTUNA_STRATEGIES}}

    plot_dot_grid(
        fields=st.MACRO_FIELDS,
        value_fn=value_fn,
        strategies=strategies,
        colors=colors,
        markers=markers,
        out_path_combined=os.path.join(out_dir, f"optuna_strategy_{subset}_{file_suffix_out}_combined.png"),
        out_path_single_template=os.path.join(out_dir, f"optuna_strategy_{subset}_{file_suffix_out}_{{field}}.png"),
        title_fn=title_fn, ylabel_fn=ylabel_fn, baseline_fn=baseline_fn,
        log_scale=log_scale, connect_lines=True, linestyles=linestyles,
        y_lim_fn=y_lim_fn, y_ticks_fn=y_ticks_fn,
        suptitle=f"{title_prefix}{title_suffix}",
        legend_title="Optuna Strategy (Left=1 ex., Center=5 ex., Right=10 ex.)",
        legend_ncol=3,
    )


# ---------------------------------------------------------------------------
# 4) YOLO vs YOLO Optuna comparison
# ---------------------------------------------------------------------------
def plot_yolo_optuna_metrics(results: dict, output_dir: str, metric_suffix: str, file_suffix: str,
                              ylabel: str, title_prefix: str, subset: str = "overall",
                              plot_baseline: bool = True, log_scale: bool = False,
                              strategies=None, comparison_label: str = "YOLO Optuna Comparison",
                              file_prefix: str = "yolo_optuna",
                              legend_strategies=None):
    """results[group][f'{n} Example ({strategy})'] = {'macro': {subset: {...}}}
    strategies:        list of series to plot (default: ['YOLO', 'YOLO Optuna']).
    legend_strategies: if provided, overrides which strategies get a marker entry
                       in the legend (useful to avoid repeating the Optuna/Matching variants)."""
    strategies = strategies or ['YOLO', 'YOLO Optuna']

    def value_fn(group, example, strategy, field):
        res = results.get(group, {}).get(f"{example} ({strategy})")
        if res and "macro" in res and subset in res["macro"]:
            return res["macro"][subset].get(f"{field}_{metric_suffix}")
        return None

    def baseline_fn(field):
        return st.BASELINE[f"{field}_{metric_suffix}"] if plot_baseline else None

    metric_label = "MAE (%)" if metric_suffix == "MAE_%" else "MAE (Value)"

    def title_fn(f):
        return f"{metric_label} - {st.FIELD_LABELS_IT.get(f, f.capitalize())}"

    def ylabel_fn(_f):
        return metric_label

    def y_lim_fn(f):
        if not log_scale:
            return None
        if metric_suffix == "MAE_%":
            return st.MAE_PERC_LOG_YLIM.get(f)
        return st.get_mae_abs_log_ylim(f, subset)

    def y_ticks_fn(f):
        if not log_scale:
            return None
        if metric_suffix == "MAE_%":
            return st.MAE_PERC_LOG_YTICKS.get(f)
        return st.get_mae_abs_log_yticks(f, subset)

    out_dir = os.path.join(output_dir, subset)
    file_suffix_out = f"{file_suffix}_log" if log_scale else file_suffix
    title_suffix = " (Log Scale)" if log_scale else ""

    # If the strategies include Optuna variants, use linestyles to distinguish them
    # from the base version: solid = classic prompting, dashed = Optuna.
    has_optuna = any('Optuna' in s for s in strategies)
    linestyles_arg = st.STRATEGY_LINESTYLES if has_optuna else None

    extra_handles = None
    if has_optuna:
        extra_handles = [
            mlines.Line2D([], [], color='#444444', linestyle='-',  linewidth=2.5,
                          label='Classic prompting'),
            mlines.Line2D([], [], color='#444444', linestyle='--', linewidth=2.5,
                          label='Optuna optimization'),
        ]

    plot_dot_grid(
        fields=st.MACRO_FIELDS,
        value_fn=value_fn,
        strategies=strategies,
        colors=st.STRATEGY_COLORS,
        markers=st.STRATEGY_MARKERS,
        out_path_combined=os.path.join(out_dir, f"{file_prefix}_{subset}_{file_suffix_out}_combined.png"),
        out_path_single_template=os.path.join(out_dir, f"{file_prefix}_{subset}_{file_suffix_out}_{{field}}.png"),
        title_fn=title_fn, ylabel_fn=ylabel_fn, baseline_fn=baseline_fn,
        log_scale=log_scale,
        connect_lines=True,
        linestyles=linestyles_arg,
        extra_legend_handles=extra_handles,
        legend_strategies=legend_strategies if legend_strategies is not None else (
            [s for s in strategies if 'Optuna' not in s] if has_optuna else None
        ),
        y_lim_fn=y_lim_fn, y_ticks_fn=y_ticks_fn,
        suptitle=f"{comparison_label} - {title_prefix}{title_suffix}",
        legend_ncol=len(strategies) + (2 if extra_handles else 0),
    )
# ---------------------------------------------------------------------------
# 5) Threshold study (reference style)
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
        ['Standardized F1-score vs Threshold', 'Standardized IoU vs Threshold'],
        ['F1-score', 'IoU']
    ):
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Similarity Threshold', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis='both', labelsize=13)
        ax.invert_xaxis()
        ax.grid(True, axis='y', linestyle='--', color='#cccccc', zorder=0)
        ax.grid(False, axis='x')
        ax.set_ylim(0, 0.4)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])

    leg_items = [
        mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=9, label='E2B model'),
        mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=9, label='E4B model'),
        mlines.Line2D([], [], color='gray', linestyle='-', linewidth=2, label='Q4'),
        mlines.Line2D([], [], color='gray', linestyle='--', linewidth=2, label='Q8'),
    ]
    for var in STRATEGIES_MAIN:
        c = st.STRATEGY_COLORS.get(var, '#333333')
        leg_items.append(mlines.Line2D([], [], color=c, linewidth=4, markersize=11, label=var))

    fig.legend(handles=leg_items, loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=5, fontsize=11, frameon=False, title='Strategies and Models', title_fontsize=12)

    plt.tight_layout(rect=[0, 0.12, 1, 1.0])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'nlp_threshold_study.png'), bbox_inches='tight', dpi=300)
    plt.close()


import matplotlib.pyplot as plt
import numpy as np


def plot_combined_strategies(data_dict, output_path, metric_name="MAE (%) - Calories"):
    """
    Generate the combined chart.
    data_dict: dictionary whose keys are (Model_Quantization), e.g. "E2B - Q4"
               and whose values are dicts keyed by strategy (e.g. "Standard_NoOptuna_NoSemanticMatching")
               each containing a list/tuple of 3 values (1, 5, 10 examples).
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Style mapping (Marker, LineStyle, Color)
    style_map = {
        # Standard family (circle 'o')
        "Standard": {"marker": "o", "ls": "-", "color": "#1f77b4"},  # Dark blue
        "Standard_Optuna": {"marker": "o", "ls": "--", "color": "#1f77b4"},
        "Standard_Matching": {"marker": "o", "ls": "-", "color": "#9ecae1"},  # Light blue
        "Standard_Optuna_Matching": {"marker": "o", "ls": "--", "color": "#9ecae1"},

        # YOLO family (cross 'P' or '+')
        "YOLO": {"marker": "P", "ls": "-", "color": "#ff7f0e"},  # Dark orange
        "YOLO_Optuna": {"marker": "P", "ls": "--", "color": "#ff7f0e"},
        "YOLO_Matching": {"marker": "P", "ls": "-", "color": "#fdd0a2"},  # Light orange
        "YOLO_Optuna_Matching": {"marker": "P", "ls": "--", "color": "#fdd0a2"},
    }

    x_labels = list(data_dict.keys())
    x_positions = np.arange(len(x_labels)) * 2  # Space out the groups (e.g. E2B-Q4, E2B-Q8)

    # Inner offset for the 3 points (1, 5, 10 examples)
    offsets = [-0.3, 0, 0.3]

    # Keep track of the labels already added to the legend
    added_to_legend = set()

    for i, x_label in enumerate(x_labels):
        strategies = data_dict[x_label]
        for strat_name, y_values in strategies.items():
            if strat_name not in style_map:
                continue

            style = style_map[strat_name]
            x_vals = x_positions[i] + np.array(offsets)

            # Plot the line and the points
            label = strat_name if strat_name not in added_to_legend else ""
            ax.plot(x_vals, y_values,
                    marker=style["marker"],
                    linestyle=style["ls"],
                    color=style["color"],
                    markersize=8,
                    linewidth=1.5,
                    label=label)

            added_to_legend.add(strat_name)

    # Grid and axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontweight='bold', fontsize=12)
    ax.set_ylabel(metric_name.split(" - ")[0], fontsize=12)
    ax.set_title(metric_name, fontweight='bold', fontsize=14)

    # Grid style similar to prompting_comparison_overall_mae_perc_log_calories.png
    ax.grid(axis='y', linestyle='-', alpha=0.3)
    for pos in x_positions[:-1]:
        ax.axvline(x=pos + 1, color='gray', linestyle=':', alpha=0.5)  # Vertical dividers

    ax.axhline(y=25, color='gray', linestyle='--')  # Baseline line (e.g. Nutrition 5k)
    ax.text(-0.5, 26, 'Nutrition 5k', color='gray', fontweight='bold')

    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=4, frameon=False, fontsize=11,
              title="Strategies (Left=1 ex., Center=5 ex., Right=10 ex.)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
