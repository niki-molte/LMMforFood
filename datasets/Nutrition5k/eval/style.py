"""
style.py
========
Single plotting style shared by all the charts of the package.

The reference style is that of the few-shot "MAE% dot-plot" charts and of the
"threshold study": per-strategy markers, fixed colors,
offset per number of examples, dashed baseline line with label,
highlighting of the points "below baseline", horizontal legend below the figure.
Any new chart must use ONLY the functions/constants of this module
for the aesthetics, so that the style stays identical everywhere.
"""
import numpy as np
import matplotlib.lines as mlines
from matplotlib.ticker import ScalarFormatter, NullFormatter, FuncFormatter

# ---------------------------------------------------------------------------
# Model groups (X axis shared by all the charts)
# ---------------------------------------------------------------------------
GROUPS = ['E2B - Q4', 'E2B - Q8', 'E4B - Q4', 'E4B - Q8']

# ---------------------------------------------------------------------------
# "Strategy" palette (color + marker), the same used in the few-shot MAE%
# charts and in the threshold study.
# ---------------------------------------------------------------------------
STRATEGY_COLORS = {
    'Standard': '#3385ff',
    'Two-Step': '#ff4d4d',
    'Hybrid v1': '#47d147',
    'Hybrid v2': '#9933ff',
    'YOLO': '#ff9933',
    'YOLO Optuna': '#ff9933',            # same color as YOLO, distinguished by a dashed line
    'Standard Optuna': '#3385ff',         # same color as Standard, distinguished by a dashed line
    # Semantic Matching variants: same symbol/scheme but a lighter color
    'Standard Matching': '#99c2ff',
    'Standard Optuna Matching': '#99c2ff',
    'YOLO Matching': '#ffc266',
    'YOLO Optuna Matching': '#ffc266',
}
STRATEGY_MARKERS = {
    'Standard': 'o',
    'Two-Step': 's',
    'Hybrid v1': '^',
    'Hybrid v2': 'D',
    'YOLO': 'P',
    'YOLO Optuna': 'P',              # same symbol as YOLO, distinguished by a dashed line
    'Standard Optuna': 'o',          # same symbol as Standard, distinguished by a dashed line
    'Standard Matching': 'o',
    'Standard Optuna Matching': 'o',
    'YOLO Matching': 'P',
    'YOLO Optuna Matching': 'P',
}

# Connection line style per strategy.
# Optuna variants use a dashed line; base strategies use a solid line.
STRATEGY_LINESTYLES = {
    'Standard': '-',
    'Two-Step': '-',
    'Hybrid v1': '-',
    'Hybrid v2': '-',
    'YOLO': '-',
    'YOLO Optuna': '--',
    'Standard Optuna': '--',
    'Standard Matching': '-',
    'Standard Optuna Matching': '--',
    'YOLO Matching': '-',
    'YOLO Optuna Matching': '--',
}

# Multiplicative factor on the scatter size (s) for markers that
# visually look smaller than their nominal size.
MARKER_SIZE_SCALE = {
    '*': 1.9,   # star: used in OPTUNA_STRATEGY_MARKERS per linear_bias_all5
}

# "Optuna strategy" palette (used in the comparison between scaling strategies)
OPTUNA_STRATEGIES = ["linear_all5", "linear_mass_only", "power_law_all5",
                     "piecewise_all5", "linear_bias_all5"]
OPTUNA_STRATEGY_COLORS = {
    "linear_all5": "#3385ff",
    "linear_mass_only": "#ff4d4d",
    "power_law_all5": "#47d147",
    "piecewise_all5": "#9933ff",
    "linear_bias_all5": "#ff9900",
}
OPTUNA_STRATEGY_MARKERS = {
    "linear_all5": "o",
    "linear_mass_only": "s",
    "power_law_all5": "^",
    "piecewise_all5": "D",
    "linear_bias_all5": "*",
}
OPTUNA_STRATEGY_LABELS = {
    "linear_all5": "Linear All5",
    "linear_mass_only": "Linear Mass Only",
    "power_law_all5": "Power Law All5",
    "piecewise_all5": "Piecewise All5",
    "linear_bias_all5": "Linear Bias All5",
}
YOLO_BASE_KEY = "__yolo_base__"
OPTUNA_STRATEGY_COLORS[YOLO_BASE_KEY] = "#555555"
OPTUNA_STRATEGY_MARKERS[YOLO_BASE_KEY] = "P"
OPTUNA_STRATEGY_LABELS[YOLO_BASE_KEY] = "YOLO (no Optuna)"

STANDARD_BASE_KEY = "__standard_base__"

# X-axis offset to visually group 1/5/10 examples
EXAMPLES = ['1 Example', '5 Examples', '10 Examples']
EX_OFFSETS = {'1 Example': -0.22, '5 Examples': 0.0, '10 Examples': 0.22}

# Display labels
FIELD_LABELS_IT = {
    'calories': 'Calories', 'mass': 'Mass', 'fat': 'Fat',
    'carbohydrates': 'Carbohydrates', 'protein': 'Protein',
}
METRIC_LABELS_IT = {'P': 'Precision', 'R': 'Recall', 'F1': 'F1-score', 'IoU': 'IoU'}
STD_KEY_IT = {'Standardized': 'Standardized', 'Non-Standardized': 'Non-Standardized'}

MACRO_FIELDS = ["calories", "mass", "fat", "carbohydrates", "protein"]

BASELINE = {
    "calories_MAE": 70.6, "calories_MAE_%": 26.1,
    "mass_MAE": 40.4, "mass_MAE_%": 18.8,
    "fat_MAE": 5.0, "fat_MAE_%": 34.2,
    "carbohydrates_MAE": 6.1, "carbohydrates_MAE_%": 31.9,
    "protein_MAE": 5.5, "protein_MAE_%": 29.5,
}

# Specific limits/ticks for MAE% on a log scale
MAE_PERC_LOG_YLIM = {
    'calories': (19, 400), 'mass': (15, 1500), 'fat': (26, 500),
    'carbohydrates': (25, 600), 'protein': (21, 250),
}
MAE_PERC_LOG_YTICKS = {
    'calories': [30, 100, 300], 'mass': [30, 100, 300, 1000],
    'fat': [30, 100, 300], 'carbohydrates': [30, 100, 300],
    'protein': [30, 50, 100, 200],
}

# Specific limits/ticks for Absolute MAE on a log scale (overall and exact_match)
MAE_ABS_LOG_YLIM = {
    'overall': {
        'calories': (55, 1000), 'mass': (32, 2600), 'fat': (4, 50),
        'carbohydrates': (5, 130), 'protein': (4.5, 65),
    },
    'exact_match': {
        'calories': (20, 200), 'mass': (9, 150), 'fat': (1.4, 12),
        'carbohydrates': (0.9, 16), 'protein': (0.9, 12),
    },
}
MAE_ABS_LOG_YTICKS = {
    'overall': {
        'calories': [70, 100, 300, 1000], 'mass': [40, 100, 300, 1000],
        'fat': [5, 10, 20, 40], 'carbohydrates': [6, 10, 30, 100],
        'protein': [6, 10, 20, 50],
    },
    'exact_match': {
        'calories': [30, 50, 100, 200], 'mass': [10, 30, 100],
        'fat': [2, 3, 5, 10], 'carbohydrates': [1, 2, 5, 10],
        'protein': [1, 2, 5, 10],
    },
}


def get_mae_abs_log_ylim(field: str, subset: str = "overall"):
    return MAE_ABS_LOG_YLIM.get(subset, MAE_ABS_LOG_YLIM['overall']).get(field)


def get_mae_abs_log_yticks(field: str, subset: str = "overall"):
    return MAE_ABS_LOG_YTICKS.get(subset, MAE_ABS_LOG_YTICKS['overall']).get(field)


def get_marker(group_name: str) -> str:
    if 'E2B' in group_name:
        return 'o'
    elif 'E4B' in group_name:
        return '^'
    return 'x'


def get_linestyle(group_name: str) -> str:
    if 'Q4' in group_name:
        return '-'
    elif 'Q8' in group_name:
        return '--'
    return '-'


def draw_baseline(ax, base_val, log_scale=False, label=' Nutrition 5k'):
    """Dashed gray baseline line + inline label (few-shot MAE% style)."""
    ax.axhline(y=base_val, color='gray', linestyle='--', linewidth=1.2, zorder=1)
    if log_scale:
        text_y = base_val * 1.05
        ax.text(x=-0.45, y=text_y, s=label, color='gray', fontsize=11,
                va='bottom', fontweight='bold', zorder=2)
    else:
        ax.annotate(label, xy=(-0.45, base_val), xytext=(0, 3),
                    textcoords='offset points', color='gray', fontsize=11,
                    va='bottom', fontweight='bold', zorder=2)


def _format_log_tick(y, _pos):
    if y <= 0:
        return ""
    if abs(y - round(y)) < 1e-6:
        return f"{int(round(y))}"
    return f"{y:g}"


def style_dot_axis(ax, ylabel, title, log_scale=False, xlabel=None,
                    y_lim=None, y_ticks=None, tick_fontsize=12, title_fontsize=14):
    """Common cosmetics of a dot-plot panel: grid, xticks=GROUPS, limits."""
    ax.set_ylabel(ylabel, fontsize=tick_fontsize)
    if log_scale:
        ax.set_yscale('log')
        data_ymin, data_ymax = ax.dataLim.ymin, ax.dataLim.ymax
        if y_lim is not None:
            low, high = y_lim
            if np.isfinite(data_ymin) and data_ymin > 0 and data_ymin < low:
                low = data_ymin * 0.88
            if np.isfinite(data_ymax) and data_ymax > high:
                high = data_ymax * 1.15
            ax.set_ylim((low, high))
        else:
            low, high = ax.get_ylim()

        candidate_pool = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 3000, 10000]
        if y_ticks is not None:
            ticks = list(y_ticks)
            for c in candidate_pool:
                if c > max(ticks) and c <= high * 0.95:
                    ticks.append(c)
            ticks = [t for t in sorted(set(ticks)) if low <= t <= high]
        else:
            ticks = [c for c in candidate_pool if low <= c <= high]

        if ticks:
            ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(_format_log_tick))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis='y', which='minor', length=3)
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    ax.set_xticks(np.arange(len(GROUPS)))
    ax.set_xticklabels(GROUPS, fontsize=tick_fontsize, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize - 1)
    ax.grid(False, axis='x')
    ax.grid(True, axis='y', linestyle='-', color='#eeeeee', zorder=0)
    ax.set_xlim(-0.5, len(GROUPS) - 0.5)


def strategy_legend_handles(strategies, colors, markers, prefix="", label_map=None):
    """Build the legend handles 'empty colored marker', few-shot style."""
    handles = []
    for s in strategies:
        m = markers[s]
        ms = 11 * MARKER_SIZE_SCALE.get(m, 1.0)
        display_label = label_map.get(s, s) if label_map else OPTUNA_STRATEGY_LABELS.get(s, s)
        handles.append(mlines.Line2D(
            [], [], color='none', marker=m,
            markerfacecolor=colors[s], markeredgecolor='white',
            markersize=ms, label=f"{prefix}{display_label}"))
    return handles


def add_bottom_legend(fig, handles, title, ncol=None, anchor=(0.5, 0.06), fontsize=11):
    ncol = ncol or len(handles)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=anchor,
               ncol=ncol, fontsize=fontsize, frameon=False,
               title=title, title_fontsize=12)
