"""
style.py
========
Stile grafico UNICO, condiviso da tutti i grafici del pacchetto.

Lo stile di riferimento è quello dei grafici few-shot "MAE% dot-plot" e dello
"studio soglia" del vecchio script: marker per strategia, colori fissi,
offset per numero di esempi, linea di baseline tratteggiata con etichetta,
evidenziazione dei punti "sotto baseline", legenda orizzontale sotto la figura.
Qualunque nuovo grafico deve usare SOLO le funzioni/costanti di questo modulo
per la parte estetica, in modo che lo stile resti identico ovunque.
"""
import numpy as np
import matplotlib.lines as mlines
from matplotlib.ticker import ScalarFormatter, NullFormatter

# ---------------------------------------------------------------------------
# Gruppi di modelli (asse X comune a tutti i grafici)
# ---------------------------------------------------------------------------
GROUPS = ['E2B - Q4', 'E2B - Q8', 'E4B - Q4', 'E4B - Q8']

# ---------------------------------------------------------------------------
# Palette "strategia" (colore + marker), identica a quella usata nei grafici
# few-shot MAE% e nello studio soglia.
# ---------------------------------------------------------------------------
STRATEGY_COLORS = {
    'Standard': '#3385ff',
    'Two-Step': '#ff4d4d',
    'Hybrid v1': '#47d147',
    'Hybrid v2': '#9933ff',
    'YOLO': '#ff9933',
    'YOLO Optuna': '#cc0000',
}
STRATEGY_MARKERS = {
    'Standard': 'o',
    'Two-Step': 's',
    'Hybrid v1': '^',
    'Hybrid v2': 'D',
    'YOLO': 'P',
    'YOLO Optuna': 'X',
}

# Palette "strategia Optuna" (usata nel confronto tra strategie di scaling)
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

# Offset sull'asse X per raggruppare visivamente 1/5/10 esempi
EXAMPLES = ['1 Example', '5 Examples', '10 Examples']
EX_OFFSETS = {'1 Example': -0.22, '5 Examples': 0.0, '10 Examples': 0.22}

# Etichette italiane
FIELD_LABELS_IT = {
    'calories': 'Calorie', 'mass': 'Massa', 'fat': 'Grassi',
    'carbohydrates': 'Carboidrati', 'protein': 'Proteine',
}
METRIC_LABELS_IT = {'P': 'Precisione', 'R': 'Recall', 'F1': 'F1-score', 'IoU': 'IoU'}
STD_KEY_IT = {'Standardized': 'Standardizzato', 'Non-Standardized': 'Non Standardizzato'}

MACRO_FIELDS = ["calories", "mass", "fat", "carbohydrates", "protein"]

BASELINE = {
    "calories_MAE": 70.6, "calories_MAE_%": 26.1,
    "mass_MAE": 40.4, "mass_MAE_%": 18.8,
    "fat_MAE": 5.0, "fat_MAE_%": 34.2,
    "carbohydrates_MAE": 6.1, "carbohydrates_MAE_%": 31.9,
    "protein_MAE": 5.5, "protein_MAE_%": 29.5,
}

# Limiti/tick specifici per il MAE% in scala log (stile originale)
MAE_PERC_LOG_YLIM = {
    'calories': (19, 400), 'mass': (15, 1500), 'fat': (26, 500),
    'carbohydrates': (25, 600), 'protein': (21, 250),
}
MAE_PERC_LOG_YTICKS = {
    'calories': [30, 100, 300], 'mass': [30, 100, 300, 1000],
    'fat': [30, 100, 300], 'carbohydrates': [30, 100, 300],
    'protein': [30, 50, 100, 200],
}


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
    """Linea di baseline tratteggiata grigia + etichetta inline (stile MAE% few-shot)."""
    ax.axhline(y=base_val, color='gray', linestyle='--', linewidth=1.2, zorder=1)
    text_y = base_val * 1.05 if log_scale else base_val + (base_val * 0.02)
    ax.text(x=-0.45, y=text_y, s=label, color='gray', fontsize=11,
            va='bottom', fontweight='bold', zorder=2)


def style_dot_axis(ax, ylabel, title, log_scale=False, xlabel=None,
                    y_lim=None, y_ticks=None, tick_fontsize=12, title_fontsize=14):
    """Cosmesi comune di un pannello dot-plot: griglia, xticks=GROUPS, limiti."""
    ax.set_ylabel(ylabel, fontsize=tick_fontsize)
    if log_scale:
        ax.set_yscale('log')
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis='y', which='minor', length=3)
        else:
            ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    ax.set_xticks(np.arange(len(GROUPS)))
    ax.set_xticklabels(GROUPS, fontsize=tick_fontsize, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize - 1)
    ax.grid(False, axis='x')
    ax.grid(True, axis='y', linestyle='-', color='#eeeeee', zorder=0)
    ax.set_xlim(-0.5, len(GROUPS) - 0.5)


def strategy_legend_handles(strategies, colors, markers, prefix=""):
    """Costruisce gli handle di legenda 'marker vuoto colorato', stile few-shot."""
    handles = []
    for s in strategies:
        handles.append(mlines.Line2D(
            [], [], color='none', marker=markers[s],
            markerfacecolor=colors[s], markeredgecolor='white',
            markersize=11, label=f"{prefix}{s}"))
    return handles


def add_bottom_legend(fig, handles, title, ncol=None, anchor=(0.5, 0.06), fontsize=11):
    ncol = ncol or len(handles)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=anchor,
               ncol=ncol, fontsize=fontsize, frameon=False,
               title=title, title_fontsize=12)
