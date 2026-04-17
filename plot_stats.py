import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import kruskal, ks_2samp

df = pd.read_csv("output.csv")

# --- Parse marker from filename ---
def parse_metadata(filepath):
    match = re.search(r'(WT|KO),?\s*(Chow|WD)', filepath, re.IGNORECASE)
    condition = match.group(1).upper() if match else None
    diet = match.group(2).upper() if match else None
    marker = filepath.split('/')[0] if '/' in filepath else None
    return condition, diet, marker

df[['condition', 'diet', 'marker']] = df['file'].apply(
    lambda f: pd.Series(parse_metadata(f))
)

# --- Plot helper ---
def plot_dab(df, dab_level, metric, ax):
    if metric == 'rgb':
        col = f"{dab_level}_dab_{metric}_mean"
        err_col = f"{dab_level}_dab_{metric}_std"
    else:
        col = f"{dab_level}_dab_mean"
        err_col = f"{dab_level}_dab_std"

    colors = {'WT': '#4C72B0', 'KO': '#DD8452'}
    diets = ['CHOW', 'WD']
    offsets = {'WT': -0.15, 'KO': 0.15}
    legend_added = set()
    group_samples = {}


    for condition, grp in df.groupby('condition'):
        color = colors[condition]
        for i, diet in enumerate(diets):
            subgrp = grp[grp['diet'] == diet]
            group_samples[(condition, diet)] = subgrp[col].values
            x_positions = i + offsets[condition] + np.random.uniform(-0.05, 0.05)
            for _, row in subgrp.iterrows():
                ax.errorbar(
                    x=x_positions,
                    y=row[col],
                    yerr=row[err_col],
                    fmt='o', color=color, capsize=3,
                    linewidth=1.2, markersize=6, alpha=0.8,
                    label=condition if condition not in legend_added else None
                )
                legend_added.add(condition)

    results = []
    for diet in diets:
        wt = group_samples.get(('WT', diet))
        ko = group_samples.get(('KO', diet))
        if wt is not None and ko is not None:
            stat, p = kruskal(wt, ko)
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
            results.append(f"{diet}: p={p_str}")

    ax.set_title(
        f"{dab_level.capitalize()} DAB ({metric.upper()}) [Kruskal-Wallis] WT v. KO\n" + "  ".join(results),
        fontsize=11, fontweight='bold'
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(diets)
    ax.set_xlim(-0.5, 1.5)
    #ax.set_title(f"{dab_level.capitalize()} DAB ({metric.upper()})", fontsize=13, fontweight='bold')
    ax.set_xlabel("Diet")
    ax.set_ylabel(f"DAB {metric.upper()} Mean ± Std")
    ax.legend(title="Condition")

def plot_dab_kde(df, dab_level, metric, ax):
    if metric == 'rgb':
        col = f"{dab_level}_dab_{metric}_mean"
    else:
        col = f"{dab_level}_dab_mean"

    colors = {'WT': '#4C72B0', 'KO': '#DD8452'}
    diets = ['CHOW', 'WD']
    linestyles = {'CHOW': '-', 'WD': '--'}
    group_samples = {}

    for condition, grp in df.groupby('condition'):
        color = colors[condition]
        for diet in diets:
            vals = grp[grp['diet'] == diet][col].values
            if len(vals) < 2:
                continue
            group_samples[(condition, diet)] = vals
            kde = gaussian_kde(vals)
            x_range = np.linspace(vals.min() - vals.std(), vals.max() + vals.std(), 200)
            ax.plot(
                x_range, kde(x_range),
                color=color, linestyle=linestyles[diet],
                linewidth=1.8, label=f"{condition} {diet}"
            )
            ax.fill_between(x_range, kde(x_range), alpha=0.15, color=color)

    results = []
    for diet in diets:
        wt = group_samples.get(('WT', diet))
        ko = group_samples.get(('KO', diet))
        if wt is not None and ko is not None:
            stat, p = ks_2samp(wt, ko)
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
            results.append(f"{diet}: p={p_str}")

    ax.set_title(
        f"{dab_level.capitalize()} DAB ({metric.upper()}) [Kolmogorov-Smirnov] WT v. KO\n" + "  ".join(results),
        fontsize=11, fontweight='bold'
    )
    ax.set_xlabel(f"DAB {metric.upper()} Mean")
    ax.set_ylabel("Density")
    #ax.set_title(f"{dab_level.capitalize()} DAB ({metric.upper()})", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

# --- Loop over markers ---
for marker, marker_df in df.groupby('marker'):
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))
    #plot_dab(marker_df, 'low',  'hed', axes1[0])
    #plot_dab(marker_df, 'high', 'hed', axes1[1])
    plot_dab_kde(marker_df, 'low',  'hed', axes1[0])
    plot_dab_kde(marker_df, 'high', 'hed', axes1[1])
    fig1.suptitle(f"{marker} — DAB Distribution by Diet and Condition", fontsize=15, fontweight='bold')
    fig1.tight_layout()
    fig1.savefig(f"{marker}_dab_comparison_kde.png", dpi=150, bbox_inches='tight')

    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))
    plot_dab(marker_df, 'low',  'hed', axes1[0])
    plot_dab(marker_df, 'high', 'hed', axes1[1])
    #plot_dab_kde(marker_df, 'low',  'hed', axes1[0])
    #plot_dab_kde(marker_df, 'high', 'hed', axes1[1])
    fig1.suptitle(f"{marker} — DAB Intensity by Diet and Condition", fontsize=15, fontweight='bold')
    fig1.tight_layout()
    fig1.savefig(f"{marker}_dab_comparison.png", dpi=150, bbox_inches='tight')

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    #plot_dab(marker_df, 'low',  'rgb', axes2[0])
    #plot_dab(marker_df, 'high', 'rgb', axes2[1])
    plot_dab_kde(marker_df, 'low',  'rgb', axes2[0])
    plot_dab_kde(marker_df, 'high', 'rgb', axes2[1])
    fig2.suptitle(f"{marker} — RGB Distribution by Diet and Condition", fontsize=15, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(f"{marker}_rgb_comparison_kde.png", dpi=150, bbox_inches='tight')

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    plot_dab(marker_df, 'low',  'rgb', axes2[0])
    plot_dab(marker_df, 'high', 'rgb', axes2[1])
    #plot_dab_kde(marker_df, 'low',  'rgb', axes2[0])
    #plot_dab_kde(marker_df, 'high', 'rgb', axes2[1])
    fig2.suptitle(f"{marker} — RGB Intensity by Diet and Condition", fontsize=15, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(f"{marker}_rgb_comparison.png", dpi=150, bbox_inches='tight')

    plt.close('all')