"""Figures built only from saved results and exploratory PCA coordinates."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _enter_project(project_dir):
    project_dir = Path(project_dir).resolve()
    os.chdir(project_dir)
    return project_dir


def create_pca_plots(project_dir, data_file, results_dir):
    """Save 25-, 4-, 7-region and region/donor/cell-type PCA views."""
    from pathlib import Path
    _enter_project(project_dir)
    DATA_FILE = Path(data_file)
    RESULTS_DIR = Path(results_dir)

    # PCA data and labels

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    raw = pd.read_csv(DATA_FILE, index_col=0)
    X = raw.T
    metadata = pd.DataFrame(index=X.index)
    parts = [name.split("_") for name in X.index]
    metadata["donor"] = [part[0] for part in parts]
    metadata["region"] = [part[1] for part in parts]
    metadata["fraction_label"] = ["Neuronal" if part[2] == "N" else "Non-neuronal"
                                  for part in parts]
    REGION_LABELS = sorted(metadata.region.unique())
    y_region = metadata.region.to_numpy()
    gene_variance = X.var(axis=0, ddof=0)
    top_variable_genes = gene_variance.sort_values(ascending=False).head(5000).index
    pca = PCA(n_components=10, random_state=42)
    pca_df = pd.DataFrame(pca.fit_transform(X[top_variable_genes]),
                          index=X.index, columns=[f"PC{i}" for i in range(1, 11)]).join(metadata)
    explained_variance = pca.explained_variance_ratio_ * 100
    print(f"PCA on {len(X)} samples; PC1 {explained_variance[0]:.1f}%, PC2 {explained_variance[1]:.1f}%")

    MACRO_GROUPS = {
        "Forebrain": ["ACC", "DLPFC", "EC", "INS", "ITC", "OFC", "PAC", "PMC", "PSC", "PSTC",
                      "PVPC", "VLPFC", "PVC", "AMY", "HIPP"],
        "Basal Ganglia": ["CN", "NAC", "GP"],
        "Midbrain/Diencephalon": ["RMTG", "VTA", "DRN", "HAB", "MDT", "ARC"],
        "Hindbrain": ["CRBLM"],
    }
    MACRO_LABELS = list(MACRO_GROUPS)
    REGION_TO_MACRO = {region: macro for macro, regions in MACRO_GROUPS.items() for region in regions}

    # Every region must appear exactly once
    assert sum(len(r) for r in MACRO_GROUPS.values()) == len(REGION_TO_MACRO)
    assert set(REGION_TO_MACRO) == set(REGION_LABELS), "Some regions are missing or wrongly assigned."

    y_macro = np.array([REGION_TO_MACRO[region] for region in y_region])

    MACRO_7_MAP = {
        # 1. Prefrontal Cortex (PFC)
        "DLPFC": "Prefrontal_Cortex", "VLPFC": "Prefrontal_Cortex",
        "OFC":   "Prefrontal_Cortex", "ACC":   "Prefrontal_Cortex",
        # 2. Sensorimotor Neocortex
        "PVC": "Sensorimotor_Cortex", "PAC": "Sensorimotor_Cortex",
        "PSC": "Sensorimotor_Cortex", "PMC": "Sensorimotor_Cortex",
        # 3. Posterior Association Cortex
        "ITC": "Parieto_Temporal_Assoc", "PSTC": "Parieto_Temporal_Assoc",
        "PVPC": "Parieto_Temporal_Assoc", "INS": "Parieto_Temporal_Assoc",
        # 4. Limbic & Allocortex
        "AMY": "Limbic_Allocortex", "HIPP": "Limbic_Allocortex", "EC": "Limbic_Allocortex",
        # 5. Basal Ganglia
        "CN": "Basal_Ganglia", "NAC": "Basal_Ganglia", "GP": "Basal_Ganglia",
        # 6. Midbrain & Diencephalon
        "VTA": "Midbrain_Diencephalon", "DRN": "Midbrain_Diencephalon", "RMTG": "Midbrain_Diencephalon",
        "HAB": "Midbrain_Diencephalon", "MDT": "Midbrain_Diencephalon", "ARC": "Midbrain_Diencephalon",
        # 7. Hindbrain / Cerebellum
        "CRBLM": "Cerebellum_Hindbrain",
    }
    MACRO_7_LABELS = list(dict.fromkeys(MACRO_7_MAP.values()))   # 7 classes, in the order above
    assert set(MACRO_7_MAP) == set(REGION_LABELS), "Some regions are missing or wrongly assigned."

    y_macro7 = np.array([MACRO_7_MAP[region] for region in y_region])
    # PCA figures

    # Reuse the Section 6 PCA coordinates, so only the coloring changes.
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_data = pca_df.copy()
    plot_data["macro_4"] = plot_data["region"].map(REGION_TO_MACRO)
    plot_data["macro_7"] = plot_data["region"].map(MACRO_7_MAP)
    assert plot_data[["region", "macro_4", "macro_7"]].notna().all().all()

    region_order = sorted(plot_data.region.unique())
    macro4_order = list(MACRO_LABELS)
    macro7_order = list(MACRO_7_LABELS)

    # Each exact region has one color. Shapes additionally identify its 4-macro group.
    region_colors = dict(zip(region_order, sns.color_palette("husl", len(region_order))))
    macro4_colors = dict(zip(macro4_order, sns.color_palette("colorblind", len(macro4_order))))
    macro7_colors = dict(zip(macro7_order, sns.color_palette("colorblind", len(macro7_order))))
    macro4_markers = dict(zip(macro4_order, ["o", "s", "^", "D"]))

    x_label = f"PC1 ({explained_variance[0]:.1f}% variance)"
    y_label = f"PC2 ({explained_variance[1]:.1f}% variance)"
    x_range = plot_data.PC1.max() - plot_data.PC1.min()
    y_range = plot_data.PC2.max() - plot_data.PC2.min()
    x_limits = (plot_data.PC1.min() - .07 * x_range, plot_data.PC1.max() + .07 * x_range)
    y_limits = (plot_data.PC2.min() - .07 * y_range, plot_data.PC2.max() + .07 * y_range)
    output_dir = RESULTS_DIR / "pca_region_scatter"
    output_dir.mkdir(parents=True, exist_ok=True)


    def plot_region_pca(label_column, order, colors, title, filename, use_macro_markers=False):
        """Draw the shared PC1–PC2 projection with a categorical legend."""
        fig, ax = plt.subplots(figsize=(13.5 if use_macro_markers else 10.5, 7))
        for label in order:
            subset = plot_data[plot_data[label_column] == label]
            marker = macro4_markers[REGION_TO_MACRO[label]] if use_macro_markers else "o"
            ax.scatter(subset.PC1, subset.PC2, color=colors[label], marker=marker,
                       s=58, alpha=.85, edgecolors="white", linewidths=.45,
                       label=label.replace("_", " "), zorder=2)
        ax.set(xlabel=x_label, ylabel=y_label, xlim=x_limits, ylim=y_limits,
               title=title)
        ax.grid(color="#E7E7E7", linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title="Brain region" if use_macro_markers else "Macro region",
                  loc="center left", bbox_to_anchor=(1.01, .5),
                  ncol=2 if use_macro_markers else 1, fontsize=8.5,
                  title_fontsize=9.5, frameon=False, markerscale=1.2,
                  columnspacing=1.2)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=250, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return fig


    _ = plot_region_pca("region", region_order, region_colors,
                        "PCA: 25 exact brain regions", "pca_25_regions.png",
                        use_macro_markers=True)
    _ = plot_region_pca("macro_4", macro4_order, macro4_colors,
                        "PCA: 4 macro regions", "pca_4_macro_regions.png")
    _ = plot_region_pca("macro_7", macro7_order, macro7_colors,
                        "PCA: 7 macro regions", "pca_7_macro_regions.png")

    # Use the Section 6 PCA coordinates (pca_df); only the labels/colors change.
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns

    three_view_data = pca_df.copy()
    assert three_view_data[["PC1", "PC2", "region", "donor", "fraction_label"]].notna().all().all()
    region_order = sorted(three_view_data.region.unique())
    donor_order = sorted(three_view_data.donor.unique())
    cell_order = ["Neuronal", "Non-neuronal"]
    region_colors = dict(zip(region_order, sns.color_palette("husl", len(region_order))))
    donor_colors = dict(zip(donor_order, sns.color_palette("colorblind", len(donor_order))))
    cell_colors = {"Neuronal": "#2878B5", "Non-neuronal": "#E07032"}
    macro4_markers = dict(zip(MACRO_LABELS, ["o", "s", "^", "D"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 9), sharex=True, sharey=True)
    panels = [
        ("region", region_order, region_colors, "25 brain regions", "Region", 5),
        ("donor", donor_order, donor_colors, "Donor", "Donor", 3),
        ("fraction_label", cell_order, cell_colors, "Cell type", "Cell type", 2),
    ]

    for ax, (column, labels, colors, title, legend_title, ncol) in zip(axes, panels):
        for label in labels:
            sample = three_view_data[three_view_data[column] == label]
            marker = (macro4_markers[REGION_TO_MACRO[label]] if column == "region" else "o")
            ax.scatter(sample.PC1, sample.PC2, s=52, color=colors[label], marker=marker,
                       alpha=.84, edgecolors="white", linewidths=.35,
                       label=label, zorder=2)
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_xlabel(f"PC1 ({explained_variance[0]:.1f}% variance)")
        ax.grid(color="#E8E8E8", linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title=legend_title, loc="upper center", bbox_to_anchor=(.5, -.13),
                  ncol=ncol, fontsize=8.2, title_fontsize=9.2, frameon=False,
                  columnspacing=1.1, handletextpad=.35, markerscale=1.25)

    axes[0].set_ylabel(f"PC2 ({explained_variance[1]:.1f}% variance)")
    fig.suptitle("The same gene-expression PCA, colored three ways", y=.97,
                 fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=.05, right=.99, top=.89, bottom=.30, wspace=.12)
    output = RESULTS_DIR / "pca_region_scatter/pca_regions_donors_cell_types.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [RESULTS_DIR / "pca_region_scatter" / name for name in
            ("pca_25_regions.png", "pca_4_macro_regions.png",
             "pca_7_macro_regions.png", "pca_regions_donors_cell_types.png")]


def create_performance_boxplots(project_dir, results_dir):
    """Save donor-wise macro F1 boxplots for all model families."""
    _enter_project(project_dir)

    # Model performance

    """Plot per-donor performance with the new PCA and joint-model results."""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Patch
    from sklearn.metrics import f1_score


    RESULTS = Path(results_dir)
    MODEL_STYLE = {
        # label: color, hatch for RBF, prefix for macro runs, direct-run folder
        "ANOVA": ("#2A78D6", "", "", "anova"),
        "ANOVA-RBF": ("#2A78D6", "////", "rbf_", "rbf_anova"),
        "MI": ("#EB6834", "", "mi_", "mi"),
        "MI-RBF": ("#EB6834", "////", "rbf_mi_", "rbf_mi"),
        "PCA": ("#1BAF7A", "", "pca_", "pca"),
        "PCA-RBF": ("#1BAF7A", "////", "rbf_pca_", "rbf_pca"),
    }

    # Joint runs use all samples. All six feature/kernel models use fixed settings.
    JOINT_FIXED_FOLDS = pd.read_csv(RESULTS / "joint_feature_comparison/fold_scores.csv")
    JOINT_CELL_FOLDS = pd.read_csv(RESULTS / "joint_feature_comparison/cell_type_fold_scores.csv")
    DIRECT_CELL_FOLDS = pd.read_csv(RESULTS / "joint_feature_comparison/direct_cell_type_fold_scores.csv")
    JOINT_METHODS = {
        "ANOVA": ("ANOVA", "linear"), "ANOVA-RBF": ("ANOVA", "RBF"),
        "PCA": ("PCA", "linear"), "PCA-RBF": ("PCA", "RBF"),
        "MI": ("MI", "linear"), "MI-RBF": ("MI", "RBF"),
    }
    REGION_SCHEMES = [
        ("Direct\nall", "direct"),
        ("Direct\nneuronal", "direct_neuronal"),
        ("Direct\nnon-neuronal", "direct_non_neuronal"),
        ("Hier. 4→25\nshared", "hier4_all"),
        ("Hier. 4→25\nN/G pooled", "hier4_fraction"),
        ("Hier. 4→25\nneuronal", "hier4_neuronal"),
        ("Hier. 4→25\nnon-neuronal", "hier4_non_neuronal"),
        ("Hier. 7→25\nshared", "hier7_all"),
        ("Hier. 7→25\nN/G pooled", "hier7_fraction"),
        ("Hier. 7→25\nneuronal", "hier7_neuronal"),
        ("Hier. 7→25\nnon-neuronal", "hier7_non_neuronal"),
        ("Joint 4→25\nall", "joint4"),
        ("Joint 4→25\nneuronal", "joint4_neuronal"),
        ("Joint 4→25\nnon-neuronal", "joint4_non_neuronal"),
        ("Joint 7→25\nall", "joint7"),
        ("Joint 7→25\nneuronal", "joint7_neuronal"),
        ("Joint 7→25\nnon-neuronal", "joint7_non_neuronal"),
    ]
    MACRO_SCHEMES = [
        (f"{number} macro\n{label}", scheme, subset)
        for number, scheme in [(4, "macro"), (7, "macro7")]
        for label, subset in [("all", "all_samples"), ("neuronal", "neuronal"),
                              ("non-neuronal", "non_neuronal")]
    ]


    def lodo_scores(folder):
        """Six macro-F1 scores, one for each donor."""
        table = pd.read_csv(RESULTS / folder / "fold_results.csv")
        assert len(table) == 6 and table.test_donor.is_unique
        return table.sort_values("test_donor").macro_f1.to_numpy()


    def hierarchical_scores(path):
        """Recompute one macro-F1 score per donor from saved held-out predictions."""
        predictions = pd.read_csv(path)
        assert len(predictions) == 265
        scores = []
        for _, donor in predictions.groupby("test_donor", sort=True):
            scores.append(f1_score(donor.y_true, donor.y_pred,
                                   labels=np.unique(donor.y_true),
                                   average="macro", zero_division=0))
        assert len(scores) == 6
        return np.array(scores)


    def region_scores(model, scheme):
        if scheme.startswith("joint"):
            if model not in JOINT_METHODS:
                return None
            features, kernel = JOINT_METHODS[model]
            n_macro = scheme[5]
            subset = JOINT_FIXED_FOLDS if "_" not in scheme else JOINT_CELL_FOLDS
            subset = subset[(subset.hierarchy == f"{n_macro} macro")
                            & (subset.features == features)
                            & (subset.kernel == kernel)]
            if "_" in scheme:
                cell_type = "Non-neuronal" if scheme.endswith("_non_neuronal") else "Neuronal"
                subset = subset[subset.cell_type == cell_type]
            subset = subset.sort_values("held_out_donor")
            assert len(subset) == 6 and subset.held_out_donor.is_unique
            return subset.macro_f1.to_numpy()
        if scheme.startswith("direct_"):
            features, kernel = JOINT_METHODS[model]
            cell_type = "Non-neuronal" if scheme.endswith("_non_neuronal") else "Neuronal"
            subset = DIRECT_CELL_FOLDS[(DIRECT_CELL_FOLDS.cell_type == cell_type)
                                       & (DIRECT_CELL_FOLDS.features == features)
                                       & (DIRECT_CELL_FOLDS.kernel == kernel)]
            subset = subset.sort_values("held_out_donor")
            assert len(subset) == 6 and subset.held_out_donor.is_unique
            return subset.macro_f1.to_numpy()
        if scheme.startswith("hier") and (scheme.endswith("_neuronal") or
                                          scheme.endswith("_non_neuronal")):
            macro = "macro" if scheme.startswith("hier4") else "macro7"
            cell_type = "Non-neuronal" if scheme.endswith("_non_neuronal") else "Neuronal"
            if model == "ANOVA" and macro == "macro":
                path = RESULTS / "hierarchical/fraction_specific_predictions.csv"
            else:
                prefix = model.lower().replace("-", "_")
                path = RESULTS / "hierarchical_grid" / f"{prefix}_{macro}_fraction_specific_predictions.csv"
            predictions = pd.read_csv(path)
            predictions = predictions[predictions.fraction == cell_type]
            scores = [f1_score(fold.y_true, fold.y_pred,
                               labels=np.unique(fold.y_true), average="macro", zero_division=0)
                      for _, fold in predictions.groupby("test_donor", sort=True)]
            assert len(scores) == 6
            return np.array(scores)
        if scheme == "direct" and model in ("PCA", "PCA-RBF"):
            # Updated search: PCA component count chosen by 50/70/90/95/99% variance.
            name = "linear" if model == "PCA" else "rbf"
            table = pd.read_csv(RESULTS / "pca_variance" / f"{name}_folds.csv")
            assert len(table) == 6 and table.test_donor.is_unique
            return table.sort_values("test_donor").macro_f1.to_numpy()
        if scheme == "direct":
            return lodo_scores(f"{MODEL_STYLE[model][3]}_25_regions_nested")

        macro = "macro" if scheme.startswith("hier4") else "macro7"
        variant = "all_samples" if scheme.endswith("all") else "fraction_specific"
        if model == "ANOVA" and macro == "macro":
            path = RESULTS / "hierarchical" / f"{variant}_predictions.csv"
        else:
            prefix = model.lower().replace("-", "_")
            path = RESULTS / "hierarchical_grid" / f"{prefix}_{macro}_{variant}_predictions.csv"
        return hierarchical_scores(path)


    def macro_scores(model, scheme, subset):
        prefix = MODEL_STYLE[model][2]
        return lodo_scores(f"{prefix}{scheme}_{subset}_nested")


    def draw_panel(ax, panel, models):
        schemes = REGION_SCHEMES if panel == "region" else MACRO_SCHEMES
        width = min(0.74 / len(models), 0.25)
        step = 0.78 / len(models)
        jitter = np.random.default_rng(0)
        for x, entry in enumerate(schemes):
            label = entry[0]
            for j, model in enumerate(models):
                values = (region_scores(model, entry[1]) if panel == "region"
                          else macro_scores(model, entry[1], entry[2]))
                if values is None:
                    continue
                color, hatch, _, _ = MODEL_STYLE[model]
                # Offset each method so all six joint-model boxes remain visible.
                pos = x + (j - (len(models) - 1) / 2) * step
                ax.boxplot(values, positions=[pos], widths=width, patch_artist=True,
                           showfliers=False,
                           boxprops=dict(facecolor=(*plt.matplotlib.colors.to_rgb(color),
                                                     0.15 if hatch else 0.35),
                                         edgecolor=color, hatch=hatch),
                           medianprops=dict(color=color, linewidth=2),
                           whiskerprops=dict(color=color), capprops=dict(color=color))
                ax.scatter(pos + jitter.uniform(-0.025, 0.025, len(values)), values,
                           s=11, color=color, edgecolor="white", linewidth=0.4, zorder=3)
        if panel == "region":
            ax.axvline(2.5, color="0.55", linestyle="--", linewidth=1)
            ax.axvline(10.5, color="0.55", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(schemes)), [entry[0] for entry in schemes])
        ax.set_xlim(-0.5, len(schemes) - 0.5)
        ax.grid(axis="y", color="0.9")
        ax.set_axisbelow(True)
        ax.set_title("Scored on the 25 regions" if panel == "region"
                     else "Scored on the macro-region classes", fontweight="bold")


    def plot_hakohige(models, filename):
        fig, axes = plt.subplots(1, 2, figsize=(40 if len(models) == 6 else 32, 7),
                                 sharey=True, layout="constrained",
                                 gridspec_kw={"width_ratios": [19, 6]})
        draw_panel(axes[0], "region", models)
        draw_panel(axes[1], "macro", models)
        axes[0].set(ylim=(0, 1.03), ylabel="Macro F1 (one dot = one held-out donor)")
        legend = [Patch(facecolor=(*plt.matplotlib.colors.to_rgb(MODEL_STYLE[m][0]),
                                   0.15 if MODEL_STYLE[m][1] else 0.35),
                        edgecolor=MODEL_STYLE[m][0], hatch=MODEL_STYLE[m][1], label=m)
                  for m in models]
        fig.legend(handles=legend, title="Feature method / SVM kernel",
                   loc="outside upper right", ncol=len(models), frameon=False)
        fig.suptitle("Donor-held-out performance: training scheme × model", fontsize=15,
                     fontweight="bold")
        # Place the methods note below the category labels; bbox_inches='tight'
        # includes it in the saved PNG.
        fig.text(0.5, -0.075,
                 "Direct, separate hierarchy and macro: nested-tuned LODO (PCA direct: variance targets).\n"
                 "Direct N/G and joint N/G: fixed 1,000 ANOVA/MI genes or PCA 95% variance, C=1. All boxes: 6 donors.",
                 ha="center", fontsize=9, color="#444444")
        fig.savefig(RESULTS / filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return fig



    _ = plot_hakohige(["ANOVA", "MI", "PCA"], "full_grid_boxplot.png")

    # The same figure with both linear and RBF SVM results.
    # PCA direct uses variance-target tuning; other PCA schemes use fixed PC counts.
    # Joint ANOVA, MI and PCA (linear/RBF) now include all, neuronal and non-neuronal runs.
    _ = plot_hakohige(list(MODEL_STYLE), "full_grid_boxplot_all_models.png")
    return [RESULTS / "full_grid_boxplot.png",
            RESULTS / "full_grid_boxplot_all_models.png"]


def create_confusion_matrices(project_dir, results_dir):
    """Save best direct, separate, joint, and macro confusion matrices."""
    _enter_project(project_dir)

    # Confusion matrices

    """Best direct 25-region classifier: held-out confusion matrix."""
    from pathlib import Path
    RESULTS = Path(results_dir)
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from sklearn.metrics import confusion_matrix

    groups_7 = {
        "Prefrontal": ["ACC", "DLPFC", "OFC", "VLPFC"],
        "Sensorimotor": ["PAC", "PMC", "PSC", "PVC"],
        "Parieto-temporal": ["INS", "ITC", "PSTC", "PVPC"],
        "Limbic": ["AMY", "EC", "HIPP"],
        "Basal Ganglia": ["CN", "GP", "NAC"],
        "Midbrain/Diencephalon": ["ARC", "DRN", "HAB", "MDT", "RMTG", "VTA"],
        "Hindbrain": ["CRBLM"],
    }
    groups_4 = {
        "Forebrain": (0, 15, "#EB6834"),
        "Basal Ganglia": (15, 18, "#1BAF7A"),
        "Midbrain/Diencephalon": (18, 24, "#EDA100"),
        "Hindbrain": (24, 25, "#E87BA4"),
    }
    regions = [region for members in groups_7.values() for region in members]
    predictions = pd.read_csv(RESULTS / "rbf_anova_25_regions_nested/predictions.csv")
    assert len(predictions) == predictions.sample_index.nunique() == 265
    assert set(predictions.y_true) == set(regions)
    matrix = confusion_matrix(predictions.y_true, predictions.y_pred, labels=regions)
    row_fraction = matrix / matrix.sum(axis=1, keepdims=True)

    cmap = ListedColormap(plt.cm.Blues(np.linspace(.3, 1, 256)))
    cmap.set_bad("#E3E3E3")
    fig, ax = plt.subplots(figsize=(12, 11))
    fig.subplots_adjust(left=.10, right=.84, top=.90, bottom=.17)
    image = ax.imshow(np.ma.masked_where(matrix == 0, row_fraction),
                      cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(26) - .5, minor=True)
    ax.set_yticks(np.arange(26) - .5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)

    start = 0
    for members in list(groups_7.values())[:4]:
        ax.add_patch(Rectangle((start - .5, start - .5), len(members), len(members),
                               fill=False, edgecolor="#333333", linewidth=1.5,
                               linestyle="--", zorder=9))
        start += len(members)
    for a, b, color in groups_4.values():
        ax.add_patch(Rectangle((a - .5, a - .5), b - a, b - a,
                               fill=False, edgecolor=color, linewidth=3,
                               clip_on=False, zorder=10))

    ax.set_xticks(range(25), regions, rotation=90)
    ax.set_yticks(range(25), regions)
    ax.set(xlabel="Predicted region", ylabel="True region")
    ax.set_title(f"Direct ANOVA + RBF SVM: {np.trace(matrix)}/{matrix.sum()} correct",
                 fontweight="bold", pad=16)
    plt.colorbar(image, cax=fig.add_axes([.87, .27, .018, .51]),
                 label="Fraction of true region (grey = 0)")
    fig.legend(handles=[Patch(facecolor="none", edgecolor=color, linewidth=3, label=name)
                        for name, (_, _, color) in groups_4.items()]
               + [Line2D([], [], color="#333333", linewidth=1.5, linestyle="--",
                         label="Forebrain subgroup (7-macro split)")],
               loc="lower center", bbox_to_anchor=(.5, .02), ncol=3, frameon=False)
    output = RESULTS / "rbf_anova_25_regions_nested/confusion_matrix_25.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ------------------------------------------------------------
    # 25 × 25 confusion heatmaps of the two hierarchical models (ANOVA, shared model on all samples):
    # 4 macro → 25 regions vs 7 macro → 25 regions. Loaded from the saved predictions (no retraining).
    # ------------------------------------------------------------
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle, Patch
    from sklearn.metrics import confusion_matrix

    HIER_RUNS = {"Hierarchical 4 macro → 25 regions": RESULTS / "hierarchical" / "all_samples_predictions.csv",
                 "Hierarchical 7 macro → 25 regions": RESULTS / "hierarchical_grid" / "anova_macro7_all_samples_predictions.csv"}

    # Regions ordered by the 7 groups; this order is also grouped by the 4 macro regions
    GROUPS_7 = {"Prefrontal": ["ACC", "DLPFC", "OFC", "VLPFC"], "Sensorimotor": ["PAC", "PMC", "PSC", "PVC"],
                "Parieto-temporal": ["INS", "ITC", "PSTC", "PVPC"], "Limbic": ["AMY", "EC", "HIPP"],
                "Basal Ganglia": ["CN", "GP", "NAC"], "Midbrain/Diencephalon": ["ARC", "DRN", "HAB", "MDT", "RMTG", "VTA"],
                "Hindbrain": ["CRBLM"]}
    GROUPS_4 = {"Forebrain": (0, 15, "#EB6834"), "Basal Ganglia": (15, 18, "#1BAF7A"),
                "Midbrain/Diencephalon": (18, 24, "#EDA100"), "Hindbrain": (24, 25, "#E87BA4")}
    ordered = [r for regions in GROUPS_7.values() for r in regions]
    n = len(ordered)
    heat_cmap = ListedColormap(plt.cm.Blues(np.linspace(0.3, 1, 256)))   # 1 sample is still visible
    heat_cmap.set_bad("#E3E3E3")                                         # grey = 0

    fig, axes = plt.subplots(1, 2, figsize=(24, 12.5), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.92, top=0.88, bottom=0.14, wspace=0.06)
    for ax, (title, path) in zip(axes, HIER_RUNS.items()):
        preds = pd.read_csv(path)
        assert set(preds["y_true"]) == set(ordered)
        cm = confusion_matrix(preds["y_true"], preds["y_pred"], labels=ordered)
        row_frac = cm / cm.sum(axis=1, keepdims=True)                   # color = fraction of the true region
        image = ax.imshow(np.ma.masked_where(cm == 0, row_frac), cmap=heat_cmap, vmin=0, vmax=1)
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", length=0)

        start = 0                                                       # thin dashed boxes: forebrain subgroups
        for name, regions in list(GROUPS_7.items())[:4]:
            ax.add_patch(Rectangle((start - 0.5, start - 0.5), len(regions), len(regions), fill=False,
                                   edgecolor="#333333", linewidth=1.5, linestyle="--", zorder=9))
            start += len(regions)
        for a, b, color in GROUPS_4.values():                           # bold boxes: 4 macro regions
            ax.add_patch(Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False,
                                   edgecolor=color, linewidth=3, clip_on=False, zorder=10))

        ax.set_xticks(range(n), ordered, rotation=90)
        ax.set_yticks(range(n), ordered)
        ax.tick_params(axis="y", labelleft=ax is axes[0])
        ax.set(xlabel="Predicted region", ylabel="True region" if ax is axes[0] else "")
        routed = (preds["macro_true"] == preds["macro_pred"]).mean()
        ax.set_title(f"{title}\n{np.trace(cm)}/{cm.sum()} correct | first stage correct for {routed:.1%} of samples",
                     fontweight="bold")

    plt.colorbar(image, cax=fig.add_axes([0.93, 0.25, 0.01, 0.5]), label="Fraction of true region (grey = 0)")
    fig.legend(handles=[Patch(facecolor="none", edgecolor=c, linewidth=3, label=name) for name, (_, _, c) in GROUPS_4.items()]
                       + [Line2D([], [], color="#333333", linewidth=1.5, linestyle="--",
                                 label="Forebrain subgroup (7-macro split)")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False)
    fig.suptitle("ANOVA + LinearSVC: hierarchical models, all samples (nested LODO)", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(RESULTS / "hierarchical_grid" / "hierarchical_4_vs_7_confusion_25.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    """Match the final hierarchical confusion-matrix cell's visual style."""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from sklearn.metrics import confusion_matrix


    RUNS = {
        "Joint 4 macro → 25 regions": RESULTS / "joint4_anova_25_regions_nested/predictions.csv",
        "Joint 7 macro → 25 regions": RESULTS / "joint7_anova_25_regions_nested/predictions.csv",
    }

    # The same region order and outlines as the hierarchical 4-vs-7 figure.
    GROUPS_7 = {
        "Prefrontal": ["ACC", "DLPFC", "OFC", "VLPFC"],
        "Sensorimotor": ["PAC", "PMC", "PSC", "PVC"],
        "Parieto-temporal": ["INS", "ITC", "PSTC", "PVPC"],
        "Limbic": ["AMY", "EC", "HIPP"],
        "Basal Ganglia": ["CN", "GP", "NAC"],
        "Midbrain/Diencephalon": ["ARC", "DRN", "HAB", "MDT", "RMTG", "VTA"],
        "Hindbrain": ["CRBLM"],
    }
    GROUPS_4 = {
        "Forebrain": (0, 15, "#EB6834"),
        "Basal Ganglia": (15, 18, "#1BAF7A"),
        "Midbrain/Diencephalon": (18, 24, "#EDA100"),
        "Hindbrain": (24, 25, "#E87BA4"),
    }
    ordered = [region for regions in GROUPS_7.values() for region in regions]
    n = len(ordered)
    heat_cmap = ListedColormap(plt.cm.Blues(np.linspace(0.3, 1, 256)))
    heat_cmap.set_bad("#E3E3E3")  # grey = no predictions in this cell

    fig, axes = plt.subplots(1, 2, figsize=(24, 12.5), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.92, top=0.88, bottom=0.14, wspace=0.06)
    for ax, (title, path) in zip(axes, RUNS.items()):
        preds = pd.read_csv(path)
        assert len(preds) == 265 and set(preds["y_true"]) == set(ordered)
        cm = confusion_matrix(preds["y_true"], preds["y_pred"], labels=ordered)
        row_frac = cm / cm.sum(axis=1, keepdims=True)
        image = ax.imshow(np.ma.masked_where(cm == 0, row_frac), cmap=heat_cmap, vmin=0, vmax=1)
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", length=0)

        # Dashed forebrain subgroups represent the 7-macro split.
        start = 0
        for regions in list(GROUPS_7.values())[:4]:
            ax.add_patch(Rectangle((start - 0.5, start - 0.5), len(regions), len(regions),
                                   fill=False, edgecolor="#333333", linewidth=1.5,
                                   linestyle="--", zorder=9))
            start += len(regions)
        for a, b, color in GROUPS_4.values():
            ax.add_patch(Rectangle((a - 0.5, a - 0.5), b - a, b - a,
                                   fill=False, edgecolor=color, linewidth=3,
                                   clip_on=False, zorder=10))

        ax.set_xticks(range(n), ordered, rotation=90)
        ax.set_yticks(range(n), ordered)
        ax.tick_params(axis="y", labelleft=ax is axes[0])
        ax.set(xlabel="Predicted region", ylabel="True region" if ax is axes[0] else "")
        ax.set_title(f"{title}\n{np.trace(cm)}/{cm.sum()} correct", fontweight="bold")

    plt.colorbar(image, cax=fig.add_axes([0.93, 0.25, 0.01, 0.5]),
                 label="Fraction of true region (grey = 0)")
    fig.legend(
        handles=[Patch(facecolor="none", edgecolor=color, linewidth=3, label=name)
                 for name, (_, _, color) in GROUPS_4.items()]
                + [Line2D([], [], color="#333333", linewidth=1.5, linestyle="--",
                          label="Forebrain subgroup (7-macro split)")],
        loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False,
    )
    fig.suptitle("ANOVA + LinearSVC: jointly trained models, all samples (nested LODO)",
                 fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(RESULTS / "joint_4_vs_7_confusion_25.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    """Plot best macro classifiers and summarize oracle-stage hierarchical runs."""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    RESULTS = Path(results_dir)
    summary = pd.read_csv(RESULTS / "full_grid_summary.csv")

    MACRO_RUNS = {
        "4 macro": {
            "folder": "rbf_mi_macro_all_samples_nested",
            "labels": ["Forebrain", "Basal Ganglia", "Midbrain/Diencephalon", "Hindbrain"],
            "output": "best_macro_4x4_confusion.png",
        },
        "7 macro": {
            "folder": "rbf_mi_macro7_all_samples_nested",
            "labels": ["Prefrontal_Cortex", "Sensorimotor_Cortex", "Parieto_Temporal_Assoc",
                       "Limbic_Allocortex", "Basal_Ganglia", "Midbrain_Diencephalon",
                       "Cerebellum_Hindbrain"],
            "output": "best_macro_7x7_confusion.png",
        },
    }

    # The seven labels are ordered so the first four form the broad Forebrain group.
    MACRO_4_BLOCKS = {
        "Forebrain": (0, 4, "#EB6834"),
        "Basal Ganglia": (4, 5, "#1BAF7A"),
        "Midbrain/Diencephalon": (5, 6, "#EDA100"),
        "Hindbrain": (6, 7, "#E87BA4"),
    }
    heat_cmap = ListedColormap(plt.cm.Blues(np.linspace(0.3, 1, 256)))
    heat_cmap.set_bad("#E3E3E3")  # Grey means no samples in that cell.


    def plot_macro_confusion(scheme, config):
        best = summary.query("scheme == @scheme and variant == 'all_samples'") \
                      .sort_values("macro_f1_mean", ascending=False).iloc[0]
        assert best.method == "MI-RBF", f"The saved best model changed for {scheme}: {best.method}"
        predictions = pd.read_csv(RESULTS / config["folder"] / "predictions.csv")
        labels = config["labels"]
        assert len(predictions) == 265 and set(predictions.y_true) == set(labels)
        cm = confusion_matrix(predictions.y_true, predictions.y_pred, labels=labels)
        fractions = cm / cm.sum(axis=1, keepdims=True)

        n = len(labels)
        fig, ax = plt.subplots(figsize=(8, 7) if n == 4 else (9.5, 9))
        if n == 7:
            fig.subplots_adjust(left=0.22, right=0.86, top=0.84, bottom=0.27)
        else:
            fig.subplots_adjust(left=0.23, right=0.85, top=0.86, bottom=0.23)
        image = ax.imshow(np.ma.masked_where(cm == 0, fractions),
                          cmap=heat_cmap, vmin=0, vmax=1)
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        if n == 7:
            for start, stop, color in MACRO_4_BLOCKS.values():
                ax.add_patch(Rectangle((start - 0.5, start - 0.5), stop - start,
                                       stop - start, fill=False, edgecolor=color,
                                       linewidth=3.5, zorder=10, clip_on=False))
        names = [label.replace("_", " ") for label in labels]
        ax.set_xticks(range(n), names, rotation=45, ha="right")
        ax.set_yticks(range(n), names)
        ax.set_xlabel("Predicted macro region")
        ax.set_ylabel("True macro region")
        accuracy = np.trace(cm) / cm.sum()
        within = sum(cm[start:stop, start:stop].sum()
                     for start, stop, _ in MACRO_4_BLOCKS.values()) if n == 7 else None
        detail = (f" | {within}/{cm.sum()} within the right 4-macro region"
                  if n == 7 else "")
        ax.set_title(f"Best {scheme} classifier: MI + RBF SVM (nested LODO)\n"
                     f"{np.trace(cm)}/{cm.sum()} correct ({accuracy:.1%}){detail}",
                     fontweight="bold")
        fig.colorbar(image, cax=fig.add_axes([0.88, 0.32, 0.025, 0.5]),
                     label="Fraction of true macro region (grey = 0)")
        if n == 7:
            fig.legend(handles=[Patch(facecolor="none", edgecolor=color, linewidth=3,
                                      label=name)
                                for name, (_, _, color) in MACRO_4_BLOCKS.items()],
                       loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4,
                       frameon=False, title="4-macro region", handlelength=1.5,
                       columnspacing=1.2)
        path = RESULTS / config["output"]
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path


    macro_plot_paths = [plot_macro_confusion(scheme, config)
                        for scheme, config in MACRO_RUNS.items()]
    return [
        RESULTS / "rbf_anova_25_regions_nested/confusion_matrix_25.png",
        RESULTS / "hierarchical_grid/hierarchical_4_vs_7_confusion_25.png",
        RESULTS / "joint_4_vs_7_confusion_25.png",
        *macro_plot_paths,
    ]
