"""Cell-specific direct and joint training, and cross-family summaries.

Joint and direct fits checkpoint each held-out donor. The summary combines
their fixed-setting scores with saved nested macro and two-stage scores.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC

from joint_feature_comparison import (DATA_FILE, OUT_DIR, MACRO_4, MACRO_7,
                                      JointHierSVM, prepare_features, rbf_features)

JOINT_CHECKPOINTS = OUT_DIR / "cell_type_checkpoints"
DIRECT_CHECKPOINTS = OUT_DIR / "direct_cell_type_checkpoints"
JOINT_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
DIRECT_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
JOINT_METHODS = ("ANOVA", "PCA", "MI")
RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "joint_feature_comparison"


def configure_paths(data_file, results_dir):
    """Use the notebook's dataset and result folders for every cell-type run."""
    global DATA_FILE, OUT_DIR, JOINT_CHECKPOINTS, DIRECT_CHECKPOINTS, RESULTS, OUT
    DATA_FILE = Path(data_file)
    RESULTS = Path(results_dir)
    OUT_DIR = RESULTS / "joint_feature_comparison"
    OUT = OUT_DIR
    JOINT_CHECKPOINTS = OUT_DIR / "cell_type_checkpoints"
    DIRECT_CHECKPOINTS = OUT_DIR / "direct_cell_type_checkpoints"
    JOINT_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    DIRECT_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
FAMILY_METHODS = {
    "ANOVA": ("", "ANOVA", "linear"),
    "ANOVA-RBF": ("rbf_", "ANOVA", "RBF"),
    "MI": ("mi_", "MI", "linear"),
    "MI-RBF": ("rbf_mi_", "MI", "RBF"),
    "PCA": ("pca_", "PCA", "linear"),
    "PCA-RBF": ("rbf_pca_", "PCA", "RBF"),
}


def _run_joint_donor(x, y, global_indices, train, test, cell_type, donor):
    score_path = JOINT_CHECKPOINTS / f"{cell_type}_{donor}_scores.csv"
    pred_path = JOINT_CHECKPOINTS / f"{cell_type}_{donor}_predictions.csv"
    if score_path.exists() and pred_path.exists():
        scores = pd.read_csv(score_path)
        predictions = pd.read_csv(pred_path)
        assert len(scores) == 12 and len(predictions) == 12 * len(test)
        print(f"Loaded {cell_type}, donor {donor}", flush=True)
        return scores, predictions

    features, pca_variance = prepare_features(x[train], y[train], x[test], JOINT_METHODS)
    rows, predictions = [], []
    for feature_name, (x_train, x_test) in features.items():
        for kernel in ("linear", "RBF"):
            fit_x, test_x = (rbf_features(x_train, x_test)
                             if kernel == "RBF" else (x_train, x_test))
            for n_macro, mapping in ((4, MACRO_4), (7, MACRO_7)):
                model = JointHierSVM(mapping).fit(fit_x, y[train])
                predicted = model.predict(test_x)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                    macro_f1 = f1_score(y[test], predicted, labels=np.unique(y[test]),
                                        average="macro", zero_division=0)
                    balanced_accuracy = balanced_accuracy_score(y[test], predicted)
                rows.append({"cell_type": cell_type, "held_out_donor": donor,
                             "hierarchy": f"{n_macro} macro", "features": feature_name,
                             "kernel": kernel, "macro_f1": macro_f1,
                             "balanced_accuracy": balanced_accuracy,
                             "n_features": x_train.shape[1],
                             "pca_variance": pca_variance if feature_name == "PCA" else np.nan,
                             "optimizer_iterations": model.n_iter_,
                             "optimizer_converged": model.converged_})
                predictions.extend({"cell_type": cell_type, "held_out_donor": donor,
                                    "hierarchy": f"{n_macro} macro", "features": feature_name,
                                    "kernel": kernel, "sample_index": int(i),
                                    "y_true": truth, "y_pred": pred}
                                   for i, truth, pred in zip(global_indices[test], y[test], predicted))
                print(f"  {cell_type} {donor}: {n_macro} macro, {feature_name} + {kernel}, "
                      f"F1={macro_f1:.3f}, BA={balanced_accuracy:.3f}", flush=True)
    scores = pd.DataFrame(rows)
    predictions = pd.DataFrame(predictions)
    scores.to_csv(score_path, index=False)
    predictions.to_csv(pred_path, index=False)
    return scores, predictions


def run_joint_cell_types():
    data = pd.read_csv(DATA_FILE, index_col=0)
    x = data.to_numpy(dtype=np.float64).T
    parsed = [re.fullmatch(r"(BB\d+)_(.+)_([GN])_RNA", name).groups()
              for name in data.columns]
    donors = np.array([row[0] for row in parsed])
    y = np.array([row[1] for row in parsed])
    fractions = np.array([row[2] for row in parsed])
    all_scores, all_predictions = [], []
    for cell_type, code in (("Neuronal", "N"), ("Non-neuronal", "G")):
        indices = np.flatnonzero(fractions == code)
        x_part, y_part, donors_part = x[indices], y[indices], donors[indices]
        for train, test in LeaveOneGroupOut().split(x_part, y_part, donors_part):
            donor = donors_part[test][0]
            print(f"Running {cell_type}, held-out donor {donor}", flush=True)
            scores, predictions = _run_joint_donor(x_part, y_part, indices, train, test,
                                            cell_type, donor)
            all_scores.append(scores)
            all_predictions.append(predictions)

    scores = pd.concat(all_scores, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    assert len(scores) == 2 * 6 * 3 * 2 * 2
    assert len(predictions) == 12 * len(data.columns)
    scores.to_csv(OUT_DIR / "cell_type_fold_scores.csv", index=False)
    predictions.to_csv(OUT_DIR / "cell_type_predictions.csv", index=False)
    summary = (scores.groupby(["cell_type", "hierarchy", "features", "kernel"], sort=False)
               .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                    balanced_accuracy_mean=("balanced_accuracy", "mean"),
                    balanced_accuracy_sd=("balanced_accuracy", "std"),
                    n_donors=("held_out_donor", "nunique"))
               .reset_index())
    summary.to_csv(OUT_DIR / "cell_type_summary.csv", index=False)
    print("\n", summary.round(3).to_string(index=False), flush=True)




def _run_direct_donor(x, y, indices, train, test, cell_type, donor):
    score_path = DIRECT_CHECKPOINTS / f"{cell_type}_{donor}_scores.csv"
    pred_path = DIRECT_CHECKPOINTS / f"{cell_type}_{donor}_predictions.csv"
    if score_path.exists() and pred_path.exists():
        scores, predictions = pd.read_csv(score_path), pd.read_csv(pred_path)
        if len(scores) == 6 and len(predictions) == 6 * len(test):
            print(f"Loaded {cell_type}, donor {donor}", flush=True)
            return scores, predictions

    features, pca_variance = prepare_features(x[train], y[train], x[test],
                                               ("ANOVA", "MI", "PCA"))
    rows, predictions = [], []
    for feature_name, (x_train, x_test) in features.items():
        for kernel in ("linear", "RBF"):
            # For RBF, sklearn's 'scale' is 1 / (n_features * variance).
            model = SVC(kernel="linear" if kernel == "linear" else "rbf",
                        C=1, gamma="scale", class_weight="balanced")
            model.fit(x_train, y[train])
            predicted = model.predict(x_test)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                macro_f1 = f1_score(y[test], predicted, labels=np.unique(y[test]),
                                    average="macro", zero_division=0)
                ba = balanced_accuracy_score(y[test], predicted)
            rows.append({"cell_type": cell_type, "held_out_donor": donor,
                         "features": feature_name, "kernel": kernel,
                         "macro_f1": macro_f1, "balanced_accuracy": ba,
                         "n_features": x_train.shape[1], "pca_variance": pca_variance})
            predictions.extend({"cell_type": cell_type, "held_out_donor": donor,
                                "features": feature_name, "kernel": kernel,
                                "sample_index": int(index), "y_true": truth, "y_pred": pred}
                               for index, truth, pred in zip(indices[test], y[test], predicted))
            print(f"  {cell_type} {donor}: {feature_name} + {kernel}, "
                  f"F1={macro_f1:.3f}, BA={ba:.3f}", flush=True)
    score_table = pd.DataFrame(rows)
    pred_table = pd.DataFrame(predictions)
    score_table.to_csv(score_path, index=False)
    pred_table.to_csv(pred_path, index=False)
    return score_table, pred_table


def run_direct_cell_types():
    data = pd.read_csv(DATA_FILE, index_col=0)
    x_all = data.to_numpy(dtype=np.float64).T
    parsed = [re.fullmatch(r"(BB\d+)_(.+)_([GN])_RNA", name).groups()
              for name in data.columns]
    y_all = np.array([row[1] for row in parsed])
    donors_all = np.array([row[0] for row in parsed])
    fractions = np.array([row[2] for row in parsed])
    scores, predictions = [], []
    for cell_type, code in (("Neuronal", "N"), ("Non-neuronal", "G")):
        indices = np.flatnonzero(fractions == code)
        x, y, donors = x_all[indices], y_all[indices], donors_all[indices]
        for train, test in LeaveOneGroupOut().split(x, y, donors):
            donor = donors[test][0]
            print(f"Running direct {cell_type}, held-out donor {donor}", flush=True)
            fold_scores, fold_preds = _run_direct_donor(x, y, indices, train, test,
                                               cell_type, donor)
            scores.append(fold_scores)
            predictions.append(fold_preds)
    score_table = pd.concat(scores, ignore_index=True)
    pred_table = pd.concat(predictions, ignore_index=True)
    assert len(score_table) == 72 and len(pred_table) == 6 * len(data.columns)
    score_table.to_csv(OUT_DIR / "direct_cell_type_fold_scores.csv", index=False)
    pred_table.to_csv(OUT_DIR / "direct_cell_type_predictions.csv", index=False)
    summary = (score_table.groupby(["cell_type", "features", "kernel"], sort=False)
               .agg(macro_f1_mean=("macro_f1", "mean"),
                    macro_f1_sd=("macro_f1", "std"),
                    balanced_accuracy_mean=("balanced_accuracy", "mean"),
                    balanced_accuracy_sd=("balanced_accuracy", "std"),
                    n_donors=("held_out_donor", "nunique"))
               .reset_index())
    summary.to_csv(OUT_DIR / "direct_cell_type_summary.csv", index=False)
    print(summary.round(3).to_string(index=False), flush=True)




def add_saved_predictions(rows, method, family, cell_type, path):
    predictions = pd.read_csv(path)
    subset = predictions[predictions.fraction == cell_type]
    for donor, fold in subset.groupby("test_donor", sort=True):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            f1 = f1_score(fold.y_true, fold.y_pred,
                          labels=np.unique(fold.y_true), average="macro", zero_division=0)
            ba = balanced_accuracy_score(fold.y_true, fold.y_pred)
        rows.append({"cell_type": cell_type, "model": method, "family": family,
                     "held_out_donor": donor, "macro_f1": f1, "balanced_accuracy": ba,
                     "training": "cell-specific, nested tuned"})


def summarize_cell_type_families():
    rows = []
    direct = pd.read_csv(OUT / "direct_cell_type_fold_scores.csv")
    joint = pd.read_csv(OUT / "cell_type_fold_scores.csv")
    for cell_type, suffix in (("Neuronal", "neuronal"),
                              ("Non-neuronal", "non_neuronal")):
        for method, (prefix, feature, kernel) in FAMILY_METHODS.items():
            direct_fold = direct[(direct.cell_type == cell_type)
                                 & (direct.features == feature) & (direct.kernel == kernel)]
            for row in direct_fold.itertuples():
                rows.append({"cell_type": cell_type, "model": method,
                             "family": "Direct 25", "held_out_donor": row.held_out_donor,
                             "macro_f1": row.macro_f1,
                             "balanced_accuracy": row.balanced_accuracy,
                             "training": "cell-specific, fixed setting"})
            for number, macro in ((4, "macro"), (7, "macro7")):
                macro_folder = RESULTS / f"{prefix}{macro}_{suffix}_nested"
                macro_scores = pd.read_csv(macro_folder / "fold_results.csv")
                for row in macro_scores.itertuples():
                    rows.append({"cell_type": cell_type, "model": method,
                                 "family": f"Macro {number}", "held_out_donor": row.test_donor,
                                 "macro_f1": row.macro_f1,
                                 "balanced_accuracy": row.balanced_accuracy,
                                 "training": "cell-specific, nested tuned"})
                if method == "ANOVA" and number == 4:
                    path = RESULTS / "hierarchical/fraction_specific_predictions.csv"
                else:
                    path = (RESULTS / "hierarchical_grid" /
                            f"{method.lower().replace('-', '_')}_{macro}_fraction_specific_predictions.csv")
                add_saved_predictions(rows, method, f"Separate {number}→25",
                                      cell_type, path)
                joint_fold = joint[(joint.cell_type == cell_type)
                                   & (joint.hierarchy == f"{number} macro")
                                   & (joint.features == feature)
                                   & (joint.kernel == kernel)]
                for row in joint_fold.itertuples():
                    rows.append({"cell_type": cell_type, "model": method,
                                 "family": f"Joint {number}→25",
                                 "held_out_donor": row.held_out_donor,
                                 "macro_f1": row.macro_f1,
                                 "balanced_accuracy": row.balanced_accuracy,
                                 "training": "cell-specific, fixed setting"})
    folds = pd.DataFrame(rows)
    counts = folds.groupby(["cell_type", "model", "family"]).held_out_donor.agg(
        ["size", "nunique"])
    assert len(counts) == 84 and (counts["size"] == 6).all()
    assert (counts["nunique"] == 6).all()
    folds.to_csv(OUT / "all_cell_type_family_folds.csv", index=False)
    summary = (folds.groupby(["cell_type", "model", "family", "training"], sort=False)
               .agg(macro_f1_mean=("macro_f1", "mean"),
                    macro_f1_sd=("macro_f1", "std"),
                    balanced_accuracy_mean=("balanced_accuracy", "mean"),
                    balanced_accuracy_sd=("balanced_accuracy", "std"))
               .reset_index())
    summary.to_csv(OUT / "all_cell_type_family_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
