"""Quick, fixed-setting comparison of joint hierarchical 25-region classifiers.

Run from this directory. All preprocessing is fitted within each outer training fold.
The RBF model uses an exact finite training-set feature map for the RBF kernel,
so the same shared-macro plus region-specific loss can be optimized in an RKHS.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(os.environ.get("BRAIN_REGION_DATA_FILE", PROJECT_DIR / "GSE211792_logCPM_cent.csv.gz")).expanduser()
OUT_DIR = PROJECT_DIR / "results" / "joint_feature_comparison"
MI_CACHE_DIR = PROJECT_DIR / "resume_checkpoints" / "mi_scores"


def configure_paths(data_file, results_dir, checkpoint_dir):
    """Use the paths selected in the analysis notebook."""
    global DATA_FILE, OUT_DIR, MI_CACHE_DIR
    DATA_FILE = Path(data_file)
    OUT_DIR = Path(results_dir) / "joint_feature_comparison"
    MI_CACHE_DIR = Path(checkpoint_dir) / "mi_scores"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed exploratory settings: identical for every outer donor and each hierarchy.
N_GENES = 1000
PCA_VARIANCE_TARGET = 0.95
PCA_MAX_COMPONENTS = 210
C = 1.0
RANDOM_STATE = 42

MACRO_4 = {
    **dict.fromkeys(["ACC", "DLPFC", "EC", "INS", "ITC", "OFC", "PAC", "PMC", "PSC",
                     "PSTC", "PVPC", "VLPFC", "PVC", "AMY", "HIPP"], "Forebrain"),
    **dict.fromkeys(["CN", "NAC", "GP"], "Basal Ganglia"),
    **dict.fromkeys(["RMTG", "VTA", "DRN", "HAB", "MDT", "ARC"], "Midbrain/Diencephalon"),
    "CRBLM": "Hindbrain",
}
MACRO_7 = {
    **dict.fromkeys(["DLPFC", "VLPFC", "OFC", "ACC"], "Prefrontal_Cortex"),
    **dict.fromkeys(["PVC", "PAC", "PSC", "PMC"], "Sensorimotor_Cortex"),
    **dict.fromkeys(["ITC", "PSTC", "PVPC", "INS"], "Parieto_Temporal_Assoc"),
    **dict.fromkeys(["AMY", "HIPP", "EC"], "Limbic_Allocortex"),
    **dict.fromkeys(["CN", "NAC", "GP"], "Basal_Ganglia"),
    **dict.fromkeys(["VTA", "DRN", "RMTG", "HAB", "MDT", "ARC"], "Midbrain_Diencephalon"),
    "CRBLM": "Cerebellum_Hindbrain",
}


class JointHierSVM:
    """Same squared-hinge joint objective as the earlier ANOVA linear runs."""

    def __init__(self, region_to_group, c=C, max_iter=5000):
        self.region_to_group = region_to_group
        self.c = c
        self.max_iter = max_iter

    def fit(self, x, y):
        # Orthogonal projection preserves scores and the minimum weight norm.
        if x.shape[1] > x.shape[0]:
            _, _, vt = np.linalg.svd(x, full_matrices=False)
            self.projection_ = vt.T
            x = x @ self.projection_
        else:
            self.projection_ = None
        self.classes_ = np.unique(y)
        n, d = x.shape
        r = len(self.classes_)
        groups = sorted({self.region_to_group[label] for label in self.classes_})
        g = len(groups)
        group_of = np.array([groups.index(self.region_to_group[label]) for label in self.classes_])
        signs = np.where(y[:, None] == self.classes_[None, :], 1.0, -1.0)
        counts = pd.Series(y).value_counts()
        weights = self.c * np.array([n / (r * counts[label]) for label in y])[:, None]

        def unpack(theta):
            group_w = theta[:g * d].reshape(g, d)
            region_w = theta[g * d:(g + r) * d].reshape(r, d)
            return group_w, region_w, theta[(g + r) * d:]

        def objective(theta):
            group_w, region_w, bias = unpack(theta)
            combined_w = region_w + group_w[group_of]
            slack = np.maximum(0, 1 - signs * (x @ combined_w.T + bias))
            loss = 0.5 * (np.sum(group_w ** 2) + np.sum(region_w ** 2)) + np.sum(weights * slack ** 2)
            derivative = -2 * weights * signs * slack
            region_gradient = derivative.T @ x
            group_gradient = np.zeros_like(group_w)
            np.add.at(group_gradient, group_of, region_gradient)
            gradient = np.concatenate([(group_gradient + group_w).ravel(),
                                       (region_gradient + region_w).ravel(),
                                       derivative.sum(axis=0)])
            return loss, gradient

        result = minimize(objective, np.zeros((g + r) * d + r), jac=True,
                          method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.group_w_, self.region_w_, self.bias_ = unpack(result.x)
        self.group_of_ = group_of
        self.n_iter_ = result.nit
        self.converged_ = result.success
        return self

    def predict(self, x):
        if self.projection_ is not None:
            x = x @ self.projection_
        scores = x @ (self.region_w_ + self.group_w_[self.group_of_]).T + self.bias_
        return self.classes_[scores.argmax(axis=1)]


def rbf_features(x_train, x_test):
    """Exact RBF feature coordinates in the span of training samples only."""
    gamma = 1.0 / (x_train.shape[1] * x_train.var())
    kernel = rbf_kernel(x_train, x_train, gamma=gamma)
    values, vectors = eigh(kernel, check_finite=False)
    keep = values > max(values.max() * 1e-10, 1e-12)
    values, vectors = values[keep], vectors[:, keep]
    train_features = vectors * np.sqrt(values)
    test_features = (rbf_kernel(x_test, x_train, gamma=gamma) @ vectors) / np.sqrt(values)
    return train_features, test_features


def prepare_features(x_train, y_train, x_test, methods):
    features = {}
    pca_variance = np.nan
    if "ANOVA" in methods:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores, _ = f_classif(x_train, y_train)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        selected = np.argsort(scores)[-N_GENES:]
        scaler = StandardScaler().fit(x_train[:, selected])
        features["ANOVA"] = (scaler.transform(x_train[:, selected]),
                             scaler.transform(x_test[:, selected]))

    if "MI" in methods:
        # Reuse the notebook's cached MI scores for this exact training fold.
        key = hashlib.sha256(np.ascontiguousarray(x_train).tobytes()
                             + "\n".join(map(str, y_train)).encode()
                             + str(RANDOM_STATE).encode()).hexdigest()[:24]
        path = MI_CACHE_DIR / f"{key}.npz"
        if path.exists():
            scores = np.load(path)["scores"]
        else:
            scores = mutual_info_classif(x_train, y_train,
                                         random_state=RANDOM_STATE, n_jobs=-1)
            MI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, scores=scores)
        selected = np.argsort(scores)[::-1][:N_GENES]
        scaler = StandardScaler().fit(x_train[:, selected])
        features["MI"] = (scaler.transform(x_train[:, selected]),
                          scaler.transform(x_test[:, selected]))

    if "PCA" in methods:
        gene_scaler = StandardScaler().fit(x_train)
        x_train_scaled = gene_scaler.transform(x_train)
        x_test_scaled = gene_scaler.transform(x_test)
        pca = PCA(n_components=min(PCA_MAX_COMPONENTS, len(x_train) - 1),
                  svd_solver="randomized", iterated_power=2,
                  random_state=RANDOM_STATE).fit(x_train_scaled)
        n_pcs = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_),
                                    PCA_VARIANCE_TARGET) + 1)
        assert n_pcs <= pca.n_components_
        train_pcs = pca.transform(x_train_scaled)[:, :n_pcs]
        test_pcs = pca.transform(x_test_scaled)[:, :n_pcs]
        pc_scaler = StandardScaler().fit(train_pcs)
        features["PCA"] = (pc_scaler.transform(train_pcs), pc_scaler.transform(test_pcs))
        pca_variance = pca.explained_variance_ratio_[:n_pcs].sum()
    return features, pca_variance


def main(methods=("ANOVA", "PCA", "MI")):
    data = pd.read_csv(DATA_FILE, index_col=0)
    x = data.to_numpy(dtype=np.float64).T
    sample_names = data.columns.tolist()
    parsed = [re.fullmatch(r"(BB\d+)_(.+)_([GN])_RNA", name).groups() for name in sample_names]
    donors = np.array([row[0] for row in parsed])
    y = np.array([row[1] for row in parsed])
    assert set(y) == set(MACRO_4) == set(MACRO_7)
    rows = []
    predictions = []
    logo = LeaveOneGroupOut()
    for train, test in logo.split(x, y, donors):
        donor = donors[test][0]
        print(f"Preparing features for held-out donor {donor}", flush=True)
        features, pca_variance = prepare_features(x[train], y[train], x[test], methods)
        for feature_name, (x_train, x_test) in features.items():
            for kernel in ["linear", "RBF"]:
                if kernel == "RBF":
                    fit_x, test_x = rbf_features(x_train, x_test)
                else:
                    fit_x, test_x = x_train, x_test
                for n_macro, mapping in [(4, MACRO_4), (7, MACRO_7)]:
                    model = JointHierSVM(mapping).fit(fit_x, y[train])
                    predicted = model.predict(test_x)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                        f1 = f1_score(y[test], predicted, labels=np.unique(y[test]),
                                      average="macro", zero_division=0)
                        ba = balanced_accuracy_score(y[test], predicted)
                    row = {"held_out_donor": donor, "hierarchy": f"{n_macro} macro",
                           "features": feature_name, "kernel": kernel,
                           "macro_f1": f1, "balanced_accuracy": ba,
                           "n_features": x_train.shape[1], "pca_variance": pca_variance,
                           "optimizer_iterations": model.n_iter_, "optimizer_converged": model.converged_}
                    rows.append(row)
                    predictions.extend({"held_out_donor": donor, "hierarchy": f"{n_macro} macro",
                                        "features": feature_name, "kernel": kernel,
                                        "sample_index": int(i), "y_true": truth, "y_pred": pred}
                                       for i, truth, pred in zip(test, y[test], predicted))
                    suffix = "_mi" if methods == ("MI",) else ""
                    pd.DataFrame(rows).to_csv(OUT_DIR / f"fold_scores{suffix}.csv", index=False)
                    pd.DataFrame(predictions).to_csv(OUT_DIR / f"predictions{suffix}.csv", index=False)
                    print(f"  {n_macro} macro, {feature_name} + {kernel}: F1={f1:.3f}, BA={ba:.3f}, "
                          f"iterations={model.n_iter_}", flush=True)
    if methods == ("MI",):
        # Preserve completed ANOVA and PCA predictions, then add the MI runs.
        prior_rows = pd.read_csv(OUT_DIR / "fold_scores.csv")
        prior_predictions = pd.read_csv(OUT_DIR / "predictions.csv")
        prior_rows = prior_rows[prior_rows.features != "MI"]
        prior_predictions = prior_predictions[prior_predictions.features != "MI"]
        rows = pd.concat([prior_rows, pd.DataFrame(rows)], ignore_index=True).to_dict("records")
        predictions = pd.concat([prior_predictions, pd.DataFrame(predictions)],
                                ignore_index=True).to_dict("records")
        pd.DataFrame(rows).to_csv(OUT_DIR / "fold_scores.csv", index=False)
        pd.DataFrame(predictions).to_csv(OUT_DIR / "predictions.csv", index=False)
    summary = (pd.DataFrame(rows).groupby(["hierarchy", "features", "kernel"], sort=False)
               .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                    balanced_accuracy_mean=("balanced_accuracy", "mean"),
                    balanced_accuracy_sd=("balanced_accuracy", "std"),
                    n_donors=("held_out_donor", "nunique"))
               .reset_index())
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print("\n", summary.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main(("MI",) if "--only-mi" in sys.argv else ("ANOVA", "PCA", "MI"))
