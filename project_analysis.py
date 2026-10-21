"""Nested donor-held-out experiments and saved result tables.

The plotting module reads the CSVs written here. The original experiment
definitions are kept together so the checkpoint format and scores stay stable.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from IPython.display import display


def _enter_project(project_dir):
    project_dir = Path(project_dir).resolve()
    os.chdir(project_dir)  # Legacy experiment outputs use paths relative to here.
    return project_dir


def run_nested_models(project_dir, data_file, results_dir, checkpoint_dir):
    """Run or resume the nested direct, macro, and separate two-stage grid."""
    from pathlib import Path
    PROJECT_DIR = _enter_project(project_dir)
    DATA_FILE = Path(data_file)
    RESULTS_DIR = Path(results_dir)
    RESUME_CHECKPOINT_DIR = Path(checkpoint_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_summary = RESULTS_DIR / "full_grid_summary.csv"
    if saved_summary.is_file():
        import pandas as pd
        print("Loaded saved nested-grid summary; remove it to rerun the grid.")
        return pd.read_csv(saved_summary)
    RESUME_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Data, features, and LODO experiment engine

    import re
    import pickle
    import os
    import hashlib
    import json
    import warnings
    from collections import Counter
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.decomposition import PCA
    from sklearn.feature_selection import f_classif, mutual_info_classif
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 confusion_matrix, ConfusionMatrixDisplay, classification_report)

    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_rows", 100)

    RANDOM_STATE = 42
    rng = np.random.default_rng(RANDOM_STATE)

    # A held-out donor may not contain every region, so sklearn warns that
    # "y_pred contains classes not in y_true". This is expected here, so we hide it.
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

    # The path variables are configured in the first code cell.
    DATA_DIR = DATA_FILE.parent
    FILE_PATH = DATA_FILE
    SAVE_RESULTS = True
    RESUME_BACKUP_DIR = PROJECT_DIR / "resume_saved_results"

    def save(obj, filename, index=False):
        """Save a table (.csv), a figure (.png) or any Python object (.pkl) into RESULTS_DIR."""
        if not SAVE_RESULTS:
            return
        path = RESULTS_DIR / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(path, index=index)
        elif isinstance(obj, plt.Figure):
            obj.savefig(path, dpi=300, bbox_inches="tight")
        else:
            with open(path, "wb") as file:
                pickle.dump(obj, file)

    def load_dataset(file_path):
        """
        Read the CSV and return:
            X        -> DataFrame, rows = samples, columns = genes  (ML layout)
            metadata -> DataFrame, one row per sample: donor, region, fraction
        """
        df = pd.read_csv(file_path, index_col=0)      # first column = gene IDs
        df.index.name = "gene_id"
        print("Original shape (genes x samples):", df.shape)
        print("Duplicated gene IDs:", df.index.duplicated().sum(),
              "| Duplicated sample names:", df.columns.duplicated().sum())

        # donor _ region _ fraction _RNA   e.g.  BB3_ARC_G_RNA
        pattern = re.compile(r"^(BB\d+)_(.+)_([GN])_RNA$")
        matches = {name: pattern.match(name) for name in df.columns}
        print("Invalid sample names:", [name for name, m in matches.items() if m is None])

        valid = [name for name, m in matches.items() if m is not None]
        metadata = pd.DataFrame([matches[name].groups() for name in valid],
                                columns=["donor", "region", "fraction"],
                                index=pd.Index(valid, name="sample"))
        metadata["fraction_label"] = metadata["fraction"].map({"N": "Neuronal", "G": "Non-neuronal"})

        X = df[valid].T                               # transpose -> samples x genes
        return X, metadata


    X, metadata = load_dataset(FILE_PATH)
    assert X.index.equals(metadata.index), "X and metadata sample order do not match."

    y = metadata["region"]                # target: brain region
    groups = metadata["donor"]            # used to split train/test by donor
    fraction = metadata["fraction_label"]

    display(metadata.head())

    X_array = X.to_numpy(dtype=np.float64)        # (265 samples, 33279 genes)
    y_region = y.to_numpy()                        # 25 brain regions
    donors = groups.to_numpy()                     # BB1 ... BB6
    fraction_labels = fraction.to_numpy()          # "Neuronal" / "Non-neuronal"
    gene_names = X.columns.to_numpy()
    REGION_LABELS = sorted(np.unique(y_region))

    K_VALUES = [50, 100, 250, 500, 1000, 2000, 5000, 10000]   # k = how many top-ranked genes the SVM may use
    SVM_C_VALUES = [0.1, 1, 10]                    # SVM regularization (small C = simpler model)

    print("Number of LODO folds:", LeaveOneGroupOut().get_n_splits(groups=donors))

    # ------------------------------------------------------------
    # Gene rankers: return gene indices sorted from best to worst
    # ------------------------------------------------------------
    def rank_by_anova(X_train, y_train):
        """ANOVA F-test: how strongly a gene's mean differs between classes."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        # genes constant within classes -> divide-by-zero warning
            scores, _ = f_classif(X_train, y_train)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)  # NaN -> bottom of ranking
        return np.argsort(scores)[::-1]


    MI_CACHE_DIR = RESUME_CHECKPOINT_DIR / "mi_scores"     # one file of MI scores per training set


    def rank_by_mutual_info(X_train, y_train):
        """Mutual information: how much a gene tells us about the class (also non-linear).
        MI is slow, so the scores of every training set (inner folds included) are saved on disk,
        keyed by the exact training data + labels, and reused by any later experiment."""
        key = hashlib.sha256(np.ascontiguousarray(X_train).tobytes()
                             + "\n".join(map(str, y_train)).encode()
                             + str(RANDOM_STATE).encode()).hexdigest()[:24]
        path = MI_CACHE_DIR / f"{key}.npz"
        if path.exists():
            scores = np.load(path)["scores"]
        else:
            scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE, n_jobs=-1)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f"{key}.tmp{os.getpid()}.npz")
            np.savez_compressed(temporary, scores=scores)
            os.replace(temporary, path)                    # never leave a half-written file
        return np.argsort(scores)[::-1]


    # ------------------------------------------------------------
    # Feature methods: (X_train, y_train, X_test, k_values) -> {k: (X_train_k, X_test_k, selected_genes)}
    # ------------------------------------------------------------
    def top_genes(ranker):
        """Keep the top-k genes of `ranker`, then standardize them (fit on training donors only)."""
        def build_features(X_train, y_train, X_test, k_values):
            ranked = ranker(X_train, y_train)
            features = {}
            for k in k_values:
                idx = ranked[:k]
                scaler = StandardScaler().fit(X_train[:, idx])
                features[k] = (scaler.transform(X_train[:, idx]),
                               scaler.transform(X_test[:, idx]),
                               set(gene_names[idx]))
            return features
        return build_features


    def pca_components(X_train, y_train, X_test, k_values):
        """Standardize all genes, fit PCA on training donors, keep the first k components,
        then standardize the components too. (y_train is not used: PCA does not look at the labels.)"""
        scaler = StandardScaler().fit(X_train)
        X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

        n_max = min(max(k_values), X_train.shape[0] - 1)   # cannot have more PCs than training samples
        pca = PCA(n_components=n_max, random_state=RANDOM_STATE).fit(X_train_scaled)
        Z_train, Z_test = pca.transform(X_train_scaled), pca.transform(X_test_scaled)

        features = {}
        for k in k_values:
            if k > n_max:
                continue
            # PC1 has ~15x larger values than the last PCs; without rescaling the SVM
            # converges extremely slowly (seconds -> minutes per fit) and PC1 dominates.
            pc_scaler = StandardScaler().fit(Z_train[:, :k])
            features[k] = (pc_scaler.transform(Z_train[:, :k]), pc_scaler.transform(Z_test[:, :k]), None)
        return features

    def make_svm(C):
        """The linear SVM used in every experiment. class_weight='balanced' -> rare regions count as much as common ones."""
        return LinearSVC(C=C, class_weight="balanced", dual="auto", max_iter=10000, random_state=RANDOM_STATE)


    def score(y_true, y_pred, f1_labels=None):
        """Macro F1, balanced accuracy and accuracy.
        f1_labels = classes averaged in macro F1 (None -> only classes present in y_true)."""
        labels = np.unique(y_true) if f1_labels is None else f1_labels
        return {"macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "accuracy": accuracy_score(y_true, y_pred)}


    def summarize(fold_results):
        """Mean and std over the held-out donors for every (k, C).
        Sorted best first: macro F1, then balanced accuracy, then accuracy; ties -> fewer features, smaller C."""
        summary = (fold_results.groupby(["k", "svm_C"])
                   .agg(mean_macro_f1=("macro_f1", "mean"), std_macro_f1=("macro_f1", "std"),
                        mean_balanced_accuracy=("balanced_accuracy", "mean"),
                        std_balanced_accuracy=("balanced_accuracy", "std"),
                        mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"))
                   .reset_index())
        return (summary.sort_values(["mean_macro_f1", "mean_balanced_accuracy", "mean_accuracy", "k", "svm_C"],
                                    ascending=[False, False, False, True, True])
                       .reset_index(drop=True))


    def evaluate_settings(X_train, y_train, X_test, y_test, feature_method, k_values, c_values, f1_labels=None):
        """ONE train/test split: build features (from training data only), then train + score an SVM
        for every (k, C). Returns one dict per (k, C) with the metrics, predictions and selected genes."""
        rows = []
        features = feature_method(X_train, y_train, X_test, k_values)
        for k, (X_train_k, X_test_k, genes_k) in features.items():
            for C in c_values:
                y_pred = make_svm(C).fit(X_train_k, y_train).predict(X_test_k)
                rows.append({"k": k, "svm_C": C, "y_pred": y_pred, "genes": genes_k,
                             **score(y_test, y_pred, f1_labels)})
        return rows


    def tune_on_training_donors(X_train, y_train, donors_train, feature_method,
                                k_values=K_VALUES, c_values=SVM_C_VALUES, f1_labels=None):
        """INNER Leave-One-Donor-Out on the training donors only -> best (k, C).
        The outer test donor is never used here, so the choice cannot be biased by it."""
        rows = []
        for inner_train, inner_val in LeaveOneGroupOut().split(X_train, y_train, donors_train):
            if len(np.unique(y_train[inner_train])) < 2:      # an SVM needs at least 2 classes
                continue
            for r in evaluate_settings(X_train[inner_train], y_train[inner_train],
                                       X_train[inner_val], y_train[inner_val],
                                       feature_method, k_values, c_values, f1_labels):
                rows.append({key: r[key] for key in ["k", "svm_C", "macro_f1", "balanced_accuracy", "accuracy"]})
        best = summarize(pd.DataFrame(rows)).iloc[0]          # same ranking rule as the outer loop
        C = best["svm_C"]                                     # a number, or a text setting for the RBF SVM (section 18)
        return int(best["k"]), (C if isinstance(C, str) else float(C))


    # ============================================================
    # MAIN EXPERIMENT LOOP — used by every experiment in Part 3
    # ============================================================
    CHECKPOINT_VERSION = 1


    def atomic_pickle(obj, path):
        """Replace a checkpoint only after the new pickle is fully written."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        with open(temporary, "wb") as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)


    def checkpoint_signature(kind, name, sample_index, y_sub, donors_sub, k_values=None,
                             c_values=None, nested=None, f1_labels=None):
        """Reject checkpoints from a different dataset or experiment setting."""
        stat = FILE_PATH.stat()
        config = {
            "version": CHECKPOINT_VERSION, "kind": kind, "name": name,
            "data_size": stat.st_size, "data_mtime_ns": stat.st_mtime_ns,
            "sample_index": np.asarray(sample_index).tolist(),
            "labels": np.asarray(y_sub).tolist(),
            "donors": np.asarray(donors_sub).tolist(),
            "k_values": list(k_values) if k_values is not None else None,
            "c_values": list(c_values) if c_values is not None else None,
            "nested": nested,
            "f1_labels": list(f1_labels) if f1_labels is not None else None,
        }
        encoded = json.dumps(config, sort_keys=True, default=str).encode()
        return hashlib.sha256(encoded).hexdigest()


    def load_checkpoint(path, signature):
        if not path.exists():
            return None
        try:
            with open(path, "rb") as file:
                payload = pickle.load(file)
        except (OSError, EOFError, pickle.UnpicklingError) as error:
            print(f"Ignoring damaged checkpoint {path}: {error}")
            return None
        if payload.get("signature") != signature:
            print(f"Ignoring checkpoint with different settings: {path}")
            return None
        return payload


    def save_experiment_tables(result, folder):
        for part in ["summary", "fold_results", "predictions"]:
            save(result[part], f"{folder}/{part}.csv")
        if result["selected_genes"]:
            save(result["selected_genes"], f"{folder}/selected_genes.pkl")


    def load_legacy_experiment(folder, nested, sample_index, y_sub, donors_sub,
                               k_values, c_values):
        """Reuse complete CSV output from the original, non-checkpointed notebook."""
        for base in (RESULTS_DIR, RESUME_BACKUP_DIR):
            location = base / folder
            files = {part: location / f"{part}.csv"
                     for part in ["summary", "fold_results", "predictions"]}
            if not all(path.exists() for path in files.values()):
                continue
            try:
                tables = {part: pd.read_csv(path) for part, path in files.items()}
                fold = tables["fold_results"]
                preds = tables["predictions"]
                expected_settings = 1 if nested else len(k_values) * len(c_values)
                expected_folds = len(np.unique(donors_sub)) * expected_settings
                if len(fold) != expected_folds or len(preds) != len(sample_index) * expected_settings:
                    continue
                if set(fold["test_donor"]) != set(donors_sub):
                    continue
                if set(preds["sample_index"]) != set(sample_index):
                    continue
                counts = preds.groupby("sample_index").size()
                if not (counts == expected_settings).all():
                    continue
                labels = preds.groupby("sample_index")["y_true"].first()
                expected_labels = pd.Series(y_sub, index=sample_index)
                if not labels.sort_index().equals(expected_labels.sort_index()):
                    continue
                if len(tables["summary"]) != expected_settings:
                    continue
                genes_path = location / "selected_genes.pkl"
                selected = None
                if genes_path.exists():
                    with open(genes_path, "rb") as file:
                        selected = pickle.load(file)
                return {"nested": nested, **tables, "selected_genes": selected}
            except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError):
                continue
        return None


    def run_lodo_experiment(name, y, feature_method, k_values=K_VALUES, c_values=SVM_C_VALUES,
                            sample_mask=None, f1_labels=None, nested=False):
        """Run LODO, saving each completed outer donor fold for later resumption."""
        sample_index = np.arange(len(y)) if sample_mask is None else np.flatnonzero(sample_mask)
        X_sub, y_sub, donors_sub = X_array[sample_index], np.asarray(y)[sample_index], donors[sample_index]
        mode = "nested" if nested else "grid"
        folder = f"{name}_{mode}"
        checkpoint_dir = RESUME_CHECKPOINT_DIR / folder
        signature = checkpoint_signature("lodo", name, sample_index, y_sub, donors_sub,
                                         k_values, c_values, nested, f1_labels)

        complete = load_checkpoint(checkpoint_dir / "complete.pkl", signature)
        if complete is not None:
            print(f"Loaded completed {folder} from checkpoint")
            result = complete["result"]
            save_experiment_tables(result, folder)
            return result

        legacy = load_legacy_experiment(folder, nested, sample_index, y_sub, donors_sub,
                                        k_values, c_values)
        if legacy is not None:
            print(f"Loaded completed {folder} from saved result files")
            atomic_pickle({"signature": signature, "result": legacy}, checkpoint_dir / "complete.pkl")
            save_experiment_tables(legacy, folder)
            return legacy

        print(f"\n{name} ({mode}): {len(y_sub)} samples | classes: {list(np.unique(y_sub))}")
        fold_rows, prediction_rows, selected_genes = [], [], {}
        for train, test in LeaveOneGroupOut().split(X_sub, y_sub, donors_sub):
            test_donor = str(donors_sub[test][0])
            fold_path = checkpoint_dir / f"donor_{test_donor}.pkl"
            payload = load_checkpoint(fold_path, signature)
            if payload is None:
                print(f"Computing {folder}, test donor {test_donor}", flush=True)
                if nested:
                    best_k, best_C = tune_on_training_donors(
                        X_sub[train], y_sub[train], donors_sub[train],
                        feature_method, k_values, c_values, f1_labels)
                    ks, cs = [best_k], [best_C]
                else:
                    ks, cs = k_values, c_values

                donor_folds, donor_predictions, donor_genes = [], [], {}
                for row in evaluate_settings(X_sub[train], y_sub[train], X_sub[test], y_sub[test],
                                             feature_method, ks, cs, f1_labels):
                    if row["genes"] is not None:
                        donor_genes[row["k"]] = row["genes"]
                    donor_folds.append({"test_donor": test_donor, "k": row["k"],
                                        "svm_C": row["svm_C"],
                                        **{m: row[m] for m in ["macro_f1", "balanced_accuracy", "accuracy"]}})
                    donor_predictions.extend(
                        {"sample_index": int(i), "test_donor": test_donor, "k": row["k"],
                         "svm_C": row["svm_C"], "y_true": true, "y_pred": pred}
                        for i, true, pred in zip(sample_index[test], y_sub[test], row["y_pred"]))
                payload = {"signature": signature, "fold_rows": donor_folds,
                           "prediction_rows": donor_predictions, "selected_genes": donor_genes}
                atomic_pickle(payload, fold_path)
                print(f"Saved {folder}, test donor {test_donor}", flush=True)
            else:
                print(f"Skipping completed {folder}, test donor {test_donor}", flush=True)
            fold_rows.extend(payload["fold_rows"])
            prediction_rows.extend(payload["prediction_rows"])
            for k, genes in payload["selected_genes"].items():
                selected_genes.setdefault(k, {})[test_donor] = genes

        fold_results = pd.DataFrame(fold_rows)
        if nested:
            summary = pd.DataFrame([{f"{stat}_{metric}": getattr(fold_results[metric], stat)()
                                     for metric in ["macro_f1", "balanced_accuracy", "accuracy"]
                                     for stat in ["mean", "std"]}])
        else:
            summary = summarize(fold_results)
        result = {"nested": nested, "fold_results": fold_results, "summary": summary,
                  "predictions": pd.DataFrame(prediction_rows),
                  "selected_genes": selected_genes or None}
        atomic_pickle({"signature": signature, "result": result}, checkpoint_dir / "complete.pkl")
        save_experiment_tables(result, folder)
        return result

    def _is_best(table, result):
        """Rows of `table` that belong to the best (k, C). For nested results every row is 'best'."""
        if result["nested"]:
            return np.ones(len(table), bool)
        best = result["summary"].iloc[0]
        return (table["k"] == best["k"]) & np.isclose(table["svm_C"], best["svm_C"])


    def best_predictions(result):
        """Out-of-fold predictions (one per sample) of the best (k, C) — or of the nested model."""
        chosen = result["predictions"][_is_best(result["predictions"], result)]
        chosen = chosen.sort_values("sample_index").reset_index(drop=True)
        assert chosen["sample_index"].is_unique
        return chosen


    def best_fold_results(result):
        """Per-donor metrics of the best (k, C) — or of the nested model (with the (k, C) chosen per donor)."""
        return result["fold_results"][_is_best(result["fold_results"], result)].reset_index(drop=True)


    def settings_label(result):
        """Short text describing which (k, C) produced the predictions (for plot titles)."""
        if result["nested"]:
            return "nested LODO (k, C tuned per donor)"
        best = result["summary"].iloc[0]
        return f"k={int(best['k'])} | C={best['svm_C']}"


    def pooled_scores(model_name, df, labels, prediction_column="y_pred"):
        """Metrics over ALL out-of-fold predictions in df (not averaged per donor)."""
        return {"model": model_name, "n_samples": len(df),
                **score(df["y_true"], df[prediction_column], labels)}


    def most_common_errors(predictions, by=()):
        """Count (true -> predicted) pairs among wrong predictions."""
        errors = predictions[predictions["y_true"] != predictions["y_pred"]]
        return (errors.groupby([*by, "y_true", "y_pred"]).size().reset_index(name="count")
                      .sort_values([*by, "count"], ascending=[True] * len(by) + [False]))


    def gene_stability(genes_by_donor):
        """How many LODO folds selected each gene (selected in all folds = stable marker)."""
        counts = Counter(gene for genes in genes_by_donor.values() for gene in genes)
        table = pd.DataFrame(counts.items(), columns=["gene_id", "fold_frequency"])
        table["selection_frequency"] = table["fold_frequency"] / len(genes_by_donor)
        return table.sort_values(["fold_frequency", "gene_id"], ascending=[False, True]).reset_index(drop=True)
    # Nested direct and macro baselines

    anova_25 = run_lodo_experiment("anova_25_regions", y_region, top_genes(rank_by_anova), nested=True)
    print("\nNested LODO (honest) — per held-out donor, with the (k, C) chosen on the training donors:")
    print(anova_25["summary"].round(4).to_string(index=False))

    best_25 = best_predictions(anova_25)
    best_25["fraction"] = fraction_labels[best_25["sample_index"]]

    FRACTION_CLASSES = ["Non-neuronal", "Neuronal"]

    binary = run_lodo_experiment("binary_fraction", fraction_labels, top_genes(rank_by_anova),
                                 f1_labels=FRACTION_CLASSES)
    print("\nBinary classification results (best first):")
    display(binary["summary"].round(4))

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

    sample_info = pd.DataFrame({"donor": donors, "macro_region": y_macro, "fraction": fraction_labels})
    display(sample_info["macro_region"].value_counts())
    display(pd.crosstab(sample_info["macro_region"], sample_info["fraction"]))
    display(pd.crosstab(sample_info["donor"], sample_info["macro_region"]))

    # Which samples each model is trained/tested on
    SUBSETS = {"all_samples":  None,
               "neuronal":     fraction_labels == "Neuronal",
               "non_neuronal": fraction_labels == "Non-neuronal"}

    macro = {name: run_lodo_experiment(f"macro_{name}", y_macro, top_genes(rank_by_anova),
                                       sample_mask=mask, f1_labels=MACRO_LABELS, nested=True)
             for name, mask in SUBSETS.items()}

    # ------------------------------------------------------------
    # Honest (nested) score of each model + the (k, C) chosen per donor
    # ------------------------------------------------------------
    macro_comparison = pd.DataFrame([
        {"experiment": name, "n_samples": result["predictions"]["sample_index"].nunique(),
         "chosen_k_per_donor": result["fold_results"]["k"].tolist(),
         "chosen_C_per_donor": result["fold_results"]["svm_C"].tolist(),
         **result["summary"].iloc[0][["mean_macro_f1", "std_macro_f1",
                                      "mean_balanced_accuracy", "mean_accuracy"]].to_dict()}
        for name, result in macro.items()])
    print("Macro-region classification (nested LODO):")
    display(macro_comparison.round(4))
    save(macro_comparison, "macro_experiment_comparison.csv")

    best_macro = {name: best_predictions(result) for name, result in macro.items()}

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
    display(pd.crosstab(pd.Series(y_macro7, name="macro_7"), pd.Series(donors, name="donor"))
              .loc[MACRO_7_LABELS])

    # Same pipeline as the 4-macro model: ANOVA top-k genes + LinearSVC, nested LODO
    macro7 = run_lodo_experiment("macro7_all_samples", y_macro7, top_genes(rank_by_anova),
                                 f1_labels=MACRO_7_LABELS, nested=True)
    print("\n7 macro regions (nested LODO) — per held-out donor, with the (k, C) chosen on the training donors:")
    display(macro7["fold_results"].round(4))
    display(macro7["summary"].round(4))
    # Separate two-stage and full nested grid

    def fit_micro_model(X_train, y_train, donors_train):
        """SVM for the regions inside ONE macro group.
        (k, C) are tuned by nested LODO on the training donors, exactly like the direct model.
        If the group has only one region in training (e.g. Hindbrain = CRBLM), always predict it."""
        classes = np.unique(y_train)
        if len(classes) == 1:
            return {"constant": classes[0], "k": None, "C": None}
        k, C = tune_on_training_donors(X_train, y_train, donors_train, top_genes(rank_by_anova))
        idx = rank_by_anova(X_train, y_train)[:k]
        scaler = StandardScaler().fit(X_train[:, idx])
        svm = make_svm(C).fit(scaler.transform(X_train[:, idx]), y_train)
        return {"constant": None, "indices": idx, "scaler": scaler, "svm": svm, "k": k, "C": C}


    def predict_micro(model, sample_index):
        if model["constant"] is not None:
            return model["constant"]
        x = X_array[[sample_index]][:, model["indices"]]
        return model["svm"].predict(model["scaler"].transform(x))[0]


    def run_hierarchical(split_by_fraction):
        """Run both hierarchical variants with a checkpoint for each test donor."""
        name = "fraction_specific" if split_by_fraction else "all_samples"
        checkpoint_dir = RESUME_CHECKPOINT_DIR / "hierarchical" / name
        signature = checkpoint_signature("hierarchical", name, np.arange(len(y_region)),
                                         y_region, donors, K_VALUES, SVM_C_VALUES,
                                         split_by_fraction)
        complete = load_checkpoint(checkpoint_dir / "complete.pkl", signature)
        if complete is not None:
            print(f"Loaded completed hierarchical/{name} from checkpoint")
            result, settings, selected_genes = (complete[key] for key in
                                                 ["results", "settings", "selected_genes"])
            save(result, f"hierarchical/{name}_predictions.csv")
            save(settings, f"hierarchical/{name}_chosen_settings.csv")
            save(selected_genes, f"hierarchical/{name}_selected_genes.pkl")
            return result, settings

        # The original notebook may have finished this variant before shutdown.
        for base in (RESULTS_DIR, RESUME_BACKUP_DIR):
            location = base / "hierarchical"
            predictions_path = location / f"{name}_predictions.csv"
            settings_path = location / f"{name}_chosen_settings.csv"
            genes_path = location / f"{name}_selected_genes.pkl"
            if not all(path.exists() for path in (predictions_path, settings_path, genes_path)):
                continue
            try:
                result = pd.read_csv(predictions_path).sort_values("sample_index").reset_index(drop=True)
                settings = pd.read_csv(settings_path)
                if len(result) != len(y_region) or len(settings) != len(np.unique(donors)) * (8 if split_by_fraction else 4):
                    continue
                if not np.array_equal(result["sample_index"], np.arange(len(y_region))):
                    continue
                if not np.array_equal(result["y_true"], y_region):
                    continue
                with open(genes_path, "rb") as file:
                    selected_genes = pickle.load(file)
                atomic_pickle({"signature": signature, "results": result, "settings": settings,
                               "selected_genes": selected_genes}, checkpoint_dir / "complete.pkl")
                print(f"Loaded completed hierarchical/{name} from saved result files")
                return result, settings
            except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError):
                continue

        macro_source = {None: "all_samples", "Neuronal": "neuronal", "Non-neuronal": "non_neuronal"}
        macro_pred = {key: preds.set_index("sample_index")["y_pred"]
                      for key, preds in best_macro.items()}
        rows, chosen_settings, selected_genes = [], [], {}

        for test_donor in np.unique(donors):
            donor_path = checkpoint_dir / f"donor_{test_donor}.pkl"
            payload = load_checkpoint(donor_path, signature)
            if payload is None:
                print(f"Computing hierarchical/{name}, test donor {test_donor}", flush=True)
                donor_rows, donor_settings, donor_genes = [], [], {}
                for frac in (["Neuronal", "Non-neuronal"] if split_by_fraction else [None]):
                    in_subset = np.ones(len(y_region), bool) if frac is None else (fraction_labels == frac)
                    train_idx = np.flatnonzero((donors != test_donor) & in_subset)
                    test_idx = np.flatnonzero((donors == test_donor) & in_subset)
                    micro_models = {}
                    for macro_name in MACRO_LABELS:
                        group_idx = train_idx[y_macro[train_idx] == macro_name]
                        assert len(group_idx) > 0, f"No training samples for {macro_name}, {test_donor}, {frac}"
                        model = fit_micro_model(X_array[group_idx], y_region[group_idx], donors[group_idx])
                        micro_models[macro_name] = model
                        donor_settings.append({"test_donor": test_donor, "fraction": frac,
                                               "macro": macro_name, "k": model["k"], "C": model["C"]})
                        if model["constant"] is None:
                            donor_genes[(test_donor, str(frac), macro_name)] = gene_names[model["indices"]].tolist()
                    for i in test_idx:
                        predicted_macro = macro_pred[macro_source[frac]].loc[i]
                        donor_rows.append({"sample_index": int(i), "test_donor": test_donor,
                                           "fraction": fraction_labels[i], "y_true": y_region[i],
                                           "macro_true": y_macro[i], "macro_pred": predicted_macro,
                                           "y_pred": predict_micro(micro_models[predicted_macro], i),
                                           "oracle_micro_pred": predict_micro(micro_models[y_macro[i]], i)})
                payload = {"signature": signature, "rows": donor_rows,
                           "settings": donor_settings, "selected_genes": donor_genes}
                atomic_pickle(payload, donor_path)
                print(f"Saved hierarchical/{name}, test donor {test_donor}", flush=True)
            else:
                print(f"Skipping completed hierarchical/{name}, test donor {test_donor}", flush=True)
            rows.extend(payload["rows"])
            chosen_settings.extend(payload["settings"])
            selected_genes.update(payload["selected_genes"])

        result = pd.DataFrame(rows).sort_values("sample_index").reset_index(drop=True)
        assert len(result) == len(y_region) and result["sample_index"].is_unique
        assert np.array_equal(result["y_true"], y_region)
        settings = pd.DataFrame(chosen_settings)
        atomic_pickle({"signature": signature, "results": result, "settings": settings,
                       "selected_genes": selected_genes}, checkpoint_dir / "complete.pkl")
        save(result, f"hierarchical/{name}_predictions.csv")
        save(settings, f"hierarchical/{name}_chosen_settings.csv")
        save(selected_genes, f"hierarchical/{name}_selected_genes.pkl")
        return result, settings


    hier_all, hier_all_settings = run_hierarchical(split_by_fraction=False)
    hier_fraction, hier_fraction_settings = run_hierarchical(split_by_fraction=True)

    print("\n(k, C) chosen for the micro models (all-samples version):")
    display(hier_all_settings.pivot(index="test_donor", columns="macro", values="k"))

    # ------------------------------------------------------------
    # Direct (Experiment 1, nested) vs hierarchical, all on the 25 regions
    # ------------------------------------------------------------
    comparison = pd.DataFrame([
        pooled_scores("Direct - 25 regions", best_25, REGION_LABELS),
        pooled_scores("Hierarchical - All samples", hier_all, REGION_LABELS),
        pooled_scores("Hierarchical - Fraction-specific", hier_fraction, REGION_LABELS),
        pooled_scores("Micro only - True macro (All samples)", hier_all, REGION_LABELS, "oracle_micro_pred"),
        pooled_scores("Micro only - True macro (Fraction-specific)", hier_fraction, REGION_LABELS, "oracle_micro_pred"),
    ])
    print("Final 25-region classification comparison (all models tuned by nested LODO):")
    display(comparison.round(4))
    save(comparison, "hierarchical/comparison.csv")

    # How good is each step?
    diagnostics = pd.DataFrame([
        {"experiment": name,
         "macro_routing_accuracy": (df["macro_true"] == df["macro_pred"]).mean(),
         "micro_accuracy_given_true_macro": (df["y_true"] == df["oracle_micro_pred"]).mean(),
         "end_to_end_accuracy": (df["y_true"] == df["y_pred"]).mean()}
        for name, df in [("All samples", hier_all), ("Fraction-specific", hier_fraction)]])
    print("\nStage-level diagnostics:")
    display(diagnostics.round(4))

    # Per-fraction comparison on matching test samples
    fraction_comparison = pd.DataFrame([
        {**pooled_scores(model_name, subset, REGION_LABELS), "fraction": fraction_name}
        for model_name, df in [("Direct 25 regions (baseline)", best_25), ("Hierarchical - All", hier_all),
                               ("Hierarchical - Fraction-specific", hier_fraction)]
        for fraction_name, subset in df.groupby("fraction")])
    display(fraction_comparison[["model", "fraction", "n_samples", "macro_f1",
                                 "balanced_accuracy", "accuracy"]].round(4))
    save(fraction_comparison, "hierarchical/fraction_comparison.csv")

    # ------------------------------------------------------------
    # Every feature method × every training scheme, all tuned by nested LODO.
    # Finished runs are loaded from resume_checkpoints/, so only missing ones are trained.
    # ------------------------------------------------------------
    from sklearn.svm import SVC

    PCA_N_COMPONENTS = [10, 20, 50, 100, 150]


    class RBFSVM:
        """RBF-kernel SVM. gamma = gamma_factor × sklearn's 'scale' value (1 / (n_features × variance)),
        so the same factors make sense for k = 50 and k = 10000 genes."""
        def __init__(self, C, gamma_factor):
            self.C, self.gamma_factor = C, gamma_factor

        def fit(self, X, y):
            gamma = self.gamma_factor / (X.shape[1] * X.var())
            self.model = SVC(kernel="rbf", C=self.C, gamma=gamma, class_weight="balanced").fit(X, y)
            return self

        def predict(self, X):
            return self.model.predict(X)


    # RBF settings are written as text, e.g. "C=10, gamma=1x scale", so they fit in the svm_C column
    RBF_PARAMS = [f"C={C}, gamma={g}x scale" for C in [1, 10, 100] for g in [0.1, 1, 10]]


    def make_svm(C):
        """Linear SVM for a number C (as in all other experiments), RBF SVM for an RBF_PARAMS text."""
        if isinstance(C, str):
            c_part, g_part = C.split(", ")
            return RBFSVM(float(c_part[2:]), float(g_part[6:].split("x")[0]))
        return LinearSVC(C=C, class_weight="balanced", dual="auto", max_iter=10000, random_state=RANDOM_STATE)


    FEATURE_METHODS = {     # name -> (prefix of the saved experiment names, feature method, k values, SVM settings)
        "ANOVA":     ("",      top_genes(rank_by_anova),       K_VALUES,         SVM_C_VALUES),
        "PCA":       ("pca_",  pca_components,                 PCA_N_COMPONENTS, SVM_C_VALUES),
        "MI":        ("mi_",   top_genes(rank_by_mutual_info), K_VALUES,         SVM_C_VALUES),
        "ANOVA-RBF": ("rbf_",  top_genes(rank_by_anova),       K_VALUES,         RBF_PARAMS),
        "PCA-RBF":   ("rbf_pca_", pca_components,              PCA_N_COMPONENTS, RBF_PARAMS),
        "MI-RBF":    ("rbf_mi_",  top_genes(rank_by_mutual_info), K_VALUES,       RBF_PARAMS),
    }
    MACRO_SCHEMES = {       # name -> (prefix of the saved experiment names, labels per sample, class list)
        "4 macro": ("macro",  y_macro,  MACRO_LABELS),
        "7 macro": ("macro7", y_macro7, MACRO_7_LABELS),
    }
    SUBSETS = {"all_samples":  None,
               "neuronal":     fraction_labels == "Neuronal",
               "non_neuronal": fraction_labels == "Non-neuronal"}
    grid_results = {}       # (method, scheme, variant) -> result


    def run_direct_and_macro(method):
        """25 regions directly + 4 and 7 macro regions (all samples, neuronal, non-neuronal)."""
        prefix, features, k_values, c_values = FEATURE_METHODS[method]
        name_25 = {"ANOVA": "anova_25_regions", "PCA": "pca_25_regions", "MI": "mi_25_regions",
                   "ANOVA-RBF": "rbf_anova_25_regions", "PCA-RBF": "rbf_pca_25_regions",
                   "MI-RBF": "rbf_mi_25_regions"}[method]
        grid_results[(method, "25 regions", "all_samples")] = run_lodo_experiment(
            name_25, y_region, features, k_values=k_values, c_values=c_values, nested=True)
        for scheme, (scheme_prefix, y_scheme, labels) in MACRO_SCHEMES.items():
            for subset, mask in SUBSETS.items():
                grid_results[(method, scheme, subset)] = run_lodo_experiment(
                    f"{prefix}{scheme_prefix}_{subset}", y_scheme, features, k_values=k_values,
                    c_values=c_values, sample_mask=mask, f1_labels=labels, nested=True)


    def micro_predictions(method, train_idx, test_idx):
        """Region SVM inside ONE macro group, (k, C) tuned by nested LODO on its training donors.
        Returns predictions for all test samples (we later keep the ones routed to this group)."""
        _, features, k_values, c_values = FEATURE_METHODS[method]
        X_train, y_train, donors_train = X_array[train_idx], y_region[train_idx], donors[train_idx]
        if len(np.unique(y_train)) == 1:                     # e.g. Cerebellum = CRBLM only
            return np.full(len(test_idx), y_train[0]), {"k": None, "C": None}, None
        if method.startswith("PCA"):
            # Small groups: keep only k values that every inner training set can provide
            smallest = min(len(tr) for tr, _ in LeaveOneGroupOut().split(X_train, y_train, donors_train))
            k_values = [k for k in k_values if k < smallest] or [smallest - 1]
        k, C = tune_on_training_donors(X_train, y_train, donors_train, features, k_values, c_values)
        X_train_k, X_test_k, genes = features(X_train, y_train, X_array[test_idx], [k])[k]
        return make_svm(C).fit(X_train_k, y_train).predict(X_test_k), {"k": k, "C": C}, genes


    def run_hierarchical_grid(method, scheme, split_by_fraction):
        """Macro model of `scheme` (same feature method) -> region model inside the predicted group."""
        variant = "fraction_specific" if split_by_fraction else "all_samples"
        if method == "ANOVA" and scheme == "4 macro":         # already done in Experiment 4
            return pd.read_csv(RESULTS_DIR / "hierarchical" / f"{variant}_predictions.csv")
        _, y_scheme, labels = MACRO_SCHEMES[scheme]
        name = f"{method.lower().replace('-', '_')}_{MACRO_SCHEMES[scheme][0]}_{variant}"
        checkpoint_dir = RESUME_CHECKPOINT_DIR / "hierarchical_grid" / name
        signature = checkpoint_signature("hierarchical_grid", name, np.arange(len(y_region)), y_region,
                                         donors, FEATURE_METHODS[method][2], FEATURE_METHODS[method][3], split_by_fraction)
        complete = load_checkpoint(checkpoint_dir / "complete.pkl", signature)
        if complete is not None:
            print(f"Loaded completed hierarchical_grid/{name} from checkpoint")
            return complete["results"]

        macro_pred = {subset: best_predictions(grid_results[(method, scheme, subset)])
                                  .set_index("sample_index")["y_pred"] for subset in SUBSETS}
        rows, settings = [], []
        for test_donor in np.unique(donors):
            donor_path = checkpoint_dir / f"donor_{test_donor}.pkl"
            payload = load_checkpoint(donor_path, signature)
            if payload is None:
                print(f"Computing hierarchical_grid/{name}, test donor {test_donor}", flush=True)
                donor_rows, donor_settings = [], []
                for subset in (["neuronal", "non_neuronal"] if split_by_fraction else ["all_samples"]):
                    in_subset = np.ones(len(y_region), bool) if SUBSETS[subset] is None else SUBSETS[subset]
                    train_idx = np.flatnonzero((donors != test_donor) & in_subset)
                    test_idx = np.flatnonzero((donors == test_donor) & in_subset)
                    group_preds = {}
                    for macro_name in labels:
                        group_idx = train_idx[y_scheme[train_idx] == macro_name]
                        group_preds[macro_name], chosen, _ = micro_predictions(method, group_idx, test_idx)
                        donor_settings.append({"test_donor": test_donor, "subset": subset,
                                               "macro": macro_name, **chosen})
                    for pos, i in enumerate(test_idx):
                        predicted_macro = macro_pred[subset].loc[i]
                        donor_rows.append({"sample_index": int(i), "test_donor": test_donor,
                                           "fraction": fraction_labels[i], "y_true": y_region[i],
                                           "macro_true": y_scheme[i], "macro_pred": predicted_macro,
                                           "y_pred": group_preds[predicted_macro][pos],
                                           "oracle_micro_pred": group_preds[y_scheme[i]][pos]})
                payload = {"signature": signature, "rows": donor_rows, "settings": donor_settings}
                atomic_pickle(payload, donor_path)
            rows.extend(payload["rows"])
            settings.extend(payload["settings"])

        result = pd.DataFrame(rows).sort_values("sample_index").reset_index(drop=True)
        assert len(result) == len(y_region) and np.array_equal(result["y_true"], y_region)
        atomic_pickle({"signature": signature, "results": result, "settings": pd.DataFrame(settings)},
                      checkpoint_dir / "complete.pkl")
        save(result, f"hierarchical_grid/{name}_predictions.csv")
        save(pd.DataFrame(settings), f"hierarchical_grid/{name}_chosen_settings.csv")
        return result


    def run_all_for(method):
        run_direct_and_macro(method)
        for scheme in MACRO_SCHEMES:
            for split in [False, True]:
                variant = "fraction_specific" if split else "all_samples"
                grid_results[(method, f"Hierarchical ({scheme})", variant)] = run_hierarchical_grid(method, scheme, split)
        print(f"{method}: all experiments done")

    # PCA: direct 25 regions, 4/7 macro (all, neuronal, non-neuronal), hierarchical 4→25 and 7→25
    run_all_for("PCA")

    # ANOVA: direct 25 regions, 4/7 macro (all, neuronal, non-neuronal), hierarchical 4→25 and 7→25
    run_all_for("ANOVA")

    # MI: direct 25 regions, 4/7 macro (all, neuronal, non-neuronal), hierarchical 4→25 and 7→25
    run_all_for("MI")

    # ANOVA genes + RBF-kernel SVM: same schemes, (k, C, gamma) tuned by nested LODO
    run_all_for("ANOVA-RBF")

    # PCA components + RBF-kernel SVM: same schemes, (k, C, gamma) tuned by nested LODO
    run_all_for("PCA-RBF")

    # MI genes + RBF-kernel SVM: same schemes, (k, C, gamma) tuned by nested LODO.
    # All MI scores are cached in resume_checkpoints/mi_scores/, so MI is never computed twice.
    run_all_for("MI-RBF")

    # ------------------------------------------------------------
    # Summary grid: macro F1, mean ± SD over the 6 held-out donors (nested LODO everywhere)
    # ------------------------------------------------------------
    def per_donor_macro_f1(result):
        """Per-donor macro F1, for LODO results (fold table) and hierarchical predictions alike."""
        if isinstance(result, dict):
            return result["fold_results"]["macro_f1"]
        return result.groupby("test_donor").apply(
            lambda d: score(d["y_true"], d["y_pred"])["macro_f1"], include_groups=False)

    SCHEME_ORDER = ["25 regions", "Hierarchical (4 macro)", "Hierarchical (7 macro)", "4 macro", "7 macro"]
    summary_rows = []
    for (method, scheme, variant), result in grid_results.items():
        f1 = per_donor_macro_f1(result)
        preds = best_predictions(result) if isinstance(result, dict) else result
        summary_rows.append({"scheme": scheme, "variant": variant, "method": method,
                             "macro_f1_mean": f1.mean(), "macro_f1_std": f1.std(),
                             "pooled_accuracy": (preds["y_true"] == preds["y_pred"]).mean()})
    full_grid = pd.DataFrame(summary_rows)
    full_grid["cell"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(full_grid["macro_f1_mean"], full_grid["macro_f1_std"])]
    table = (full_grid.pivot(index=["scheme", "variant"], columns="method", values="cell")
             .reindex(columns=["ANOVA", "ANOVA-RBF", "MI", "MI-RBF", "PCA", "PCA-RBF"])
             .sort_index(key=lambda idx: idx.map(lambda v: SCHEME_ORDER.index(v) if v in SCHEME_ORDER
                                                 else ["all_samples", "neuronal", "non_neuronal",
                                                       "fraction_specific"].index(v))))
    print("Macro F1 (mean ± SD over held-out donors). Rows 25 regions / Hierarchical are scored on the "
          "25 regions; 4 / 7 macro rows are scored on their own classes (not comparable across rows).")
    display(table)
    save(full_grid, "full_grid_summary.csv")
    return full_grid


def run_pca_variance_search(project_dir, data_file, results_dir):
    """Tune PCA explained-variance targets using nested donor splits."""
    _enter_project(project_dir)
    DATA_FILE = Path(data_file)
    RESULTS_DIR = Path(results_dir)
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC, SVC
    RANDOM_STATE = 42

    # PCA variance search

    # Names used by the variance-search functions are kept local to this section.
    from sklearn.svm import SVC
    OUT = RESULTS_DIR / "pca_variance"
    OUT.mkdir(parents=True, exist_ok=True)
    VARIANCE_TARGETS = (0.50, 0.70, 0.90, 0.95, 0.99)
    LINEAR_C = (0.1, 1.0, 10.0)
    RBF_C = (1.0, 10.0, 100.0)
    GAMMA_FACTORS = (0.1, 1.0, 10.0)
    df = pd.read_csv(DATA_FILE, index_col=0)
    X = df.T.to_numpy(dtype=np.float32)
    names = df.columns.to_numpy()
    parts = [name.split("_") for name in names]
    donors = np.array([p[0] for p in parts])
    y = np.array([p[1] for p in parts])

    def pca_features(X_train, X_test, targets=VARIANCE_TARGETS):
        """One PCA fit per split; return matrices and actual PC counts for all targets."""
        gene_scale = StandardScaler().fit(X_train)
        train_scaled = gene_scale.transform(X_train)
        test_scaled = gene_scale.transform(X_test)

        # n_train - 1 is the maximum rank after centering; full SVD gives exact
        # cumulative variance, including the 99% cut-off.
        pca = PCA(n_components=min(len(X_train) - 1, X_train.shape[1]),
                  svd_solver='full', random_state=RANDOM_STATE).fit(train_scaled)
        z_train = pca.transform(train_scaled)
        z_test = pca.transform(test_scaled)
        cumulative = np.cumsum(pca.explained_variance_ratio_)

        features = {}
        for target in targets:
            k = min(int(np.searchsorted(cumulative, target) + 1), z_train.shape[1])
            # The RBF distance and linear penalty should not be dominated by PC1.
            pc_scale = StandardScaler().fit(z_train[:, :k])
            features[target] = (pc_scale.transform(z_train[:, :k]),
                                pc_scale.transform(z_test[:, :k]), k,
                                float(cumulative[k - 1]))
        return features


    def make_model(kernel, C, gamma_factor=None):
        if kernel == 'linear':
            return LinearSVC(C=C, class_weight='balanced', dual='auto',
                             max_iter=10000, random_state=RANDOM_STATE)
        # After PC standardization, gamma='scale' is approximately 1 / k.
        return SVC(kernel='rbf', C=C, gamma=gamma_factor,
                   class_weight='balanced')


    def settings(kernel):
        if kernel == 'linear':
            return [(t, C, None) for t in VARIANCE_TARGETS for C in LINEAR_C]
        return [(t, C, g) for t in VARIANCE_TARGETS
                for C in RBF_C for g in GAMMA_FACTORS]


    def metrics(y_true, y_pred):
        # Match the earlier project notebook: macro F1 over classes present in
        # this held-out donor. The pooled score is reported separately below.
        return {'macro_f1': f1_score(y_true, y_pred, labels=np.unique(y_true),
                                     average='macro', zero_division=0),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'accuracy': accuracy_score(y_true, y_pred)}

    def candidate_scores(X_train, y_train, donors_train, kernel):
        """Evaluate all settings using only the five outer-training donors."""
        rows = []
        for inner_train, inner_val in LeaveOneGroupOut().split(X_train, y_train, donors_train):
            features = pca_features(X_train[inner_train], X_train[inner_val])
            for target, C, gamma_factor in settings(kernel):
                a, b, n_pc, actual_variance = features[target]
                gamma = None if kernel == 'linear' else gamma_factor / a.shape[1]
                model = make_model(kernel, C, gamma)
                pred = model.fit(a, y_train[inner_train]).predict(b)
                rows.append({'inner_donor': donors_train[inner_val][0],
                             'target': target, 'C': C,
                             'gamma_factor': gamma_factor if gamma_factor is not None else 0.0,
                             'n_pc': n_pc, 'actual_variance': actual_variance,
                             **metrics(y_train[inner_val], pred)})
        return pd.DataFrame(rows)


    def choose_setting(scores):
        grouped = (scores.groupby(['target', 'C', 'gamma_factor'])
                   [['macro_f1', 'balanced_accuracy', 'accuracy']].mean().reset_index())
        ranked = grouped.sort_values(
            ['macro_f1', 'balanced_accuracy', 'accuracy', 'target', 'C', 'gamma_factor'],
            ascending=[False, False, False, True, True, True])
        return ranked.iloc[0], ranked


    def evaluate_kernel(kernel):
        folds, predictions, choices = [], [], []
        for outer_train, outer_test in LeaveOneGroupOut().split(X, y, donors):
            donor = donors[outer_test][0]
            print(f'{kernel}: held-out donor {donor}', flush=True)
            inner = candidate_scores(X[outer_train], y[outer_train], donors[outer_train], kernel)
            winner, ranked = choose_setting(inner)
            choices.append({'test_donor': donor, **winner.to_dict()})

            target, C, factor = winner['target'], winner['C'], winner['gamma_factor']
            a, b, n_pc, actual_variance = pca_features(
                X[outer_train], X[outer_test], targets=(target,))[target]
            gamma = None if kernel == 'linear' else factor / a.shape[1]
            pred = make_model(kernel, C, gamma).fit(a, y[outer_train]).predict(b)
            folds.append({'kernel': kernel, 'test_donor': donor, 'target': target,
                          'C': C, 'gamma_factor': factor, 'n_pc': n_pc,
                          'actual_variance': actual_variance, **metrics(y[outer_test], pred)})
            predictions.extend({'kernel': kernel, 'test_donor': donor,
                                'sample_name': names[i], 'y_true': y[i], 'y_pred': label}
                               for i, label in zip(outer_test, pred))
            # Save completed folds so results remain available during a long run.
            pd.DataFrame(folds).to_csv(OUT / f'{kernel}_folds.csv', index=False)
            pd.DataFrame(predictions).to_csv(OUT / f'{kernel}_predictions.csv', index=False)
            pd.DataFrame(choices).to_csv(OUT / f'{kernel}_choices.csv', index=False)
        return pd.DataFrame(folds), pd.DataFrame(predictions), pd.DataFrame(choices)

    # Skip the expensive search when both complete six-donor outputs exist.
    def load_or_evaluate(kernel):
        paths = [OUT / f"{kernel}_{part}.csv" for part in ("folds", "predictions", "choices")]
        if all(path.exists() for path in paths):
            tables = [pd.read_csv(path) for path in paths]
            if len(tables[0]) == 6 and tables[0].test_donor.nunique() == 6:
                print(f"Loaded completed {kernel} PCA variance search")
                return tables
        return evaluate_kernel(kernel)

    linear_folds, linear_pred, linear_choices = load_or_evaluate("linear")
    rbf_folds, rbf_pred, rbf_choices = load_or_evaluate("rbf")

    folds = pd.concat([linear_folds, rbf_folds], ignore_index=True)
    predictions = pd.concat([linear_pred, rbf_pred], ignore_index=True)
    choices = pd.concat([linear_choices, rbf_choices], ignore_index=True)

    summary = (folds.groupby('kernel')
               .agg(macro_f1_mean=('macro_f1', 'mean'), macro_f1_sd=('macro_f1', 'std'),
                    accuracy_mean=('accuracy', 'mean'), accuracy_sd=('accuracy', 'std'),
                    n_pc_min=('n_pc', 'min'), n_pc_max=('n_pc', 'max'))
               .reset_index())
    summary['pooled_accuracy'] = [accuracy_score(d.y_true, d.y_pred)
                                  for _, d in predictions.groupby('kernel')]
    print(summary.round(3).to_string(index=False))
    print('\nChosen variance targets by held-out donor:')
    print(folds.pivot(index='test_donor', columns='kernel', values='target').to_string())
    print('\nChosen number of components by held-out donor:')
    print(folds.pivot(index='test_donor', columns='kernel', values='n_pc').to_string())
    summary.to_csv(OUT / 'summary.csv', index=False)
    folds.to_csv(OUT / 'all_folds.csv', index=False)
    return summary


def run_fixed_comparisons(project_dir, data_file, results_dir, checkpoint_dir):
    """Run or reuse all-sample joint and cell-specific fixed-setting models."""
    _enter_project(project_dir)
    results_dir = Path(results_dir)
    target = results_dir / "joint_feature_comparison"

    def complete(filename, rows):
        path = target / filename
        return path.exists() and len(pd.read_csv(path)) == rows

    import joint_feature_comparison as joint
    import cell_type_comparison as cell
    joint.configure_paths(data_file, results_dir, checkpoint_dir)
    cell.configure_paths(data_file, results_dir)

    if not complete("summary.csv", 12):
        joint.main()
    if not complete("cell_type_summary.csv", 24):
        cell.run_joint_cell_types()
    if not complete("direct_cell_type_summary.csv", 12):
        cell.run_direct_cell_types()
    if not complete("all_cell_type_family_summary.csv", 84):
        cell.summarize_cell_type_families()
    return pd.read_csv(target / "summary.csv")


def save_comparison_tables(project_dir, data_file, results_dir):
    """Aggregate held-out predictions and check the saved comparison files."""
    from pathlib import Path
    _enter_project(project_dir)
    FILE_PATH = Path(data_file)
    RESULTS_DIR = Path(results_dir)

    # Tables and validation

    """Summarize saved 25-region predictions by cell type and held-out donor."""
    from pathlib import Path
    import warnings

    import numpy as np
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, f1_score

    result_dir = RESULTS_DIR
    model_files = {
        "Direct 25 regions (ANOVA + RBF)":
            (result_dir / "rbf_anova_25_regions_nested/predictions.csv", None),
        "Joint 4 macro + 25 regions (ANOVA + RBF)":
            (result_dir / "joint_feature_comparison/predictions.csv", "4 macro"),
        "Joint 7 macro + 25 regions (ANOVA + RBF)":
            (result_dir / "joint_feature_comparison/predictions.csv", "7 macro"),
    }

    # Sample indices in the prediction files follow the original CSV column order.
    sample_names = pd.read_csv(FILE_PATH, nrows=0, index_col=0).columns
    cell_type_by_sample = pd.Series(
        ["Neuronal" if name.split("_")[2] == "N" else "Non-neuronal" for name in sample_names],
        index=range(len(sample_names)),
    )

    rows = []
    for model_name, (file, hierarchy) in model_files.items():
        predictions = pd.read_csv(file)
        if hierarchy is not None:
            predictions = predictions[
                (predictions.hierarchy == hierarchy)
                & (predictions.features == "ANOVA")
                & (predictions.kernel == "RBF")
            ].copy()
            predictions = predictions.rename(columns={"held_out_donor": "test_donor"})
        assert len(predictions) == len(sample_names) and predictions.sample_index.is_unique
        predictions["cell_type"] = predictions.sample_index.map(cell_type_by_sample)

        for cell_type in ["Neuronal", "Non-neuronal"]:
            group = predictions[predictions.cell_type == cell_type]
            donor_scores = []
            for donor, held_out in group.groupby("test_donor"):
                # Use the regions actually represented for this donor and cell type,
                # as in the notebook's other donor-wise macro-F1 summaries.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                    donor_scores.append({
                        "test_donor": donor,
                        "accuracy": (held_out.y_true == held_out.y_pred).mean(),
                        "balanced_accuracy": balanced_accuracy_score(held_out.y_true, held_out.y_pred),
                        "macro_f1": f1_score(held_out.y_true, held_out.y_pred,
                                              labels=np.unique(held_out.y_true),
                                              average="macro", zero_division=0),
                    })
            donor_scores = pd.DataFrame(donor_scores)
            assert len(donor_scores) == 6
            row = {"cell_type": cell_type, "model": model_name,
                   "n_samples": len(group), "n_donors": len(donor_scores)}
            for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
                row[f"{metric}_mean"] = donor_scores[metric].mean()
                row[f"{metric}_sd"] = donor_scores[metric].std(ddof=1)
            rows.append(row)

    joint_by_cell_type = pd.DataFrame(rows)
    joint_by_cell_type.to_csv(result_dir / "by_cell_type_direct_vs_joint.csv", index=False)

    # Display one compact mean ± SD cell per metric.
    table = joint_by_cell_type[["cell_type", "model", "n_samples"]].copy()
    for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
        table[metric] = [f"{mean:.3f} ± {sd:.3f}" for mean, sd in
                         zip(joint_by_cell_type[f"{metric}_mean"], joint_by_cell_type[f"{metric}_sd"])]
    print("Mean ± SD across six held-out donors; best ANOVA + RBF run in each model family.")
    display(table)

    from pathlib import Path
    import pandas as pd

    joint_comparison = pd.read_csv(RESULTS_DIR / "joint_feature_comparison/summary.csv")
    joint_comparison["model"] = joint_comparison["features"] + " + " + joint_comparison["kernel"] + " joint SVM"
    joint_comparison["macro_f1"] = [f"{m:.3f} ± {s:.3f}" for m, s in
                                    zip(joint_comparison.macro_f1_mean, joint_comparison.macro_f1_sd)]
    joint_comparison["balanced_accuracy"] = [f"{m:.3f} ± {s:.3f}" for m, s in
                                             zip(joint_comparison.balanced_accuracy_mean,
                                                 joint_comparison.balanced_accuracy_sd)]
    display(joint_comparison[["hierarchy", "model", "macro_f1", "balanced_accuracy"]])

    from pathlib import Path
    import pandas as pd

    joint_cell_summary = pd.read_csv(RESULTS_DIR / "joint_feature_comparison/cell_type_summary.csv")
    joint_cell_summary["model"] = joint_cell_summary.features + " + " + joint_cell_summary.kernel + " joint SVM"
    joint_cell_summary["macro_f1"] = [f"{m:.3f} ± {s:.3f}" for m, s in
                                       zip(joint_cell_summary.macro_f1_mean,
                                           joint_cell_summary.macro_f1_sd)]
    joint_cell_summary["balanced_accuracy"] = [f"{m:.3f} ± {s:.3f}" for m, s in
                                                zip(joint_cell_summary.balanced_accuracy_mean,
                                                    joint_cell_summary.balanced_accuracy_sd)]
    display(joint_cell_summary[["cell_type", "hierarchy", "model", "macro_f1", "balanced_accuracy"]])

    from pathlib import Path
    import pandas as pd

    comparison = pd.read_csv(RESULTS_DIR / "joint_feature_comparison/all_cell_type_family_summary.csv")
    family_order = ["Direct 25", "Separate 4→25", "Separate 7→25",
                    "Joint 4→25", "Joint 7→25", "Macro 4", "Macro 7"]
    model_order = ["ANOVA", "ANOVA-RBF", "MI", "MI-RBF", "PCA", "PCA-RBF"]
    for cell_type in ["Neuronal", "Non-neuronal"]:
        rows = comparison[comparison.cell_type == cell_type].copy()
        rows["macro_f1"] = [f"{mean:.3f} ± {sd:.3f}" for mean, sd in
                            zip(rows.macro_f1_mean, rows.macro_f1_sd)]
        table = rows.pivot(index="model", columns="family", values="macro_f1")
        table = table.reindex(index=model_order, columns=family_order)
        print(f"{cell_type}: macro F1, mean ± SD across held-out donors")
        display(table)

    required = [
        RESULTS_DIR / "full_grid_summary.csv",
        RESULTS_DIR / "joint_feature_comparison/summary.csv",
        RESULTS_DIR / "joint_feature_comparison/cell_type_summary.csv",
        RESULTS_DIR / "joint_feature_comparison/direct_cell_type_summary.csv",
        RESULTS_DIR / "joint_feature_comparison/all_cell_type_family_summary.csv",
        RESULTS_DIR / "rbf_anova_25_regions_nested/predictions.csv",
    ]
    missing = [path for path in required if not path.exists()]
    assert not missing, f"Missing results: {missing}"
    print(f"Checked {len(required)} key result files. Plotting notebook can now be run.")
    return pd.read_csv(RESULTS_DIR / "joint_feature_comparison/all_cell_type_family_summary.csv")
