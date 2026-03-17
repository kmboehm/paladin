import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import wandb
from PIL import Image
from scipy.special import expit
from scipy.stats import BootstrapMethod, pearsonr
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error, precision_recall_curve, roc_auc_score
from statsmodels.stats.multitest import multipletests

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 6


def get_pearson_ci(target, logit):
    pearson = pearsonr(target, logit, alternative="greater")
    pearson_lower, pearson_upper = pearson.confidence_interval(
        confidence_level=0.95, method=BootstrapMethod(n_resamples=1000, method="basic")
    )
    return pearson.statistic, pearson_lower, pearson_upper, pearson.pvalue


def get_auroc_ci(target, logit):
    try:
        auroc = roc_auc_score(target, logit)
    except Exception as e:
        print(e)
        return np.nan, np.nan, np.nan
    auroc_lower, auroc_upper = bootstrap_auroc_ci(target, logit, n_bootstraps=1000)
    return auroc, auroc_lower, auroc_upper


def bootstrap_auroc_ci(target, logit, n_bootstraps=1000, **kwargs):
    auroc_values = []
    for _ in range(n_bootstraps):
        bootstrap_indices = np.random.choice(np.arange(len(target)), len(target), replace=True)
        bootstrap_target = target.iloc[bootstrap_indices]
        bootstrap_logit = logit.iloc[bootstrap_indices]
        try:
            auroc_values.append(roc_auc_score(bootstrap_target, bootstrap_logit, **kwargs))
        except ValueError:
            continue
    if len(auroc_values) < 0.9 * n_bootstraps:
        print("Warning: fewer than 90 percent of bootstraps converged. Returning nan for AUROC CI.")
        return np.nan, np.nan
    return np.percentile(auroc_values, [100 * 0.025, 100 * 0.975])


def get_recall_precision(target, logit):
    precision_threshold = 0.95
    precision, recall, _ = precision_recall_curve(target, logit)
    index = np.argmax(precision >= precision_threshold)
    return recall[index], precision[index]


def get_binary_single_class_classification_metrics(target, logit, histology):
    auroc, auroc_lower, auroc_upper = get_auroc_ci(target, logit)
    # print(target.shape)
    # print(target)
    # print(logit.shape)
    # print(logit)
    # recall, precision = get_recall_precision(target, logit)
    return {
        "histology": histology,
        "auroc": auroc,
        "auroc_lower_95": auroc_lower,
        "auroc_upper_95": auroc_upper,
        # "recall_at_best_operating_point": recall,
        # "precision_at_best_operating_point": precision,
        "n": len(target),
        "n_positive": target.sum(),
        "n_negative": len(target) - target.sum(),
    }


def create_binary_auroc_table(test_df):
    results = []
    for histology, single_hist_single_target_df in test_df.groupby("histology"):
        results.append(
            get_binary_single_class_classification_metrics(
                single_hist_single_target_df["target"], single_hist_single_target_df["logit"], histology
            )
        )
    unique_histologies = test_df["histology"].unique()
    for histology in unique_histologies:
        if "-" in histology and "0" in histology:
            root_histology = histology.split("-")[0]
            root_histology_df = test_df[test_df["histology"].str.split("-").str[0] == root_histology]
            results.append(
                get_binary_single_class_classification_metrics(
                    root_histology_df["target"], root_histology_df["logit"], root_histology
                )
            )
    results = pd.DataFrame(results)
    results = results.sort_values(by="auroc", ascending=False)

    artifact = wandb.Artifact(name="test_auroc", type="auroc")
    table = wandb.Table(dataframe=results)
    artifact.add(table, "test_auroc")
    return {"table": table, "artifact": artifact, "df": results}


def plot_calibration_curve(test_df):
    palette = sns.color_palette("colorblind", 3)

    if len(test_df.histology.unique()) > 1:
        print("Calibration curve can only be plotted for a single histology")
        return None
    else:
        fig = plt.figure(figsize=(2, 2))
        if test_df["prob"].max() > 1.0 or test_df["prob"].min() < 0.0:
            test_prob = expit(test_df["prob"])
            print("Probabilities are not between 0 and 1, so we are using the sigmoid function to transform them")
        else:
            test_prob = test_df["prob"]

        prob_true, prob_pred = calibration_curve(test_df["target"], test_prob, n_bins=5, strategy="uniform")
        if (
            np.isnan(prob_true).sum() > 0  # noqa
            or np.isnan(prob_pred).sum() > 0  # noqa
            or np.isinf(prob_true).sum() > 0  # noqa
            or np.isinf(prob_pred).sum() > 0  # noqa
        ):
            print("nan or inf in prob_true or prob_pred")
            print(prob_true)
            print(prob_pred)

        bootstrapped_lowess_y = []
        bootstrapped_lowess_x = np.linspace(min(prob_pred), max(prob_pred), 20)
        for _ in range(1000):
            bootstrap_indices = np.random.choice(np.arange(len(test_df)), len(test_df), replace=True)
            bootstrap_prob_true, bootstrap_prob_pred = calibration_curve(
                test_df["target"].iloc[bootstrap_indices],
                test_prob.iloc[bootstrap_indices],
                n_bins=5,
                strategy="quantile",
            )
            if np.isnan(bootstrap_prob_true).sum() > 0 or np.isnan(bootstrap_prob_pred).sum() > 0:
                # print("nan or inf in bootstrap_prob_true or bootstrap_prob_pred")
                # print(bootstrap_prob_true)
                # print(bootstrap_prob_pred)
                continue
            _single_bootstrap_lowess_y = sm.nonparametric.lowess(
                bootstrap_prob_true, bootstrap_prob_pred, frac=0.66, xvals=bootstrapped_lowess_x
            )
            if np.isnan(_single_bootstrap_lowess_y).sum() > 0 or np.isinf(_single_bootstrap_lowess_y).sum() > 0:
                # print("nan or inf in _single_bootstrap_lowess_y")
                # print(_single_bootstrap_lowess_y)
                continue
            bootstrapped_lowess_y.append(_single_bootstrap_lowess_y)

        bootstrapped_lowess_y = np.array(bootstrapped_lowess_y)
        # print(np.isnan(bootstrapped_lowess_y).sum())
        # print(np.isinf(bootstrapped_lowess_y).sum())
        # print('-'*30)
        print(bootstrapped_lowess_y.shape)
        if len(bootstrapped_lowess_y) == 0:
            return None
        lowess_lower = np.percentile(bootstrapped_lowess_y, 2.5, axis=0)
        lowess_upper = np.percentile(bootstrapped_lowess_y, 97.5, axis=0)
        # print(np.isnan(lowess_lower).sum())

        fig = plt.figure(figsize=(3, 3))
        plt.plot(
            bootstrapped_lowess_x,
            bootstrapped_lowess_x,
            label="perfect calibration",
            color="black",
            linestyle=":",
            alpha=0.5,
        )
        uniform_prob_true, uniform_prob_pred = calibration_curve(
            test_df["target"], test_df["prob"], n_bins=10, strategy="uniform"
        )
        plt.plot(
            bootstrapped_lowess_x,
            sm.nonparametric.lowess(uniform_prob_true, uniform_prob_pred, frac=0.5, xvals=bootstrapped_lowess_x),
            color=palette[1],
        )
        plt.fill_between(
            x=bootstrapped_lowess_x,
            y1=lowess_lower,
            y2=lowess_upper,
            alpha=0.2,
            color=palette[1],
            interpolate=True,
        )
        plt.plot(prob_pred, prob_true, "+", label="classifier", color="black")

        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)

        plt.xlabel("Inferred probability")
        plt.ylabel("Probability")
        plt.ylim(-0.22, 1)
        plt.xlim(0, 1)
        plt.title("")

        # add histogram of the probabilities along the y = -0.1 linea
        ax1 = plt.gca()
        ax2 = plt.gca().twinx()
        # Plot histogram for negative class (target=0) going down
        neg_hist, bins = np.histogram(test_prob[test_df["target"] == 0], bins=50, density=True)
        pos_hist, _ = np.histogram(test_prob[test_df["target"] == 1], bins=bins, density=True)

        # Scale down the histogram height and position it at y=-0.1
        # Calculate scaling factor to ensure max bar height is 0.1
        max_hist_height = max(np.max(neg_hist), np.max(pos_hist))
        scaling_factor = 0.1 / max_hist_height if max_hist_height > 0 else 0.02
        neg_hist_scaled = neg_hist * scaling_factor

        ax2.bar(
            bins[:-1],
            -neg_hist_scaled,
            width=np.diff(bins),
            alpha=1.0,
            color=palette[2],
            label="Negative class",
            align="edge",
            bottom=-0.1,
        )

        # Plot histogram for positive class (target=1) going up
        # Scale down the histogram height
        pos_hist_scaled = pos_hist * scaling_factor
        ax2.bar(
            bins[:-1],
            pos_hist_scaled,
            width=np.diff(bins),
            alpha=1.0,
            color=palette[1],
            label="Positive class",
            align="edge",
            bottom=-0.1,
        )

        ax2.set_ylim(-0.22, 1)
        # Add text labels for True and False classes
        n_pos = int(test_df["target"].sum())
        n_neg = int(len(test_df) - test_df["target"].sum())
        ax2.text(
            np.max(bins),
            np.max(pos_hist_scaled) - 0.1,
            f"1 (n={n_pos})",
            verticalalignment="bottom",
            horizontalalignment="left",
            fontsize=6,
        )
        ax2.text(
            np.max(bins),
            -np.max(neg_hist_scaled) - 0.1,
            f"0 (n={n_neg})",
            verticalalignment="bottom",
            horizontalalignment="left",
            fontsize=6,
        )
        # Remove y-axis labels and ticks
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        # Set y-ticks between 0 and 1, excluding -0.2
        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # ax2.set_yticklabels([])  # Keep labels hidden

        # Add legend for the histograms
        plt.tight_layout()
        plt.savefig("test.png", dpi=300)

        # add a smoothed line considering every point using splines
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        image = Image.open(buf)
        image = wandb.Image(image)
        buf.close()
        plt.close()
        return image


def net_benefit(ground_truth, probability, threshold):
    assert threshold >= 0 and threshold <= 1, "Threshold must be between 0 and 1"
    assert ground_truth.isin([0, 1]).all(), "Ground truth must be binary"
    tp = ((probability >= threshold) & (ground_truth == 1)).sum()
    fp = ((probability >= threshold) & (ground_truth == 0)).sum()
    tn = ((probability < threshold) & (ground_truth == 0)).sum()
    fn = ((probability < threshold) & (ground_truth == 1)).sum()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    phi = ground_truth.sum() / float(len(ground_truth))
    benefit = (sens * phi) - ((1 - spec) * (1 - phi) * (threshold / (1 - threshold)))
    return benefit


def treat_all_benefit(ground_truth, threshold):
    assert threshold >= 0 and threshold <= 1, "Threshold must be between 0 and 1"
    assert ground_truth.isin([0, 1]).all(), "Ground truth must be binary"
    phi = ground_truth.sum() / float(len(ground_truth))
    return phi - ((1 - phi) * (threshold / (1 - threshold)))


def plot_decision_curve(test_df):
    palette = sns.diverging_palette(250, 30, l=65, center="dark")
    if len(test_df.histology.unique()) > 1:
        print("Decision curve can only be plotted for a single histology")
        return None
    thresholds = np.linspace(0.01, 0.99, 99)
    benefits = [net_benefit(test_df["target"], test_df["prob"], threshold) for threshold in thresholds]
    treat_all_benefits = [treat_all_benefit(test_df["target"], threshold) for threshold in thresholds]
    treat_none_benefits = [0] * len(thresholds)
    approach = ["Treat all"] * len(thresholds) + ["Treat none"] * len(thresholds) + ["Model"] * len(thresholds)
    benefit = np.concatenate([treat_all_benefits, treat_none_benefits, benefits])
    thresholds = np.concatenate([thresholds, thresholds, thresholds])
    df = pd.DataFrame({"threshold": thresholds, "approach": approach, "benefit": benefit})
    fig = plt.figure(figsize=(3, 2))
    plt.plot(
        df[df["approach"] == "Treat all"]["threshold"],
        df[df["approach"] == "Treat all"]["benefit"],
        label="treat all",
        color=palette[0],
    )
    plt.plot(
        df[df["approach"] == "Treat none"]["threshold"],
        df[df["approach"] == "Treat none"]["benefit"],
        label="treat none",
        color="black",
        alpha=0.5,
        linestyle="--",
    )
    plt.plot(
        df[df["approach"] == "Model"]["threshold"],
        df[df["approach"] == "Model"]["benefit"],
        label="model",
        color=palette[1],
    )
    # remove title from legend
    plt.legend(title="", loc="best", fontsize=6)
    plt.xlabel("Threshold")
    plt.ylabel("Net benefit")
    plt.ylim(-0.2, max(df["benefit"].max() * 1.1, 0.2))
    plt.xlim(0, 1)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig("test_decision.png", dpi=300)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    image = Image.open(buf)
    image = wandb.Image(image)
    buf.close()
    plt.close()
    return image


def create_pearson_table(test_df):
    results = []
    for histology, single_hist_single_target_df in test_df.groupby("histology"):
        mse = mean_squared_error(single_hist_single_target_df["target"], single_hist_single_target_df["logit"])
        pearson, pearson_lower, pearson_upper, p_value = get_pearson_ci(
            single_hist_single_target_df["target"], single_hist_single_target_df["logit"]
        )
        results.append(
            {
                "histology": histology,
                "mse": mse,
                "pearson": pearson,
                "pearson_lower": pearson_lower,
                "pearson_upper": pearson_upper,
                "p_value": p_value,
                "n": len(single_hist_single_target_df),
            }
        )
        results = pd.DataFrame(results)
        results["q_value"] = multipletests(results["p_value"], method="fdr_bh")[1]
        results = results.sort_values(by="pearson", ascending=False)
        artifact = wandb.Artifact(name="test_pearson", type="pearson")
        table = wandb.Table(dataframe=results)
        artifact.add(table, "test_pearson")
        return {"table": table, "artifact": artifact, "df": results}


def multiclass_roc_auc_score_with_confidence(target, logit_df, int_to_name_mapping):
    auc_values_by_class = []
    unique_classes = np.unique(target)
    print(f"unique_classes: {unique_classes}")
    for unique_class in unique_classes:
        single_class_target = (target == unique_class).astype(int)
        single_class_logit = logit_df[int_to_name_mapping[unique_class]]
        auc_values_by_class.append(
            get_binary_single_class_classification_metrics(single_class_target, single_class_logit, unique_class)
        )
    print(f"auc_values_by_class: {auc_values_by_class}")
    auc_values_by_class = pd.DataFrame(auc_values_by_class)
    print(f"auc_values_by_class df: {auc_values_by_class}")
    auc_values_by_class = auc_values_by_class.sort_values(by="auroc", ascending=False)
    try:
        macro_ovr_auroc = roc_auc_score(target, logit_df, multi_class="ovr", average="macro")
        macro_ovr_auroc_lower, macro_ovr_auroc_upper = bootstrap_auroc_ci(
            target, logit_df, multi_class="ovr", average="macro"
        )
        macro_ovr_auroc_metrics = {
            "auroc": macro_ovr_auroc,
            "auroc_lower_95": macro_ovr_auroc_lower,
            "auroc_upper_95": macro_ovr_auroc_upper,
            "n": len(target),
        }
        macro_ovo_auroc = roc_auc_score(target, logit_df, multi_class="ovo", average="macro")
        macro_ovo_auroc_lower, macro_ovo_auroc_upper = bootstrap_auroc_ci(
            target, logit_df, multi_class="ovo", average="macro"
        )
        macro_ovo_auroc_metrics = {
            "auroc": macro_ovo_auroc,
            "auroc_lower_95": macro_ovo_auroc_lower,
            "auroc_upper_95": macro_ovo_auroc_upper,
            "n": len(target),
        }
        auc_values_by_class = pd.concat(
            [pd.DataFrame([macro_ovr_auroc_metrics, macro_ovo_auroc_metrics]), auc_values_by_class]
        )
    except ValueError:
        print("Could not calculate macro AUROC")
    return auc_values_by_class


if __name__ == "__main__":
    df = pd.read_csv("wandb_export_2025-02-21T08_33_58.319-05_00.csv")
    df = df[df.split == "test"]
    # df["prob"] = np.clip(df["prob"], 0.01, 0.99)
    plot_calibration_curve(df)
    plot_decision_curve(df)
