import sys
sys.path.append("/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/RQ3_SYNTHETIC_DATA_GENERATION/neighboring")

import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from neighboring import find_neighbors

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ----------------------------------------------------------------------
# Pfade
# ----------------------------------------------------------------------
SYNTHETIC_FILE = Path(
    "/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/OUTPUT/10_synthetic_transcripts_long.csv"
)

FEATURES_FILE = Path(
    "/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/INPUT/DAIC_META.csv"
)

TRANSCRIPT_FILE = Path(
    "/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/INPUT/DAIC_FULL_NO_TAGS.csv"
)

OUTPUT_DIR = Path(
    "/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/OUTPUT/synthetic_neighbor_eval"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Konstanten
# ----------------------------------------------------------------------
FEATURE_COLUMNS = [
    "PHQ8_Concentrating",
    "PHQ8_Appetite",
    "PHQ8_Depressed",
    "PHQ8_Tired",
    "PHQ8_NoInterest",
    "PHQ8_Failure",
    "PHQ8_Moving",
    "PHQ8_Sleep",
]

TARGET_COLUMNS = [f"target_{col}" for col in FEATURE_COLUMNS]

N_NEIGHBORS = 5
SHORT_TURN_WORD_THRESHOLD = 3
PHQ_CUTOFF = 14


# ----------------------------------------------------------------------
# Daten laden
# ----------------------------------------------------------------------
synthetic_df = pd.read_csv(SYNTHETIC_FILE)
features_df = pd.read_csv(FEATURES_FILE)
transcript_df = pd.read_csv(TRANSCRIPT_FILE)

features_df["Participant_ID"] = pd.to_numeric(features_df["Participant_ID"], errors="coerce")
transcript_df["Participant_ID"] = pd.to_numeric(transcript_df["Participant_ID"], errors="coerce")

features_df = features_df.dropna(subset=["Participant_ID"]).copy()
features_df["Participant_ID"] = features_df["Participant_ID"].astype(int)

transcript_df = transcript_df.dropna(subset=["Participant_ID", "speaker", "value"]).copy()
transcript_df["Participant_ID"] = transcript_df["Participant_ID"].astype(int)
transcript_df["speaker"] = transcript_df["speaker"].astype(str).str.strip()
transcript_df["value"] = transcript_df["value"].astype(str).str.strip()

if "start_time" in transcript_df.columns:
    transcript_df = transcript_df.sort_values(["Participant_ID", "start_time"]).reset_index(drop=True)
else:
    transcript_df = transcript_df.sort_values(["Participant_ID"]).reset_index(drop=True)

synthetic_df["synthetic_id"] = synthetic_df["synthetic_id"].astype(str)
synthetic_df["speaker"] = synthetic_df["speaker"].astype(str).str.strip()
synthetic_df["value"] = synthetic_df["value"].astype(str).str.strip()

for col in TARGET_COLUMNS:
    synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors="coerce")

for col in FEATURE_COLUMNS:
    features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

# ----------------------------------------------------------------------
# PHQ-8 Scores berechnen
# ----------------------------------------------------------------------
synthetic_df["target_PHQ8_Score"] = synthetic_df[TARGET_COLUMNS].sum(axis=1)
synthetic_df["target_PHQ8_binary"] = synthetic_df["target_PHQ8_Score"] >= PHQ_CUTOFF
synthetic_df["target_PHQ8_group"] = np.where(
    synthetic_df["target_PHQ8_binary"],
    f"PHQ >= {PHQ_CUTOFF} ",
    f"PHQ < {PHQ_CUTOFF}"
)

features_df["real_PHQ8_Score_from_items"] = features_df[FEATURE_COLUMNS].sum(axis=1)
features_df["real_PHQ8_binary_from_items"] = features_df["real_PHQ8_Score_from_items"] >= PHQ_CUTOFF
features_df["real_PHQ8_group_from_items"] = np.where(
    features_df["real_PHQ8_binary_from_items"],
    f"PHQ >= {PHQ_CUTOFF}",
    f"PHQ < {PHQ_CUTOFF}"
)

print("Synthetic rows:", synthetic_df.shape)
print("Synthetic transcripts:", synthetic_df["synthetic_id"].nunique())
print("Real transcript rows:", transcript_df.shape)
print("Real participants:", transcript_df["Participant_ID"].nunique())
print()
print("Synthetic PHQ groups:")
print(
    synthetic_df[["synthetic_id", "target_PHQ8_group"]]
    .drop_duplicates()
    ["target_PHQ8_group"]
    .value_counts()
)
print()
print("Real PHQ groups in metadata:")
print(features_df["real_PHQ8_group_from_items"].value_counts())


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------
def normalize_speaker(speaker):
    """
    Vereinheitlicht Speaker Labels für echte und synthetische Daten.
    DAIC hat häufig Ellie, synthetische Daten haben Therapist.
    """
    s = str(speaker).strip().lower()

    if s in ["participant", "patient", "client"]:
        return "Participant"

    if s in ["ellie", "therapist", "interviewer", "clinician"]:
        return "Therapist"

    return str(speaker).strip()


def tokenize(text):
    """
    Einfache Wort-Tokenisierung.
    """
    return re.findall(r"[A-Za-z']+", str(text).lower())


def compute_pattern_features(df, collapse_same_speaker=True):
    """
    Berechnet einfache strukturelle und lexikalische Pattern-Metriken
    für ein Transcript im Long Format.

    Wenn collapse_same_speaker=True, werden direkt aufeinanderfolgende
    Zeilen desselben Speakers zu einem Turn zusammengefasst.
    """
    tmp = df.copy()
    tmp["speaker_norm"] = tmp["speaker"].apply(normalize_speaker)
    tmp["value"] = tmp["value"].fillna("").astype(str).str.strip()

    tmp = tmp[tmp["value"] != ""].copy()

    if collapse_same_speaker and len(tmp) > 0:
        tmp["_speaker_block"] = (
            tmp["speaker_norm"] != tmp["speaker_norm"].shift()
        ).cumsum()

        tmp = (
            tmp
            .groupby("_speaker_block", as_index=False)
            .agg(
                speaker_norm=("speaker_norm", "first"),
                value=("value", lambda x: " ".join(x))
            )
        )

    tmp["tokens"] = tmp["value"].apply(tokenize)
    tmp["n_words"] = tmp["tokens"].apply(len)

    n_turns = len(tmp)

    participant = tmp[tmp["speaker_norm"] == "Participant"]
    therapist = tmp[tmp["speaker_norm"] == "Therapist"]

    n_participant_turns = len(participant)
    n_therapist_turns = len(therapist)

    total_words = int(tmp["n_words"].sum())
    participant_words = int(participant["n_words"].sum())
    therapist_words = int(therapist["n_words"].sum())

    participant_word_counts = participant["n_words"].to_numpy()

    if n_turns > 1:
        speaker_sequence = tmp["speaker_norm"].tolist()
        speaker_changes = sum(
            speaker_sequence[i] != speaker_sequence[i - 1]
            for i in range(1, len(speaker_sequence))
        )
        speaker_alternation_rate = speaker_changes / (n_turns - 1)
    else:
        speaker_alternation_rate = np.nan

    participant_tokens = [
        tok
        for tokens in participant["tokens"]
        for tok in tokens
    ]

    if len(participant_tokens) > 0:
        ttr_participant = len(set(participant_tokens)) / len(participant_tokens)
    else:
        ttr_participant = np.nan

    if n_therapist_turns > 0:
        therapist_question_ratio = therapist["value"].str.strip().str.endswith("?").mean()
    else:
        therapist_question_ratio = np.nan

    if n_participant_turns > 0:
        short_participant_turn_ratio = (
            participant["n_words"] <= SHORT_TURN_WORD_THRESHOLD
        ).mean()
    else:
        short_participant_turn_ratio = np.nan

    features = {
        "n_turns": n_turns,
        "n_participant_turns": n_participant_turns,
        "n_therapist_turns": n_therapist_turns,
        "participant_turn_ratio": n_participant_turns / n_turns if n_turns > 0 else np.nan,

        "total_words": total_words,
        "participant_words": participant_words,
        "therapist_words": therapist_words,
        "participant_word_ratio": participant_words / total_words if total_words > 0 else np.nan,

        "mean_participant_words_per_turn": np.mean(participant_word_counts) if len(participant_word_counts) > 0 else np.nan,
        "median_participant_words_per_turn": np.median(participant_word_counts) if len(participant_word_counts) > 0 else np.nan,
        "median_participant_words": np.median(participant_word_counts) if len(participant_word_counts) > 0 else np.nan,
        "sd_participant_words": np.std(participant_word_counts, ddof=1) if len(participant_word_counts) > 1 else np.nan,

        "short_participant_turn_ratio": short_participant_turn_ratio,
        "ttr_participant": ttr_participant,
        "speaker_alternation_rate": speaker_alternation_rate,
        "therapist_question_ratio": therapist_question_ratio,
    }

    return features


def safe_z_score(value, mean, sd):
    if pd.isna(value) or pd.isna(mean) or pd.isna(sd) or sd == 0:
        return np.nan
    return (value - mean) / sd


def safe_pct_diff(value, mean):
    if pd.isna(value) or pd.isna(mean) or mean == 0:
        return np.nan
    return (value - mean) / mean * 100


def compute_neighbor_distances(neighbor_df, query_point, feature_cols):
    """
    Berechnet euklidische Distanzen zwischen Query-Target und gefundenen Nachbar:innen.
    """
    tmp = neighbor_df.copy()

    for col in feature_cols:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    query_vector = np.array([query_point[col] for col in feature_cols], dtype=float)
    neighbor_matrix = tmp[feature_cols].to_numpy(dtype=float)

    distances = np.sqrt(((neighbor_matrix - query_vector) ** 2).sum(axis=1))
    tmp["neighbor_distance"] = distances

    return tmp


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
comparison_rows = []
neighbor_id_rows = []

synthetic_targets = (
    synthetic_df[
        [
            "synthetic_id",
            *TARGET_COLUMNS,
            "target_PHQ8_Score",
            "target_PHQ8_binary",
            "target_PHQ8_group"
        ]
    ]
    .drop_duplicates("synthetic_id")
    .reset_index(drop=True)
)

for _, syn_row in tqdm(synthetic_targets.iterrows(), total=len(synthetic_targets)):
    synthetic_id = syn_row["synthetic_id"]

    # -----------------------------
    # Target-Dict für Neighbor-Suche
    # -----------------------------
    query_point = {
        feature: syn_row[f"target_{feature}"]
        for feature in FEATURE_COLUMNS
    }

    # -----------------------------
    # Neighbors suchen
    # -----------------------------
    neighbor_df = find_neighbors(
        csv_path=FEATURES_FILE,
        query=query_point,
        feature_cols=FEATURE_COLUMNS,
        k=N_NEIGHBORS
    )

    neighbor_df = compute_neighbor_distances(
        neighbor_df=neighbor_df,
        query_point=query_point,
        feature_cols=FEATURE_COLUMNS
    )

    neighbor_ids = (
        neighbor_df["Participant_ID"]
        .drop_duplicates()
        .astype(int)
        .tolist()
    )

    # Metadaten der gefundenen Nachbar:innen ergänzen
    neighbor_meta = features_df[
        features_df["Participant_ID"].isin(neighbor_ids)
    ].copy()

    mean_neighbor_distance = neighbor_df["neighbor_distance"].mean()
    min_neighbor_distance = neighbor_df["neighbor_distance"].min()
    max_neighbor_distance = neighbor_df["neighbor_distance"].max()

    mean_neighbor_phq8_score = neighbor_meta["real_PHQ8_Score_from_items"].mean()
    n_neighbor_phq_ge_cutoff = int(neighbor_meta["real_PHQ8_binary_from_items"].sum())
    share_neighbor_phq_ge_cutoff = neighbor_meta["real_PHQ8_binary_from_items"].mean()

    neighbor_id_rows.append({
        "synthetic_id": synthetic_id,

        "target_PHQ8_Score": syn_row["target_PHQ8_Score"],
        "target_PHQ8_binary": syn_row["target_PHQ8_binary"],
        "target_PHQ8_group": syn_row["target_PHQ8_group"],

        "neighbor_ids": json.dumps(neighbor_ids),

        "mean_neighbor_distance": mean_neighbor_distance,
        "min_neighbor_distance": min_neighbor_distance,
        "max_neighbor_distance": max_neighbor_distance,

        "mean_neighbor_PHQ8_Score": mean_neighbor_phq8_score,
        "n_neighbor_PHQ_ge_cutoff": n_neighbor_phq_ge_cutoff,
        "share_neighbor_PHQ_ge_cutoff": share_neighbor_phq_ge_cutoff,
    })

        # -----------------------------
    # Synthetische Pattern berechnen
    # -----------------------------
    syn_transcript = synthetic_df[synthetic_df["synthetic_id"] == synthetic_id]

    syn_patterns_utterance = compute_pattern_features(
        syn_transcript,
        collapse_same_speaker=False
    )

    syn_patterns_turn = compute_pattern_features(
        syn_transcript,
        collapse_same_speaker=True
    )

    # Level explizit in die Metriknamen schreiben
    syn_patterns = {
        **{f"{k}_utterancelevel": v for k, v in syn_patterns_utterance.items()},
        **{f"{k}_turnlevel": v for k, v in syn_patterns_turn.items()},
    }

    # -----------------------------
    # Neighbor-Pattern berechnen
    # -----------------------------
    neighbor_pattern_rows = []

    for pid in neighbor_ids:
        real_transcript = transcript_df[transcript_df["Participant_ID"] == pid]

        if len(real_transcript) == 0:
            continue

        pattern_utterance = compute_pattern_features(
            real_transcript,
            collapse_same_speaker=False
        )

        pattern_turn = compute_pattern_features(
            real_transcript,
            collapse_same_speaker=True
        )

        pattern = {
            **{f"{k}_utterancelevel": v for k, v in pattern_utterance.items()},
            **{f"{k}_turnlevel": v for k, v in pattern_turn.items()},
        }

        pattern["Participant_ID"] = pid
        neighbor_pattern_rows.append(pattern)

    neighbor_patterns_df = pd.DataFrame(neighbor_pattern_rows)

    if neighbor_patterns_df.empty:
        print(f"Keine Neighbor-Transkripte gefunden für {synthetic_id}")
        continue

    # -----------------------------
    # Vergleich synthetisch vs. Neighbor-Mittelwert
    # -----------------------------
    for metric_name, synthetic_value in syn_patterns.items():
        if metric_name not in neighbor_patterns_df.columns:
            continue

        neighbor_values = neighbor_patterns_df[metric_name].dropna()

        if len(neighbor_values) == 0:
            continue

        neighbor_mean = neighbor_values.mean()
        neighbor_sd = neighbor_values.std(ddof=1)
        neighbor_min = neighbor_values.min()
        neighbor_max = neighbor_values.max()

        diff = synthetic_value - neighbor_mean
        abs_diff = abs(diff)
        z_diff = safe_z_score(synthetic_value, neighbor_mean, neighbor_sd)

        comparison_rows.append({
            "synthetic_id": synthetic_id,
            "metric": metric_name,

            "target_PHQ8_Score": syn_row["target_PHQ8_Score"],
            "target_PHQ8_binary": syn_row["target_PHQ8_binary"],
            "target_PHQ8_group": syn_row["target_PHQ8_group"],

            "synthetic_value": synthetic_value,
            "neighbor_mean": neighbor_mean,
            "neighbor_sd": neighbor_sd,
            "neighbor_min": neighbor_min,
            "neighbor_max": neighbor_max,

            "diff": diff,
            "abs_diff": abs_diff,
            "z_diff": z_diff,
            "abs_z_diff": abs(z_diff) if not pd.isna(z_diff) else np.nan,
            "pct_diff": safe_pct_diff(synthetic_value, neighbor_mean),

            "within_neighbor_range": (
                synthetic_value >= neighbor_min and synthetic_value <= neighbor_max
            ),

            "mean_neighbor_distance": mean_neighbor_distance,
            "min_neighbor_distance": min_neighbor_distance,
            "max_neighbor_distance": max_neighbor_distance,

            "mean_neighbor_PHQ8_Score": mean_neighbor_phq8_score,
            "n_neighbor_PHQ_ge_cutoff": n_neighbor_phq_ge_cutoff,
            "share_neighbor_PHQ_ge_cutoff": share_neighbor_phq_ge_cutoff,

            "n_neighbors_used": len(neighbor_values)
        })

comparison_df = pd.DataFrame(comparison_rows)
neighbor_ids_df = pd.DataFrame(neighbor_id_rows)

if comparison_df.empty:
    raise RuntimeError("comparison_df ist leer. Prüfe Synthetic-Datei, Target-Spalten und Neighbor-Suche.")


# ----------------------------------------------------------------------
# Summary pro Metrik
# ----------------------------------------------------------------------
summary_df = (
    comparison_df
    .groupby("metric")
    .agg(
        mean_synthetic_value=("synthetic_value", "mean"),
        mean_neighbor_value=("neighbor_mean", "mean"),
        mean_diff=("diff", "mean"),
        mean_abs_diff=("abs_diff", "mean"),
        median_abs_diff=("abs_diff", "median"),
        mean_abs_z_diff=("abs_z_diff", "mean"),
        median_abs_z_diff=("abs_z_diff", "median"),
        within_neighbor_range_rate=("within_neighbor_range", "mean"),
        mean_neighbor_distance=("mean_neighbor_distance", "mean"),
        n=("synthetic_id", "nunique")
    )
    .reset_index()
    .sort_values("mean_abs_z_diff", ascending=False)
)

summary_by_phq_group_df = (
    comparison_df
    .groupby(["target_PHQ8_group", "metric"])
    .agg(
        mean_synthetic_value=("synthetic_value", "mean"),
        mean_neighbor_value=("neighbor_mean", "mean"),
        mean_abs_diff=("abs_diff", "mean"),
        median_abs_diff=("abs_diff", "median"),
        mean_abs_z_diff=("abs_z_diff", "mean"),
        median_abs_z_diff=("abs_z_diff", "median"),
        within_neighbor_range_rate=("within_neighbor_range", "mean"),
        mean_neighbor_distance=("mean_neighbor_distance", "mean"),
        n=("synthetic_id", "nunique")
    )
    .reset_index()
    .sort_values(["metric", "target_PHQ8_group"])
)

synthetic_level_df = (
    comparison_df
    .groupby(["synthetic_id", "target_PHQ8_Score", "target_PHQ8_binary", "target_PHQ8_group"])
    .agg(
        mean_abs_z_diff=("abs_z_diff", "mean"),
        median_abs_z_diff=("abs_z_diff", "median"),
        mean_neighbor_distance=("mean_neighbor_distance", "mean"),
        within_neighbor_range_rate=("within_neighbor_range", "mean")
    )
    .reset_index()
)


plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_BAR = "#6C8EAD"          # muted steel blue
COLOR_BAR_2 = "#7A9E7E"       # muted sage
COLOR_LOW = "lightcoral"          # PHQ < {PHQ_CUTOFF}
COLOR_HIGH = "darkblue"         # PHQ >= {PHQ_CUTOFF}
COLOR_REF = "gray"
COLOR_GRID = "lightgray"


def prettify_ax(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, alpha=1, linewidth=0.8, color=COLOR_GRID)
    ax.set_axisbelow(True)
    ax.spines["left"].set_alpha(1)
    ax.spines["bottom"].set_alpha(1)
    ax.tick_params(axis="both", length=0)


def pretty_metric_name(metric):
    return (
        str(metric)
        .replace("_", " ")
        .replace("phq", "PHQ")
        .replace("ttr", "TTR")
        .title()
    )


def add_bar_labels(ax, bars, values, fmt="{:.2f}", offset_frac=0.015):
    max_val = np.nanmax(np.abs(values))
    if np.isnan(max_val) or max_val == 0:
        max_val = 1

    offset = max_val * offset_frac

    for bar, value in zip(bars, values):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        ax.text(
            x + offset,
            y,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=8.5,
            color="#333333"
        )


def set_scatter_limits(ax, x, y):
    min_val = min(np.nanmin(x), np.nanmin(y))
    max_val = max(np.nanmax(x), np.nanmax(y))

    value_range = max_val - min_val
    if value_range == 0:
        value_range = 1

    pad = value_range * 0.08

    ax.set_xlim(min_val - pad, max_val + pad)
    ax.set_ylim(min_val - pad, max_val + pad)


def scatter_by_phq_group(
    ax,
    data,
    x_col,
    y_col,
    xlabel,
    ylabel,
    title,
    diagonal=False,
    ylim=None
):
    low = data[data["target_PHQ8_binary"] == False]
    high = data[data["target_PHQ8_binary"] == True]

    ax.scatter(
        low[x_col],
        low[y_col],
        alpha=1.0,
        s=42,
        marker="o",
        color=COLOR_LOW,
        edgecolor="white",
        linewidth=0.6,
        label=f"Target PHQ < {PHQ_CUTOFF}"
    )

    ax.scatter(
        high[x_col],
        high[y_col],
        alpha=1.0,
        s=42,
        marker="o",
        color=COLOR_HIGH,
        edgecolor="white",
        linewidth=0.6,
        label=f"Target PHQ ≥ {PHQ_CUTOFF}"
    )

    if diagonal:
        min_val = min(data[x_col].min(), data[y_col].min())
        max_val = max(data[x_col].max(), data[y_col].max())

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            linewidth=1,
            color=COLOR_REF,
            alpha=1.0,
            label="Perfect agreement"
        )

        set_scatter_limits(ax, data[x_col], data[y_col])

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.axvline(
        PHQ_CUTOFF,
        linestyle="--",
        linewidth=1,
        color=COLOR_REF,
        alpha=1.0
    ) if x_col == "target_PHQ8_Score" else None

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=10)

    prettify_ax(ax, grid_axis="both")

    ax.legend(
        frameon=False,
        loc="best"
    )


# ============================================================
# Plot 1: Mean absolute z-difference pro Pattern
# ============================================================

plot_df = summary_df.dropna(subset=["mean_abs_z_diff"]).copy()
plot_df = plot_df.sort_values("mean_abs_z_diff", ascending=True)
plot_df["metric_label"] = plot_df["metric"].apply(pretty_metric_name)

fig, ax = plt.subplots(figsize=(8.5, 6.2))

bars = ax.barh(
    plot_df["metric_label"],
    plot_df["mean_abs_z_diff"],
    color=COLOR_BAR,
    edgecolor="white",
    linewidth=1.0,
    alpha=1
)

add_bar_labels(
    ax,
    bars,
    plot_df["mean_abs_z_diff"].values,
    fmt="{:.2f}"
)

ax.set_xlabel("Mean absolute z-difference to neighbor mean")
ax.set_ylabel("Pattern metric")
ax.set_title(
    "Synthetic Transcripts vs. Nearest-Neighbor Patterns",
    fontweight="bold",
    pad=12
)

xmax = max(plot_df["mean_abs_z_diff"].max() * 1.18, 1.15)
ax.set_xlim(0, xmax)

prettify_ax(ax, grid_axis="x")
ax.legend(frameon=False, loc="lower right")

plt.tight_layout()

plot1_file = OUTPUT_DIR / "mean_abs_z_diff_by_metric.png"
plt.savefig(plot1_file, dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# Plot 2: Range Coverage pro Pattern
# ============================================================

coverage_df = summary_df.dropna(subset=["within_neighbor_range_rate"]).copy()
coverage_df = coverage_df.sort_values("within_neighbor_range_rate", ascending=True)
coverage_df["metric_label"] = coverage_df["metric"].apply(pretty_metric_name)

fig, ax = plt.subplots(figsize=(8.5, 6.2))

bars = ax.barh(
    coverage_df["metric_label"],
    coverage_df["within_neighbor_range_rate"],
    color=COLOR_BAR_2,
    edgecolor="white",
    linewidth=1.0,
    alpha=1
)

add_bar_labels(
    ax,
    bars,
    coverage_df["within_neighbor_range_rate"].values,
    fmt="{:.2f}",
    offset_frac=0.012
)

ax.set_xlim(0, 1.08)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))

ax.set_xlabel("Share within neighbor min-max range")
ax.set_ylabel("Pattern metric")
ax.set_title(
    "Coverage of Neighbor Ranges by Synthetic Patterns",
    fontweight="bold",
    pad=12
)

prettify_ax(ax, grid_axis="x")

plt.tight_layout()

plot2_file = OUTPUT_DIR / "neighbor_range_coverage_by_metric.png"
plt.savefig(plot2_file, dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# Plot 3: Scatterplots pro Metrik mit PHQ-Markierung
# ============================================================

# selected_metrics = [
#     "n_turns_turnlevel",
#     "participant_words_turnlevel",
#     "median_participant_words_per_turn_turnlevel",
#     "short_participant_turn_ratio_turnlevel",
#     "ttr_participant_turnlevel",
# ]

selected_metrics = [
    "mean_participant_words_per_turn_utterancelevel",
    "mean_participant_words_per_turn_turnlevel",
    "median_participant_words_per_turn_utterancelevel",
    "median_participant_words_per_turn_turnlevel",
]

for metric in selected_metrics:
    tmp = comparison_df[comparison_df["metric"] == metric].dropna(
        subset=["synthetic_value", "neighbor_mean", "target_PHQ8_binary"]
    ).copy()

    if tmp.empty:
        continue

    fig, ax = plt.subplots(figsize=(5.6, 5.2))

    scatter_by_phq_group(
        ax=ax,
        data=tmp,
        x_col="neighbor_mean",
        y_col="synthetic_value",
        xlabel="Neighbor mean",
        ylabel="Synthetic value",
        title=f"Synthetic vs. Neighbor Mean\n{pretty_metric_name(metric)}",
        diagonal=True
    )

    plt.tight_layout()

    plot_file = OUTPUT_DIR / f"scatter_synthetic_vs_neighbor_{metric}_phq_group.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Plot 4: Sind PHQ >= {PHQ_CUTOFF} Targets weiter von ihren Nachbar:innen entfernt?
# ============================================================

distance_df = neighbor_ids_df.dropna(
    subset=["target_PHQ8_Score", "mean_neighbor_distance", "target_PHQ8_binary"]
).copy()

fig, ax = plt.subplots(figsize=(6.2, 5.2))

scatter_by_phq_group(
    ax=ax,
    data=distance_df,
    x_col="target_PHQ8_Score",
    y_col="mean_neighbor_distance",
    xlabel="Target PHQ-8 score",
    ylabel="Mean neighbor distance",
    title="Neighbor Distance by Target PHQ-8 Score",
    diagonal=False
)

plt.tight_layout()

plot4_file = OUTPUT_DIR / "neighbor_distance_by_target_phq_score.png"
plt.savefig(plot4_file, dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# Plot 5: Durchschnittliche Pattern-Abweichung pro synthetischem Transcript
# ============================================================

plot5_df = synthetic_level_df.dropna(
    subset=["target_PHQ8_Score", "mean_abs_z_diff", "target_PHQ8_binary"]
).copy()

fig, ax = plt.subplots(figsize=(6.2, 5.2))

scatter_by_phq_group(
    ax=ax,
    data=plot5_df,
    x_col="target_PHQ8_Score",
    y_col="mean_abs_z_diff",
    xlabel="Target PHQ-8 score",
    ylabel="Mean absolute z-difference",
    title="Overall Pattern Deviation by Target PHQ-8 Score",
    diagonal=False
)

plt.tight_layout()

plot5_file = OUTPUT_DIR / "overall_pattern_deviation_by_target_phq_score.png"
plt.savefig(plot5_file, dpi=300, bbox_inches="tight")
plt.show()