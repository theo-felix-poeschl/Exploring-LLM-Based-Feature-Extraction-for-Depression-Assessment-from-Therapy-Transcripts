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


# ----------------------------------------------------------------------
# Daten laden
# ----------------------------------------------------------------------
synthetic_df = pd.read_csv(SYNTHETIC_FILE)
features_df = pd.read_csv(FEATURES_FILE)
transcript_df = pd.read_csv(TRANSCRIPT_FILE)

features_df["Participant_ID"] = pd.to_numeric(features_df["Participant_ID"], errors="coerce")
transcript_df["Participant_ID"] = pd.to_numeric(transcript_df["Participant_ID"], errors="coerce")

transcript_df = transcript_df.dropna(subset=["Participant_ID", "speaker", "value"]).copy()
transcript_df["Participant_ID"] = transcript_df["Participant_ID"].astype(int)
transcript_df["speaker"] = transcript_df["speaker"].astype(str).str.strip()
transcript_df["value"] = transcript_df["value"].astype(str).str.strip()

if "start_time" in transcript_df.columns:
    transcript_df = transcript_df.sort_values(["Participant_ID", "start_time"])
else:
    transcript_df = transcript_df.sort_values(["Participant_ID"]).reset_index(drop=True)

synthetic_df["synthetic_id"] = synthetic_df["synthetic_id"].astype(str)
synthetic_df["speaker"] = synthetic_df["speaker"].astype(str).str.strip()
synthetic_df["value"] = synthetic_df["value"].astype(str).str.strip()

for col in TARGET_COLUMNS:
    synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors="coerce")


print("Synthetic rows:", synthetic_df.shape)
print("Synthetic transcripts:", synthetic_df["synthetic_id"].nunique())
print("Real transcript rows:", transcript_df.shape)
print("Real participants:", transcript_df["Participant_ID"].nunique())


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
    Für deine Zwecke ausreichend, weil es um relative Basic Patterns geht.
    """
    return re.findall(r"[A-Za-z']+", str(text).lower())


def compute_pattern_features(df):
    """
    Berechnet einfache strukturelle und lexikalische Pattern-Metriken
    für ein Transcript im Long Format.
    Erwartete Spalten: speaker, value
    """
    tmp = df.copy()
    tmp["speaker_norm"] = tmp["speaker"].apply(normalize_speaker)
    tmp["value"] = tmp["value"].fillna("").astype(str)
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

        "mean_participant_words": np.mean(participant_word_counts) if len(participant_word_counts) > 0 else np.nan,
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


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
comparison_rows = []
neighbor_id_rows = []

synthetic_targets = (
    synthetic_df[["synthetic_id", *TARGET_COLUMNS]]
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

    neighbor_ids = (
        neighbor_df["Participant_ID"]
        .drop_duplicates()
        .astype(int)
        .tolist()
    )

    neighbor_id_rows.append({
        "synthetic_id": synthetic_id,
        "neighbor_ids": json.dumps(neighbor_ids)
    })

    # -----------------------------
    # Synthetische Pattern berechnen
    # -----------------------------
    syn_transcript = synthetic_df[synthetic_df["synthetic_id"] == synthetic_id]
    syn_patterns = compute_pattern_features(syn_transcript)

    # -----------------------------
    # Neighbor-Pattern berechnen
    # -----------------------------
    neighbor_pattern_rows = []

    for pid in neighbor_ids:
        real_transcript = transcript_df[transcript_df["Participant_ID"] == pid]

        if len(real_transcript) == 0:
            continue

        pattern = compute_pattern_features(real_transcript)
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
        neighbor_values = neighbor_patterns_df[metric_name].dropna()

        if len(neighbor_values) == 0:
            continue

        neighbor_mean = neighbor_values.mean()
        neighbor_sd = neighbor_values.std(ddof=1)
        neighbor_min = neighbor_values.min()
        neighbor_max = neighbor_values.max()

        diff = synthetic_value - neighbor_mean
        abs_diff = abs(diff)

        comparison_rows.append({
            "synthetic_id": synthetic_id,
            "metric": metric_name,

            "synthetic_value": synthetic_value,
            "neighbor_mean": neighbor_mean,
            "neighbor_sd": neighbor_sd,
            "neighbor_min": neighbor_min,
            "neighbor_max": neighbor_max,

            "diff": diff,
            "abs_diff": abs_diff,
            "z_diff": safe_z_score(synthetic_value, neighbor_mean, neighbor_sd),
            "abs_z_diff": abs(safe_z_score(synthetic_value, neighbor_mean, neighbor_sd)),
            "pct_diff": safe_pct_diff(synthetic_value, neighbor_mean),

            "within_neighbor_range": (
                synthetic_value >= neighbor_min and synthetic_value <= neighbor_max
            ),

            "n_neighbors_used": len(neighbor_values)
        })


comparison_df = pd.DataFrame(comparison_rows)
neighbor_ids_df = pd.DataFrame(neighbor_id_rows)


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
        n=("synthetic_id", "nunique")
    )
    .reset_index()
    .sort_values("mean_abs_z_diff", ascending=False)
)


# ----------------------------------------------------------------------
# Speichern
# ----------------------------------------------------------------------
comparison_file = OUTPUT_DIR / "neighbor_pattern_comparison_long.csv"
summary_file = OUTPUT_DIR / "neighbor_pattern_summary.csv"
neighbor_ids_file = OUTPUT_DIR / "synthetic_neighbor_ids.csv"

comparison_df.to_csv(comparison_file, index=False)
summary_df.to_csv(summary_file, index=False)
neighbor_ids_df.to_csv(neighbor_ids_file, index=False)

print("Gespeichert:")
print(comparison_file)
print(summary_file)
print(neighbor_ids_file)


# ----------------------------------------------------------------------
# Plot 1: Mean absolute z-difference pro Pattern
# ----------------------------------------------------------------------
plot_df = summary_df.dropna(subset=["mean_abs_z_diff"]).copy()
plot_df = plot_df.sort_values("mean_abs_z_diff", ascending=True)

plt.figure(figsize=(8, 6))
plt.barh(plot_df["metric"], plot_df["mean_abs_z_diff"])
plt.axvline(1.0, linestyle="--", linewidth=1)
plt.xlabel("Mean absolute z-difference to neighbor mean")
plt.ylabel("Pattern metric")
plt.title("Synthetic transcripts vs. nearest-neighbor patterns")
plt.tight_layout()

plot1_file = OUTPUT_DIR / "mean_abs_z_diff_by_metric.png"
plt.savefig(plot1_file, dpi=300)
plt.show()


# ----------------------------------------------------------------------
# Plot 2: Range Coverage pro Pattern
# ----------------------------------------------------------------------
coverage_df = summary_df.sort_values("within_neighbor_range_rate", ascending=True)

plt.figure(figsize=(8, 6))
plt.barh(coverage_df["metric"], coverage_df["within_neighbor_range_rate"])
plt.xlim(0, 1)
plt.xlabel("Share within neighbor min-max range")
plt.ylabel("Pattern metric")
plt.title("How often synthetic patterns fall within neighbor range")
plt.tight_layout()

plot2_file = OUTPUT_DIR / "neighbor_range_coverage_by_metric.png"
plt.savefig(plot2_file, dpi=300)
plt.show()


# ----------------------------------------------------------------------
# Optional: Scatterplots für wenige zentrale Metriken
# ----------------------------------------------------------------------
selected_metrics = [
    "n_turns",
    "participant_words",
    "mean_participant_words",
    "short_participant_turn_ratio",
    "ttr_participant",
]

for metric in selected_metrics:
    tmp = comparison_df[comparison_df["metric"] == metric].dropna(
        subset=["synthetic_value", "neighbor_mean"]
    )

    if tmp.empty:
        continue

    plt.figure(figsize=(5, 5))
    plt.scatter(tmp["neighbor_mean"], tmp["synthetic_value"], alpha=0.8)

    min_val = min(tmp["neighbor_mean"].min(), tmp["synthetic_value"].min())
    max_val = max(tmp["neighbor_mean"].max(), tmp["synthetic_value"].max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1)

    plt.xlabel("Neighbor mean")
    plt.ylabel("Synthetic value")
    plt.title(f"Synthetic vs. neighbor mean: {metric}")
    plt.tight_layout()

    plot_file = OUTPUT_DIR / f"scatter_synthetic_vs_neighbor_{metric}.png"
    plt.savefig(plot_file, dpi=300)
    plt.show()