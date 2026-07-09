import sys
sys.path.append("/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/RQ3_SYNTHETIC_DATA_GENERATION/neighboring")

import os
import time
import json
import random
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from neighboring import find_neighbors
from create_personas import generate_random_persona, GENDERS

# ----------------------------------------------------------------------
# Pfade & Konstanten
# ----------------------------------------------------------------------
FEATURES_FILE = Path("/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/INPUT/DAIC_META.csv")
TRANSCRIPT_FILE = Path("/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/INPUT/DAIC_FULL_NO_TAGS.csv")
NEIGHBORING_FILE = Path("/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/INPUT/DAIC_META.csv")

OUTPUT_FILE = Path("/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/OUTPUT/10_synthetic_transcripts.csv")
ERROR_FILE = Path("/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/OUTPUT/synthetic_transcripts_errors.csv")

MODEL_NAME = "openai/gpt-oss-120b"
BASE_URL = "https://llmchat.idm.uk-augsburg.science/api"

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

N_SYNTHETIC = 10
N_FEW_SHOT   = 5
MAX_EXAMPLE_TURNS = None  # None → komplette Beispiel-Transkripte

random.seed(42)
np.random.seed(42)

# ----------------------------------------------------------------------
# OpenAI-Client
# ----------------------------------------------------------------------
client = OpenAI(
    api_key="sk-ab5121e5ffda4b76bdf1240d07e552f9",
    base_url=BASE_URL,
)

# ----------------------------------------------------------------------
# Daten einlesen
# ----------------------------------------------------------------------
features_df = pd.read_csv(FEATURES_FILE)
transcript_df = pd.read_csv(TRANSCRIPT_FILE)

features_df["Participant_ID"] = pd.to_numeric(features_df["Participant_ID"])
transcript_df["Participant_ID"] = pd.to_numeric(transcript_df["Participant_ID"])

transcript_df = transcript_df.dropna(subset=["Participant_ID", "value"])
transcript_df["value"]   = transcript_df["value"].astype(str).str.strip()
transcript_df["speaker"] = transcript_df["speaker"].astype(str).str.strip()

transcript_df = transcript_df.sort_values(["Participant_ID", "start_time"]).reset_index(drop=True)

print("Features:", features_df.shape)
print("Transcripts:", transcript_df.shape)
print("Participants mit Transcript:", transcript_df["Participant_ID"].nunique())

# ----------------------------------------------------------------------
# Ziel-Feature-Kombinationen erzeugen
# ----------------------------------------------------------------------
target_features = pd.DataFrame({
    "PHQ8_Concentrating": np.random.randint(0, 3, size=N_SYNTHETIC),
    "PHQ8_Appetite":      np.random.randint(0, 3, size=N_SYNTHETIC),
    "PHQ8_Depressed":     np.random.randint(0, 3, size=N_SYNTHETIC),
    "PHQ8_Tired":         np.random.randint(0, 3, size=N_SYNTHETIC),
    "PHQ8_NoInterest":    np.random.randint(0, 3, size=N_SYNTHETIC),
    "PHQ8_Failure":       np.random.randint(0, 3, size=N_SYNTHETIC),
    "PHQ8_Moving":        np.random.randint(0, 3, size=N_SYNTHETIC),
    "PHQ8_Sleep":         np.random.randint(0, 3, size=N_SYNTHETIC),
})

target_features["synthetic_id"] = [f"syn_{i:04d}" for i in range(len(target_features))]

# ----------------------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------------------
def make_transcript(group):
    """Erzeugt einen Text-Block aus den Zeilen einer Person."""
    group = group.sort_values("start_time").reset_index(drop=True)

    if MAX_EXAMPLE_TURNS is not None and len(group) > MAX_EXAMPLE_TURNS:
        start_idx = random.randint(0, len(group) - MAX_EXAMPLE_TURNS)
        group = group.iloc[start_idx:start_idx + MAX_EXAMPLE_TURNS]

    lines = [f"{row['speaker']} | {row['value']}" for _, row in group.iterrows()]
    return "\n".join(lines)


def get_few_shot_examples(query_point: dict) -> list:
    """
    Sucht die N_FEW_SHOT nächsten Nachbarn im Neighbor-File und
    gibt für jede gefundene Person ein Dictionary mit den 8 PHQ-8-Werten
    sowie dem zugehörigen Transkript-Excerpt zurück.
    """
    neighbor_df = find_neighbors(
        csv_path=NEIGHBORING_FILE,
        query=query_point,
        feature_cols=list(query_point.keys()),
        k = N_FEW_SHOT
    )

    neighbor_df = pd.merge(
        neighbor_df,
        transcript_df[["Participant_ID", "start_time", "speaker", "value"]],
        on="Participant_ID",
        how="inner"
    )

    examples = []
    participant_ids = neighbor_df["Participant_ID"].drop_duplicates().head(N_FEW_SHOT)

    for pid in participant_ids:
        grp = neighbor_df[neighbor_df["Participant_ID"] == pid].copy()

        example = {
            "Participant_ID": pid,
            "PHQ8_Concentrating": grp["PHQ8_Concentrating"].iloc[0],
            "PHQ8_Appetite":      grp["PHQ8_Appetite"].iloc[0],
            "PHQ8_Depressed":     grp["PHQ8_Depressed"].iloc[0],
            "PHQ8_Tired":         grp["PHQ8_Tired"].iloc[0],
            "PHQ8_NoInterest":    grp["PHQ8_NoInterest"].iloc[0],
            "PHQ8_Failure":       grp["PHQ8_Failure"].iloc[0],
            "PHQ8_Moving":        grp["PHQ8_Moving"].iloc[0],
            "PHQ8_Sleep":         grp["PHQ8_Sleep"].iloc[0],
            # Das eigentliche Beispiel-Transkript:
            "transcript": make_transcript(grp)
        }
        examples.append(example)

    return examples


def make_persona_text(persona: dict) -> str:
    """Formatiert die Persona-Informationen für das Prompt."""
    return f"""
    Name: {persona["first_name"]} {persona["last_name"]}
    Gender: {persona["gender"]}
    Age: {persona["age"]}
    Occupation: {persona["occupation"]}""".strip()


def build_messages(target_feats: dict, persona: dict, examples: list) -> list:
    system_prompt = f"""
        You generate synthetic therapy dialog transcripts for research.

        You will be given the features and transcript of the {N_FEW_SHOT}-nearest neighbors as examples.
        These examples serve as a template for content, conversation structure, response style, and length.

        PHQ8 Features (higher values → higher degree of symptoms):
        - PHQ8_NoInterest:   Little interest or pleasure in doing things.
        - PHQ8_Depressed:    Feeling down, depressed, or hopeless.
        - PHQ8_Sleep:        Trouble falling/staying asleep, or sleeping too much.
        - PHQ8_Tired:        Feeling tired or having little energy.
        - PHQ8_Appetite:     Poor appetite or overeating.
        - PHQ8_Failure:      Feeling bad about yourself, or that you are a failure.
        - PHQ8_Concentrating: Trouble concentrating on things.
        - PHQ8_Moving:       Moving/speaking slowly, or being fidgety/restless.

        Rules:
        - Output only the transcript.
        - Use this format: speaker | value | target_PHQ8_Concentrating | target_PHQ8_Appetite | target_PHQ8_Depressed | target_PHQ8_Tired | target_PHQ8_NoInterest | target_PHQ8_Failure | target_PHQ8_Moving | target_PHQ8_Sleep
        - Use proper grammar and punctuation even if the given examples might not do that.
        - In the speaker value there can only be the values "Therapist" or "Participant"
        - The Participant should give answers of varying lengths, just like in real interviews.
        - Do not copy the examples.
        - Do not explain anything.
        """.strip()

    messages = [{"role": "system", "content": system_prompt}]

            # ---- Few-Shot-Beispiele -------------------------------------------------
    for ex in examples:
        user_msg = f"""
        Generate a transcript with these features:

        PHQ8_Concentrating: {ex["PHQ8_Concentrating"]}
        PHQ8_Appetite:      {ex["PHQ8_Appetite"]}
        PHQ8_Depressed:     {ex["PHQ8_Depressed"]}
        PHQ8_Tired:         {ex["PHQ8_Tired"]}
        PHQ8_NoInterest:    {ex["PHQ8_NoInterest"]}
        PHQ8_Failure:       {ex["PHQ8_Failure"]}
        PHQ8_Moving:        {ex["PHQ8_Moving"]}
        PHQ8_Sleep:         {ex["PHQ8_Sleep"]}
        """.strip()

        assistant_msg = ex["transcript"]

        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    # ---- eigentliche Anfrage ------------------------------------------------
    final_user_msg = f"""
        Generate a new synthetic transcript.

        Patient persona:
        {make_persona_text(persona)}

        Target features:
        PHQ8_Concentrating: {target_feats["PHQ8_Concentrating"]}
        PHQ8_Appetite:      {target_feats["PHQ8_Appetite"]}
        PHQ8_Depressed:     {target_feats["PHQ8_Depressed"]}
        PHQ8_Tired:         {target_feats["PHQ8_Tired"]}
        PHQ8_NoInterest:    {target_feats["PHQ8_NoInterest"]}
        PHQ8_Failure:       {target_feats["PHQ8_Failure"]}
        PHQ8_Moving:        {target_feats["PHQ8_Moving"]}
        PHQ8_Sleep:         {target_feats["PHQ8_Sleep"]}

        Use this format:

        speaker | value | target_PHQ8_Concentrating | target_PHQ8_Appetite | target_PHQ8_Depressed | target_PHQ8_Tired | target_PHQ8_NoInterest | target_PHQ8_Failure | target_PHQ8_Moving | target_PHQ8_Sleep
        """.strip()

    messages.append({"role": "user", "content": final_user_msg})
    return messages


def call_llm(messages: list) -> str:
    """Wrapper-Funktion für den Chat-Aufruf mit einfachem Retry-Mechanismus."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Fehler beim LLM-Call:", e)
            time.sleep(2 + attempt * 2)

    raise RuntimeError("LLM-Call nach 3 Versuchen fehlgeschlagen")


# ----------------------------------------------------------------------
# Haupt-Loop
# ----------------------------------------------------------------------
results = []
errors  = []

for _, row in tqdm(target_features.iterrows(), total=len(target_features)):
    synthetic_id = row["synthetic_id"]

    # ---- Ziel-Feature-Dictionary für diesen Durchlauf --------------------
    target_dict = {
        "PHQ8_Concentrating": row["PHQ8_Concentrating"],
        "PHQ8_Appetite":      row["PHQ8_Appetite"],
        "PHQ8_Depressed":     row["PHQ8_Depressed"],
        "PHQ8_Tired":         row["PHQ8_Tired"],
        "PHQ8_NoInterest":    row["PHQ8_NoInterest"],
        "PHQ8_Failure":       row["PHQ8_Failure"],
        "PHQ8_Moving":        row["PHQ8_Moving"],
        "PHQ8_Sleep":         row["PHQ8_Sleep"],
    }

    # ---- Nachbarschaftssuche (Few-Shot) ---------------------------------
    query_point = target_dict  # wir verwenden exakt die 8 Features

    try:
        persona   = generate_random_persona(gender=random.choice(GENDERS), age_range=(18, 80))
        examples  = get_few_shot_examples(query_point)

        messages  = build_messages(target_feats=target_dict, persona=persona, examples=examples)
        transcript = call_llm(messages)

        result = {
            "synthetic_id": synthetic_id,
            **target_dict,
            "persona": json.dumps(persona, ensure_ascii=False),
            "few_shot_participants": json.dumps([ex["Participant_ID"] for ex in examples], ensure_ascii=False),
            "synthetic_transcript": transcript,
            "messages": json.dumps(messages, ensure_ascii=False),
        }

        results.append(result)

        # Zwischenspeichern, damit kein Datenverlust bei einem Crash entsteht
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

        time.sleep(0.5)

    except Exception as e:
        error = {"synthetic_id": synthetic_id, "error": repr(e)}
        errors.append(error)
        pd.DataFrame(errors).to_csv(ERROR_FILE, index=False)
        print("Fehler bei", synthetic_id, e)

# ----------------------------------------------------------------------
# Ergebnis-Export
# ----------------------------------------------------------------------
results_df = pd.DataFrame(results)
errors_df  = pd.DataFrame(errors)

results_df.to_csv(OUTPUT_FILE, index=False)
errors_df.to_csv(ERROR_FILE, index=False)

print("Fertig.")
print("Gespeichert unter:", OUTPUT_FILE)
print("Fehler gespeichert unter:", ERROR_FILE)