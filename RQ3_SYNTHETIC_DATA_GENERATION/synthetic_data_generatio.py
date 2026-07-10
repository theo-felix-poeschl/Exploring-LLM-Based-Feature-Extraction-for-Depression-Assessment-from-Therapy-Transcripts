import sys
sys.path.append("/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/RQ3_SYNTHETIC_DATA_GENERATION/neighboring")

import os
import time
import json
import random
import re
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from neighboring import find_neighbors
from create_personas import generate_random_persona, GENDERS

from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Literal


# ----------------------------------------------------------------------
# Pfade & Konstanten
# ----------------------------------------------------------------------
FEATURES_FILE = Path("/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/INPUT/DAIC_META.csv")
TRANSCRIPT_FILE = Path("/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/INPUT/DAIC_FULL_NO_TAGS.csv")
NEIGHBORING_FILE = Path("/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/INPUT/DAIC_META.csv")

OUTPUT_FILE = Path("/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/OUTPUT/10_synthetic_transcripts_long.csv")
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

N_SYNTHETIC = 189
N_FEW_SHOT   = 5
MAX_EXAMPLE_TURNS = None  # None → komplette Beispiel-Transkripte

random.seed(42)
np.random.seed(42)


# ----------------------------------------------------------------------
# Pydantic-Schema für LLM-Ausgabe
# ----------------------------------------------------------------------
class TranscriptTurn(BaseModel):
    speaker: Literal["Therapist", "Participant"]
    value: str = Field(min_length=1)

    @field_validator("value")
    @classmethod
    def clean_value(cls, v):
        v = str(v).strip()
        if not v:
            raise ValueError("value darf nicht leer sein")
        return v


class SyntheticTranscript(BaseModel):
    turns: list[TranscriptTurn] = Field(min_length=1)


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
    "PHQ8_Concentrating": np.random.randint(0, 4, size=N_SYNTHETIC),
    "PHQ8_Appetite":      np.random.randint(0, 4, size=N_SYNTHETIC),
    "PHQ8_Depressed":     np.random.randint(0, 4, size=N_SYNTHETIC),
    "PHQ8_Tired":         np.random.randint(0, 4, size=N_SYNTHETIC),
    "PHQ8_NoInterest":    np.random.randint(0, 4, size=N_SYNTHETIC),
    "PHQ8_Failure":       np.random.randint(0, 4, size=N_SYNTHETIC),
    "PHQ8_Moving":        np.random.randint(0, 4, size=N_SYNTHETIC),
    "PHQ8_Sleep":         np.random.randint(0, 4, size=N_SYNTHETIC),
})

target_features["synthetic_id"] = [f"syn_{i:04d}" for i in range(len(target_features))]


# ----------------------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------------------
def normalize_speaker(speaker):
    s = str(speaker).strip().lower()

    if s in ["participant", "patient", "client"]:
        return "Participant"

    if s in ["ellie", "therapist", "interviewer", "clinician"]:
        return "Therapist"

    return None

def make_transcript(group):
    """Erzeugt einen Text-Block aus den Zeilen einer Person."""
    group = group.sort_values("start_time").reset_index(drop=True)

    if MAX_EXAMPLE_TURNS is not None and len(group) > MAX_EXAMPLE_TURNS:
        start_idx = random.randint(0, len(group) - MAX_EXAMPLE_TURNS)
        group = group.iloc[start_idx:start_idx + MAX_EXAMPLE_TURNS]

    lines = []

    for _, row in group.iterrows():
        speaker = normalize_speaker(row["speaker"])
        value = str(row["value"]).strip()

        if speaker is None or not value:
            continue

        lines.append(f"{speaker} | {value}")

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
        k=N_FEW_SHOT
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

        transcript = make_transcript(grp)

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

            # Anzahl der Turns im Neighbor-Transcript
            "n_turns": len(grp),

            # Das eigentliche Beispiel-Transkript
            "transcript": transcript
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


def build_messages(
    target_feats: dict,
    persona: dict,
    examples: list,
    neighbor_mean_n_turns: int
) -> list:
    system_prompt = f"""
        You generate synthetic therapy dialog transcripts for research.

        You will be given the features and transcript of the {N_FEW_SHOT}-nearest neighbors as examples.
        These examples serve as a template for content, conversation structure, response style, and length.

        PHQ8 Features (higher values => higher degree of symptoms):
        - PHQ8_NoInterest:   Little interest or pleasure in doing things.
        - PHQ8_Depressed:    Feeling down, depressed, or hopeless.
        - PHQ8_Sleep:        Trouble falling/staying asleep, or sleeping too much.
        - PHQ8_Tired:        Feeling tired or having little energy.
        - PHQ8_Appetite:     Poor appetite or overeating.
        - PHQ8_Failure:      Feeling bad about yourself, or that you are a failure.
        - PHQ8_Concentrating: Trouble concentrating on things.
        - PHQ8_Moving:       Moving/speaking slowly, or being fidgety/restless.

        Rules:
        - Output only valid JSON.
        - Do not use markdown.
        - Do not explain anything.
        - Do not copy the examples.
        - Use proper grammar and punctuation even if the given examples might not do that.
        - The speaker value can only be "Therapist" or "Participant".
        - The Participant should give answers of varying lengths, just like in real interviews.
        - The generated transcript should contain approximately {neighbor_mean_n_turns} turns.
        - The JSON must have exactly this structure:

        {{
          "turns": [
            {{
              "speaker": "Therapist",
              "value": "utterance text"
            }},
            {{
              "speaker": "Participant",
              "value": "utterance text"
            }}
          ]
        }}
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

        # Few-shot assistant message jetzt ebenfalls im gewünschten JSON-Format.
        example_turns = []
        for line in ex["transcript"].splitlines():
            if "|" not in line:
                continue

            speaker, value = line.split("|", 1)
            speaker = speaker.strip()
            value = value.strip()

            if speaker not in ["Therapist", "Participant"]:
                continue

            if value:
                example_turns.append({
                    "speaker": speaker,
                    "value": value
                })

        assistant_msg = json.dumps(
            {"turns": example_turns},
            ensure_ascii=False
        )

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

        Length requirement:
                The generated transcript should contain approximately {neighbor_mean_n_turns} turns.

        Return only valid JSON with this exact structure:

        {{
          "turns": [
            {{
              "speaker": "Therapist",
              "value": "utterance text"
            }},
            {{
              "speaker": "Participant",
              "value": "utterance text"
            }}
          ]
        }}
        """.strip()

    messages.append({"role": "user", "content": final_user_msg})
    return messages


def extract_json_object(text: str) -> str:
    """
    Extrahiert das erste JSON-Objekt aus einer Modellantwort.
    Hilft, falls das Modell doch ```json oder Text drumherum produziert.
    """
    text = text.strip()

    # Markdown-Fences entfernen, falls vorhanden
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Direkt gültiges JSON
    if text.startswith("{") and text.endswith("}"):
        return text

    # Erstes JSON-Objekt herausziehen
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError("Kein JSON-Objekt in der LLM-Ausgabe gefunden.")


def parse_pipe_fallback(raw_text: str) -> SyntheticTranscript:
    """
    Notfall-Fallback, falls das Modell doch kein JSON liefert.
    Akzeptiert Zeilen wie:
    Therapist | hello
    Participant | hi
    """
    turns = []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue

        speaker = parts[0]
        value = parts[1]

        if speaker in ["Therapist", "Participant"] and value:
            turns.append({
                "speaker": speaker,
                "value": value
            })

    if not turns:
        raise ValueError("Auch Pipe-Fallback konnte keine Turns parsen.")

    return SyntheticTranscript(turns=turns)


def parse_llm_output(raw_text: str) -> SyntheticTranscript:
    """
    Validiert die LLM-Ausgabe mit Pydantic.
    Erst JSON, dann robuster Pipe-Fallback.
    """
    try:
        json_text = extract_json_object(raw_text)
        data = json.loads(json_text)
        return SyntheticTranscript.model_validate(data)

    except Exception:
        return parse_pipe_fallback(raw_text)


def call_llm(messages: list) -> SyntheticTranscript:
    """
    Wrapper-Funktion für den Chat-Aufruf mit Retry.
    Die API wird nicht mit response_format/schema gezwungen, weil das bei
    manchen OpenAI-kompatiblen Endpoints ständig Fehler verursacht.
    Stattdessen wird lokal mit Pydantic validiert.
    """
    last_error = None
    last_raw_text = None

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.8
            )

            raw_text = response.choices[0].message.content.strip()
            last_raw_text = raw_text

            parsed = parse_llm_output(raw_text)
            return parsed

        except Exception as e:
            last_error = e
            print(f"Fehler beim LLM-Call oder Parsing, Versuch {attempt + 1}/3:", e)

            # Bei Parsing-Fehlern dem Modell beim nächsten Versuch klarer sagen,
            # was schiefgelaufen ist.
            messages = messages + [
                {
                    "role": "assistant",
                    "content": last_raw_text if last_raw_text else ""
                },
                {
                    "role": "user",
                    "content": """
                    Your previous answer could not be parsed.
                    Return only valid JSON.
                    Do not use markdown.
                    Use exactly this structure:

                    {
                      "turns": [
                        {
                          "speaker": "Therapist",
                          "value": "utterance text"
                        },
                        {
                          "speaker": "Participant",
                          "value": "utterance text"
                        }
                      ]
                    }
                    """.strip()
                }
            ]

            time.sleep(2 + attempt * 2)

    raise RuntimeError(f"LLM-Call nach 3 Versuchen fehlgeschlagen: {repr(last_error)}")

def transcript_to_long_rows(
    synthetic_id: str,
    parsed_transcript: SyntheticTranscript,
    target_dict: dict
) -> list[dict]:
    """
    Wandelt das validierte Pydantic-Objekt in Long Format um.

    Ausgabe:
    synthetic_id,
    target_PHQ8_...,
    target_PHQ8_Score,
    speaker,
    value
    """
    rows = []

    target_cols = {
        f"target_{key}": int(value)
        for key, value in target_dict.items()
    }

    target_score = int(sum(target_dict.values()))

    for turn in parsed_transcript.turns:
        rows.append({
            "synthetic_id": synthetic_id,
            **target_cols,
            "target_PHQ8_Score": target_score,
            "speaker": turn.speaker,
            "value": turn.value
        })

    return rows


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
    query_point = target_dict

    try:
        persona = generate_random_persona(gender=random.choice(GENDERS), age_range=(18, 80))
        examples = get_few_shot_examples(query_point)
        for ex in examples[:1]:
            print(ex["transcript"].splitlines()[:20])

        neighbor_mean_n_turns = int(round(np.mean([ex["n_turns"] for ex in examples])))

        messages = build_messages(
            target_feats=target_dict,
            persona=persona,
            examples=examples,
            neighbor_mean_n_turns=neighbor_mean_n_turns
        )

        transcript = call_llm(messages)

        long_rows = transcript_to_long_rows(
            synthetic_id=synthetic_id,
            parsed_transcript=transcript,
            target_dict=target_dict
        )

        results.extend(long_rows)

        # Zwischenspeichern, damit kein Datenverlust bei einem Crash entsteht
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

        time.sleep(0.5)

    except Exception as e:
        error = {
            "synthetic_id": synthetic_id,
            "error": repr(e)
        }
        errors.append(error)
        pd.DataFrame(errors).to_csv(ERROR_FILE, index=False)
        print("Fehler bei", synthetic_id, e)


# ----------------------------------------------------------------------
# Ergebnis-Export
# ----------------------------------------------------------------------
target_output_columns = [f"target_{col}" for col in FEATURE_COLUMNS]

results_df = pd.DataFrame(
    results,
    columns=[
        "synthetic_id",
        *target_output_columns,
        "target_PHQ8_Score",
        "speaker",
        "value"
    ]
)
errors_df  = pd.DataFrame(errors)

results_df.to_csv(OUTPUT_FILE, index=False)
errors_df.to_csv(ERROR_FILE, index=False)

print("Fertig.")
print("Gespeichert unter:", OUTPUT_FILE)
print("Fehler gespeichert unter:", ERROR_FILE)