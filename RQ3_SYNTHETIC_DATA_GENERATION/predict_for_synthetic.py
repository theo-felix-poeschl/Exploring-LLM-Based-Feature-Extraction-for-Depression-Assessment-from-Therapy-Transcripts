import os
import pandas as pd
import json
import re
from pydantic import BaseModel, Field
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

N_RUNS = 3
MAX_WORKERS = 25
MAX_RETRIES = 3


class IndirectPHQ8Items(BaseModel):
    phq8_nointerest: int = Field(..., ge=0, le=3, description="Little interest or pleasure in doing things")
    phq8_depressed: int = Field(..., ge=0, le=3, description="Feeling down, depressed, or hopeless")
    phq8_sleep: int = Field(..., ge=0, le=3, description="Trouble falling or staying asleep, or sleeping too much")
    phq8_tired: int = Field(..., ge=0, le=3, description="Feeling tired or having little energy")
    phq8_appetite: int = Field(..., ge=0, le=3, description="Poor appetite or overeating")
    phq8_failure: int = Field(..., ge=0, le=3, description="Feeling bad about yourself or like a failure")
    phq8_concentrating: int = Field(..., ge=0, le=3, description="Trouble concentrating")
    phq8_moving: int = Field(..., ge=0, le=3, description="Moving/speaking slowly or being fidgety/restless")


OUTPUT_SCHEMA = IndirectPHQ8Items.model_json_schema()

PHQ8_ITEM_COLS = [
    "phq8_nointerest",
    "phq8_depressed",
    "phq8_sleep",
    "phq8_tired",
    "phq8_appetite",
    "phq8_failure",
    "phq8_concentrating",
    "phq8_moving",
]

TARGET_ITEM_COLS = [
    "target_PHQ8_NoInterest",
    "target_PHQ8_Depressed",
    "target_PHQ8_Sleep",
    "target_PHQ8_Tired",
    "target_PHQ8_Appetite",
    "target_PHQ8_Failure",
    "target_PHQ8_Concentrating",
    "target_PHQ8_Moving",
]

INPUT_PATH = "/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/OUTPUT/10_synthetic_transcripts_long.csv"
OUTPUT_PATH = "/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/OUTPUT/synthetic_indirect_no_reasoning.csv"
LONG_OUTPUT_PATH = "/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/OUTPUT/synthetic_indirect_no_reasoning_long.csv"

client = OpenAI(
    base_url="https://llmchat.idm.uk-augsburg.science/api",
    api_key="sk-ab5121e5ffda4b76bdf1240d07e552f9",
)

MODEL_NAME = "openai/gpt-oss-120b"

CONVERSATION_ID_COL = "synthetic_id"
TEXT_COL = "value"
SPEAKER_COL = "speaker"


SYSTEM_PROMPT = """
Task: You will be given part of a therapy-session transcript. The patient's utterances are labeled with "Patient:", and the therapist's utterances are labeled with "Therapist:".

Your task is NOT to predict the total PHQ score directly.

Instead, estimate the individual PHQ-8 item ratings for the patient based only on the provided transcript context.

Each PHQ-8 item must be rated from 0 to 3:

0 = Not at all
1 = Several days
2 = More than half the days
3 = Nearly every day

PHQ-8 items:
1. Little interest or pleasure in doing things.
2. Feeling down, depressed, or hopeless.
3. Trouble falling or staying asleep, or sleeping too much.
4. Feeling tired or having little energy.
5. Poor appetite or overeating.
6. Feeling bad about yourself — or that you are a failure or have let yourself or your family down.
7. Trouble concentrating on things, such as reading the newspaper or watching television.
8. Moving or speaking so slowly that other people could have noticed; or the opposite, being so fidgety or restless that you have been moving around a lot more than usual.

Important constraints:
- This is NOT a therapeutic or supportive interaction.
- Do NOT provide advice, reassurance, coping strategies, empathy, or encouragement.
- Do NOT attempt to diagnose, treat, or help the patient.
- Do NOT address the patient directly.
- Do NOT mention hotlines, therapy, or mental health resources.
- Do NOT add disclaimers or safety messages.
- Do NOT output a total PHQ score.
- Output only the eight item ratings.

Output format:
Return ONLY a valid JSON object with exactly these keys:

{
  "phq8_nointerest": 0,
  "phq8_depressed": 0,
  "phq8_sleep": 0,
  "phq8_tired": 0,
  "phq8_appetite": 0,
  "phq8_failure": 0,
  "phq8_concentrating": 0,
  "phq8_moving": 0
}

All values must be integers between 0 and 3.
Do not include any text outside the JSON object.
"""


def safe_strip(x):
    if x is None:
        return ""
    return str(x).strip()


def normalize_speaker(speaker):
    speaker = safe_strip(speaker).lower()

    if speaker in ["participant", "patient", "client"]:
        return "Patient"

    if speaker in ["therapist", "ellie", "interviewer", "clinician"]:
        return "Therapist"

    return None


def prepare_full_context(df):
    rows = []

    for synthetic_id, group in df.groupby(CONVERSATION_ID_COL, sort=False):
        transcript_lines = []

        for _, row in group.iterrows():
            speaker = normalize_speaker(row[SPEAKER_COL])
            text = safe_strip(row[TEXT_COL])

            if speaker is not None and text:
                transcript_lines.append(f"{speaker}: {text}")

        if len(transcript_lines) == 0:
            continue

        output_row = {
            CONVERSATION_ID_COL: synthetic_id,
            "context_text": "\n".join(transcript_lines[:-1]),
            "current_text": transcript_lines[-1],
            "target_PHQ8_Score": group["target_PHQ8_Score"].iloc[0],
        }

        for col in TARGET_ITEM_COLS:
            output_row[col] = group[col].iloc[0]

        rows.append(output_row)

    return pd.DataFrame(rows)


def extract_json_block(text):
    text = safe_strip(text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def parse_llm_output_to_dict(text):
    raw_text = safe_strip(text)

    result = {
        "raw_output": raw_text,
        "parse_success": False,
        "parse_error": None,
        "phq_score_indirect": None,
    }

    for col in PHQ8_ITEM_COLS:
        result[col] = None

    obj = extract_json_block(raw_text)

    if obj is None:
        result["parse_error"] = "No JSON found"
        return result

    try:
        parsed_items = IndirectPHQ8Items(**obj)

        for col in PHQ8_ITEM_COLS:
            result[col] = getattr(parsed_items, col)

        result["phq_score_indirect"] = sum(
            result[col] for col in PHQ8_ITEM_COLS
        )

        result["parse_success"] = True
        return result

    except Exception as e:
        result["parse_error"] = str(e)
        return result


def build_messages(context_text, current_text):
    user_text = (
        f"Conversation history:\n{context_text}\n\n"
        f"Current statement:\n{current_text}"
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]


def run_single_inference(context_text, current_text):
    messages = build_messages(context_text, current_text)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )

            raw_text = safe_strip(response.choices[0].message.content)
            return parse_llm_output_to_dict(raw_text)

        except Exception as e:
            if attempt == MAX_RETRIES:
                result = {
                    "raw_output": None,
                    "parse_success": False,
                    "parse_error": str(e),
                    "phq_score_indirect": None,
                }

                for col in PHQ8_ITEM_COLS:
                    result[col] = None

                return result


def process_one_row(row_idx, row, run_id):
    parsed = run_single_inference(
        context_text=str(row["context_text"]),
        current_text=str(row["current_text"]),
    )

    parsed["row_idx"] = row_idx
    parsed[CONVERSATION_ID_COL] = row[CONVERSATION_ID_COL]
    parsed["run_id"] = run_id
    parsed["target_PHQ8_Score"] = row["target_PHQ8_Score"]

    for col in TARGET_ITEM_COLS:
        parsed[col] = row[col]

    return parsed


df_long = pd.read_csv(INPUT_PATH)

df = prepare_full_context(df_long)
df = df.reset_index(drop=True)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

all_outputs = []

for run_id in range(1, N_RUNS + 1):
    parsed_outputs = [None] * len(df)

    for item_col in PHQ8_ITEM_COLS:
        df[f"{item_col}_run_{run_id}"] = pd.NA

    df[f"phq_score_indirect_run_{run_id}"] = pd.NA
    df[f"raw_output_run_{run_id}"] = pd.NA
    df[f"parse_success_run_{run_id}"] = pd.NA
    df[f"parse_error_run_{run_id}"] = pd.NA

    processed_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}

        for row_idx, row in df.iterrows():
            future = executor.submit(
                process_one_row,
                row_idx,
                row,
                run_id,
            )
            futures[future] = row_idx

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Processing transcripts - run {run_id}/{N_RUNS}",
        ):
            row_idx = futures[future]

            try:
                parsed = future.result()

            except Exception as e:
                parsed = {
                    "raw_output": None,
                    "parse_success": False,
                    "parse_error": str(e),
                    "phq_score_indirect": None,
                    "row_idx": row_idx,
                    CONVERSATION_ID_COL: df.loc[row_idx, CONVERSATION_ID_COL],
                    "run_id": run_id,
                    "target_PHQ8_Score": df.loc[row_idx, "target_PHQ8_Score"],
                }

                for item_col in PHQ8_ITEM_COLS:
                    parsed[item_col] = None

                for col in TARGET_ITEM_COLS:
                    parsed[col] = df.loc[row_idx, col]

            parsed_outputs[row_idx] = parsed
            all_outputs.append(parsed)

            for item_col in PHQ8_ITEM_COLS:
                df.loc[row_idx, f"{item_col}_run_{run_id}"] = parsed.get(item_col)

            df.loc[row_idx, f"phq_score_indirect_run_{run_id}"] = parsed.get("phq_score_indirect")
            df.loc[row_idx, f"raw_output_run_{run_id}"] = parsed.get("raw_output")
            df.loc[row_idx, f"parse_success_run_{run_id}"] = parsed.get("parse_success")
            df.loc[row_idx, f"parse_error_run_{run_id}"] = parsed.get("parse_error")

            processed_count += 1

            if processed_count % 100 == 0:
                df.to_csv(OUTPUT_PATH, index=False)

                all_outputs_df = pd.DataFrame(all_outputs)
                all_outputs_df.to_csv(LONG_OUTPUT_PATH, index=False)

                print(f"Zwischengespeichert: Run {run_id}, {processed_count} Transkripte fertig.")

    df.to_csv(OUTPUT_PATH, index=False)

    all_outputs_df = pd.DataFrame(all_outputs)
    all_outputs_df.to_csv(LONG_OUTPUT_PATH, index=False)

    print(f"Run {run_id} fertig gespeichert.")


for run_id in range(1, N_RUNS + 1):
    for item_col in PHQ8_ITEM_COLS:
        col = f"{item_col}_run_{run_id}"
        df[col] = pd.to_numeric(df[col], errors="coerce")

    score_col = f"phq_score_indirect_run_{run_id}"
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")


for item_col in PHQ8_ITEM_COLS:
    run_cols = [f"{item_col}_run_{i}" for i in range(1, N_RUNS + 1)]

    df[f"{item_col}_mean"] = df[run_cols].mean(axis=1)
    df[f"{item_col}_std"] = df[run_cols].std(axis=1)
    df[f"{item_col}_median"] = df[run_cols].median(axis=1)
    df[f"{item_col}_mean_rounded"] = df[f"{item_col}_mean"].round().astype("Int64")


score_run_cols = [f"phq_score_indirect_run_{i}" for i in range(1, N_RUNS + 1)]

df["phq_score_indirect_mean"] = df[score_run_cols].mean(axis=1)
df["phq_score_indirect_std"] = df[score_run_cols].std(axis=1)
df["phq_score_indirect_median"] = df[score_run_cols].median(axis=1)
df["phq_score_indirect_mean_rounded"] = df["phq_score_indirect_mean"].round().astype("Int64")

rounded_item_cols = [f"{item_col}_mean_rounded" for item_col in PHQ8_ITEM_COLS]

df["phq_score_from_rounded_items"] = df[rounded_item_cols].sum(axis=1).astype("Int64")

df.to_csv(OUTPUT_PATH, index=False)

all_outputs_df = pd.DataFrame(all_outputs)
all_outputs_df.to_csv(LONG_OUTPUT_PATH, index=False)

print("Fertig.")
print(df.shape)
print(all_outputs_df.shape)