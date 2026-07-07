import os
import pandas as pd
import json
import re
from pydantic import BaseModel, Field
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

N_RUNS = 5
MAX_WORKERS = 25
MAX_RETRIES = 3

# =========================
# SCHEMA FÜR DIE AUSGABE
# =========================
class DirectPHQScore(BaseModel):
    phq_score: int = Field(..., ge=0, le=27, description="PHQ Score = Severity score for depression")

OUTPUT_SCHEMA = DirectPHQScore.model_json_schema()

# =========================
# KONFIGURATION
# =========================
INPUT_PATH = "/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/INPUT/DAIC_FULL_NO_TAGS.csv"
OUTPUT_PATH = "/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/OUTPUT/direct_phq_no_reasoning.csv"
LONG_OUTPUT_PATH = "/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/OUTPUT/direct_phq_no_reasoning_long.csv"

client = OpenAI(
    base_url="https://llmchat.idm.uk-augsburg.science/api",
    api_key="sk-ab5121e5ffda4b76bdf1240d07e552f9",
)

MODEL_NAME = "openai/gpt-oss-120b"

CONVERSATION_ID_COL = "Participant_ID"
TEXT_COL = "value"

# =========================
# PROMPT
# =========================
SYSTEM_PROMPT = """
Task: You will be given a whole transcript of a therapy session. The patient's utterances are labeled with "Patient: ", the utterances from the therapist with "Therapist: ".

Based on this transcript, rate the Patient's PHQ (Patient-Health Questionnaire) score. The PHQ score is calculated by summing the ratings of the following items, each rated on a scale from 0 to 3:
    1. Little interest or pleasure in doing things.
    2. Feeling down, depressed, or hopeless.
    3. Trouble falling or staying asleep, or sleeping too much.
    4. Feeling tired or having little energy.
    5. Poor appetite or overeating.
    6. Feeling bad about yourself - or that you are a failure or have let yourself or your family down.
    7. Trouble concentrating on things, such as reading the newspaper or watching television.
    8. Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual.
    9. Thoughts that you would be better off dead or of hurting yourself in some way.

Important constraints:
- This is NOT a therapeutic or supportive interaction.
- Do NOT provide advice, reassurance, coping strategies, empathy, or encouragement.
- Do NOT attempt to diagnose, treat, or help the patient.
- Do NOT address the patient directly.
- Do NOT mention hotlines, therapy, or mental health resources.
- Do NOT add disclaimers or safety messages.

Output format:
Return ONLY a valid JSON object that conforms to the following schema:
{
  "type": "object",
  "properties": {
    "phq_score": {
      "type": "integer",
      "minimum": 0,
      "maximum": 27
    }
  },
  "required": ["phq_score"]
}

Do not include any text outside the JSON object.
"""

# =========================
# DATA PREP
# =========================

def parse_llm_output_to_dict(text):
    raw_text = safe_strip(text)

    result = {
        "phq_score": None,
        "raw_output": raw_text,
        "parse_success": False,
        "parse_error": None
    }

    obj = extract_json_block(raw_text)

    if obj is None:
        result["parse_error"] = "No JSON found"

        # Fallback: irgendeine Zahl zwischen 0 und 27 suchen
        m = re.search(r"\b([0-9]+)\b", raw_text)
        if m:
            value = int(m.group(1))
            if 0 <= value <= 27:
                result["phq_score"] = value
                result["parse_success"] = True
                result["parse_error"] = "Used numeric fallback"

        return result

    # phq_score parsen
    try:
        score = int(obj.get("phq_score"))
        if 0 <= score <= 27:
            result["phq_score"] = score
        else:
            result["parse_error"] = "phq_score outside 0-27"
            return result
    except Exception:
        result["parse_error"] = "Could not parse phq_score"
        return result

    result["parse_success"] = True

    return result

def add_cumulative_context(df, text_col, id_col="Participant_ID"):
    context_texts = []

    for _, group in df.groupby(id_col, sort=False):
        history = []

        for _, row in group.iterrows():
            context = "\n".join(history)
            context_texts.append(context)
            history.append(str(row[text_col]))

    df = df.copy()
    df["context_text"] = context_texts
    return df

# =========================
# INFERENCE
# =========================
def build_messages(context_text, current_text):
    user_text = (
        f"Conversation history:\n{context_text}\n\n"
        f"Current patient statement:\n{current_text}"
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

def extract_json_block(text):
    text = text.strip()

    # Direkt JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # JSON in Code Fence
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Erstes JSON-artiges Objekt
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None

def parse_phq_score(text):
    obj = extract_json_block(text)

    if obj:
        try:
            score_obj = DirectPHQScore(**obj)
            return score_obj.phq_score
        except Exception:
            pass

    # Fallback: nackte Zahl
    m = re.search(r"\b([0-9]+)\b", text)
    if m:
        try:
            value = int(m.group(1))
            if 0 <= value <= 27:
                return value
        except Exception:
            pass

    return None

def safe_strip(x):
    if x is None:
        return ""
    return str(x).strip()

def run_single_inference(context_text, current_text):
    messages = build_messages(context_text, current_text)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages
            )

            raw_text = safe_strip(response.choices[0].message.content)
            return parse_llm_output_to_dict(raw_text)

        except Exception as e:
            if attempt == MAX_RETRIES:
                return {
                    "phq_score": None,
                    "raw_output": None,
                    "parse_success": False,
                    "parse_error": str(e)
                }
                
def process_one_row(row_idx, row, run_id):
    parsed = run_single_inference(
        context_text=str(row["context_text"]),
        current_text=str(row[TEXT_COL]),
    )

    parsed["row_idx"] = row_idx
    parsed[CONVERSATION_ID_COL] = row[CONVERSATION_ID_COL]
    parsed["run_id"] = run_id

    return parsed

# =========================
# MAIN
# =========================

df = pd.read_csv(INPUT_PATH)

df = add_cumulative_context(
    df,
    text_col=TEXT_COL,
    id_col=CONVERSATION_ID_COL,
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

all_outputs = []

for run_id in range(1, N_RUNS + 1):
    parsed_outputs = [None] * len(df)

    # Spalten für diesen Run schon vorher anlegen
    df[f"phq_score_run_{run_id}"] = pd.NA
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
                run_id
            )
            futures[future] = row_idx

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Processing rows - run {run_id}/{N_RUNS}"
        ):
            row_idx = futures[future]

            try:
                parsed = future.result()

            except Exception as e:
                parsed = {
                    "phq_score": None,
                    "raw_output": None,
                    "parse_success": False,
                    "parse_error": str(e),
                    "row_idx": row_idx,
                    CONVERSATION_ID_COL: df.loc[row_idx, CONVERSATION_ID_COL],
                    "run_id": run_id
                }

            parsed_outputs[row_idx] = parsed
            all_outputs.append(parsed)

            # Ergebnis direkt in df schreiben
            df.loc[row_idx, f"phq_score_run_{run_id}"] = parsed["phq_score"]
            df.loc[row_idx, f"raw_output_run_{run_id}"] = parsed["raw_output"]
            df.loc[row_idx, f"parse_success_run_{run_id}"] = parsed["parse_success"]
            df.loc[row_idx, f"parse_error_run_{run_id}"] = parsed["parse_error"]

            processed_count += 1

            # Alle 100 fertig bearbeiteten Zeilen zwischenspeichern
            if processed_count % 100 == 0:
                df.to_csv(OUTPUT_PATH, index=False)

                all_outputs_df = pd.DataFrame(all_outputs)
                all_outputs_df.to_csv(LONG_OUTPUT_PATH, index=False)

                print(f"Zwischengespeichert: Run {run_id}, {processed_count} Zeilen fertig.")

    # Am Ende jedes Runs nochmal speichern
    df.to_csv(OUTPUT_PATH, index=False)

    all_outputs_df = pd.DataFrame(all_outputs)
    all_outputs_df.to_csv(LONG_OUTPUT_PATH, index=False)

    print(f"Run {run_id} fertig gespeichert.")


# =========================
# AGGREGATION
# =========================

run_cols = [f"phq_score_run_{i}" for i in range(1, N_RUNS + 1)]

df["phq_score_mean"] = df[run_cols].mean(axis=1)
df["phq_score_std"] = df[run_cols].std(axis=1)
df["phq_score_median"] = df[run_cols].median(axis=1)
df["phq_score_mean_rounded"] = df["phq_score_mean"].round().astype("Int64")

df.to_csv(OUTPUT_PATH, index=False)

all_outputs_df = pd.DataFrame(all_outputs)
all_outputs_df.to_csv(LONG_OUTPUT_PATH, index=False)

print("Fertig.")
print(df.shape)
print(all_outputs_df.shape)