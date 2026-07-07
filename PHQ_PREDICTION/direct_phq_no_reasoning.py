import os
import pandas as pd
import json
import re
from pydantic import BaseModel, Field
from tqdm import tqdm
from openai import OpenAI

N_RUNS = 5

# =========================
# SCHEMA FÜR DIE AUSGABE
# =========================
class DirectPHQScore(BaseModel):
    phq_score: int = Field(..., ge=0, le=27)
    reasoning: int = Field(...)

OUTPUT_SCHEMA = DirectPHQScore.model_json_schema()

# =========================
# KONFIGURATION
# =========================
INPUT_PATH = "/home/jovyan/thesisnew-datavol-1/input/DAIC_FULL_NO_TAGS.csv"
OUTPUT_PATH = "/home/jovyan/thesisnew-datavol-1/output/DIRECT_PHQ/direct_phq_scores_1.csv"

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
    "reasoning": {
        "type": "string"
    } 
  },
  "required": ["phq_score", "reasoning"]
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
        "reasoning": None,
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

    # reasoning parsen
    reasoning = obj.get("reasoning", None)

    if reasoning is not None:
        reasoning = str(reasoning).strip()

    result["reasoning"] = reasoning
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

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )

    raw_text = safe_strip(response.choices[0].message.content)

    return parse_llm_output_to_dict(raw_text)

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
    parsed_outputs = []

    for row_idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Processing rows - run {run_id}/{N_RUNS}"
    ):
        parsed = run_single_inference(
            context_text=str(row["context_text"]),
            current_text=str(row[TEXT_COL]),
        )

        parsed["row_idx"] = row_idx
        parsed[CONVERSATION_ID_COL] = row[CONVERSATION_ID_COL]
        parsed["run_id"] = run_id

        parsed_outputs.append(parsed)
        all_outputs.append(parsed)

    run_output_df = pd.DataFrame(parsed_outputs)

    df[f"phq_score_run_{run_id}"] = run_output_df["phq_score"].values
    df[f"reasoning_run_{run_id}"] = run_output_df["reasoning"].values
    df[f"raw_output_run_{run_id}"] = run_output_df["raw_output"].values
    df[f"parse_success_run_{run_id}"] = run_output_df["parse_success"].values

    df.to_csv(OUTPUT_PATH, index=False)

all_outputs_df = pd.DataFrame(all_outputs)

all_outputs_df.to_csv(
    "/home/jovyan/thesisnew-datavol-1/output/DIRECT_PHQ/direct_phq_outputs_long.csv",
    index=False
)