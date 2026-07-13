import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import json
import re
from typing import Literal
from pydantic import BaseModel, Field
from tqdm import tqdm

# =========================
# SCHEMA FÜR DIE AUSGABE
# =========================
class DirectPHQScore(BaseModel):
    # reasoning: str = Field(...)
    phq_score: int = Field(..., ge=0, le=27)

# JSON Schema für das Forcieren
OUTPUT_SCHEMA = DirectPHQScore.model_json_schema()

# =========================
# KONFIGURATION
# =========================
INPUT_PATH = "/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/INPUT/DAIC_FULL_NO_TAGS.csv"
OUTPUT_PATH = "/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/OUTPUT/QWEN-2B.csv"

MODEL_NAME = "Qwen/Qwen3.5-2B"

# Passe diese Spaltennamen an deine CSV an, falls sie anders heißen:
CONVERSATION_ID_COL = "Participant_ID"

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
def add_cumulative_context(df: pd.DataFrame, text_col: str, max_turns: int = 40) -> pd.DataFrame:
    context_texts = []

    for _, group in df.groupby("Participant_ID", sort=False):
        history = []

        for _, row in group.iterrows():
            context = "\n".join(history[-max_turns:])
            context_texts.append(context)
            history.append(str(row[text_col]))

    df = df.copy()
    df["context_text"] = context_texts
    return df

# =========================
# MODELL
# =========================
def load_model_and_processor():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="cuda",
    )
    model.eval()
    return processor, model

# =========================
# INFERENCE
# =========================
def build_messages(context_text: str, current_text: str):
    user_text = (
        f"Conversation history:\n{context_text}\n\n"
        f"Current patient statement:\n{current_text}"
    )

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]

def extract_json_block(text: str):
    text = text.strip()

    # Direkt JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except:
        pass

    # JSON in Code Fence
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except:
            pass

    # Erstes JSON-artiges Objekt
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except:
            pass

    return None

def parse_phq_score(text: str):
    obj = extract_json_block(text)
    if obj:
        try:
            # Validierung mit Pydantic
            score_obj = DirectPHQScore(**obj)
            return score_obj
        except Exception:
            pass

    # Fallback: nackte Zahl
    m = re.search(r"\b([0-9]+)\b", text)
    if m:
        try:
            value = int(m.group(1))
            if 0 <= value <= 27:
                return value
        except:
            pass

    return None

def run_single_inference(processor, model, context_text: str, current_text: str):
    messages = build_messages(context_text, current_text)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=82, pad_token_id=248044)

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_text = processor.decode(generated_tokens, skip_special_tokens=True).strip()

    return parse_phq_score(raw_text)

# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv(INPUT_PATH)
    df = add_cumulative_context(df, "value", 200)

    processor, model = load_model_and_processor()

    scores = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        score = run_single_inference(processor, model, str(row["context_text"]), str(row["value"]))
        scores.append(score)

    df["phq_score"] = scores
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()