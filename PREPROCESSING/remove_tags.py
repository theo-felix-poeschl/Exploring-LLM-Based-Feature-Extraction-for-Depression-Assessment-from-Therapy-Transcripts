import re
import pandas as pd
from pathlib import Path

INPUT_PATH = "/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/INPUT/DAIC_FULL_NO_TAGS.csv"
OUTPUT_PATH = "/home/jovyan/Exploring-LLM-Based-Feature-Extraction-for-Depression-Assessment-from-Therapy-Transcripts/INPUT/DAIC_FULL_NO_TAGS.csv"
 
NONVERBAL_TAGS = ['laughter']  

def clean_transcript_line(text):
    if pd.isna(text):
        return None

    text = str(text).strip()

    # 1. Remove technical marker (e.g. <sync>, <laughter>)
    text = re.sub(r"<[^>]+>", " ", text).strip()

    if not text:
        return None

    # 2. Remove Ellie-Intent-Labels, keep spoken content
    # Example: how_doingV (so how are you doing today) -> so how are you doing today
    match = re.match(r"^[A-Za-z][A-Za-z0-9_]*\s*\((.*)\)\s*$", text)
    if match:
        text = match.group(1).strip()

    # 3. Remove non-verbal tags
    # Example: (laughter) -> None
    tag_match = re.fullmatch(r"\((.*?)\)", text.strip())
    if tag_match:
        tag_content = tag_match.group(1).strip().lower()
        if tag_content in NONVERBAL_TAGS:
            return None

    # 4. Remove non-verbal tags if inside an utterance
    # Example: yeah (laughter) I guess -> yeah I guess
    def remove_nonverbal_tag(match):
        content = match.group(1).strip().lower()
        if content in NONVERBAL_TAGS:
            return " "
        return match.group(0)

    text = re.sub(r"\((.*?)\)", remove_nonverbal_tag, text)

    # 5. Whitespace normalization
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else None
 
if __name__ == '__main__':
    df = pd.read_csv(INPUT_PATH)
    out = df.copy()
    out['value'] = out['value'].apply(clean_transcript_line)
    # drop lines if transcript cleaning resulted in empty line
    out = out.dropna()
    out.to_csv(OUTPUT_PATH, index=False)