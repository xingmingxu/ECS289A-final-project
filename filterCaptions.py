import os
import json
import re
import hashlib
from tqdm import tqdm
from openai import OpenAI

import time
import random
from openai import RateLimitError

def llm_call_with_backoff(messages, model="gpt-4o-mini", max_retries=10):
    """Retry OpenAI calls with exponential backoff when hitting rate limits."""
    delay = 0.5

    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages
            )
        except RateLimitError as e:
            print(f"Rate limit hit. Retry {attempt+1}/{max_retries} in {delay:.2f}s...")
            time.sleep(delay)
            delay *= 2  # exponential backoff
            delay += random.uniform(0, 0.25)  # small jitter

    raise Exception("Exceeded max retries for LLM call")

client = OpenAI()

############################################
# Configuration
############################################
CAPTIONS_DIR = "dataset/captions/"
IMAGES_DIR = "dataset/images/"
OUTPUT_FILE = "filtered_dataset.json"

MIN_LEN = 20
MAX_LEN = 350
QUALITY_THRESHOLD = 4 # this is the threshold we used for "quality" of caption descriptiveness
BATCH_SIZE = 20

############################################
# LLM Scoring Prompt
############################################
SCORING_PROMPT = """
Your task is to evaluate whether a caption is a good training example
for a chart-analysis model.

A GOOD caption MUST:
- Objectively describe the chart content
- Mention trends, comparisons, or numeric information
- Directly correspond to what is shown in a chart
- Contain no opinions, headlines, or irrelevant information

A BAD caption:
- Is a generic headline
- Is unrelated to chart content
- Contains no numerical or comparative information
- Is too vague or too short
- Is an article snippet, not a chart description

Score from 1–5:
1 = unusable
2 = poor
3 = borderline
4 = good
5 = excellent

Respond ONLY with the number.
"""

############################################
# Utility Functions
############################################

def hash_text(text):
    """Deduplication hash."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def load_caption(txt_path):
    """Read caption text from a file."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except:
        return None


def find_matching_image(base):
    """Return matching image file for base (without extension)."""
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(IMAGES_DIR, base + ext)
        if os.path.exists(path):
            return path
    return None


############################################
# Heuristic Filters
############################################

def passes_heuristics(caption):
    if not caption:
        return False

    # Length filter
    if len(caption) < MIN_LEN or len(caption) > MAX_LEN:
        return False

    # Forbidden patterns
    forbidden = [
        r"statista",
        r"©",
        r"http[s]?:",
        r"all rights reserved",
        r"source:"
    ]
    for pat in forbidden:
        if re.search(pat, caption, re.IGNORECASE):
            return False

    # Must contain semantic info
    if not re.search(r"[A-Za-z]", caption):
        return False

    return True


############################################
# LLM Scoring
############################################

def score_captions_llm(batch):
    """Scores captions in a batch, one message per batch."""
    messages = []
    for c in batch:
        messages.append({
            "role": "user",
            "content": f"{SCORING_PROMPT}\n\nCaption:\n{c}"
        })

    response = llm_call_with_backoff(messages)

    scores = []
    for choice in response.choices:
        text = choice.message.content.strip()
        try:
            score = int(re.findall(r"\d", text)[0])
        except:
            score = 1  # default safe
        scores.append(score)

    return scores


############################################
# Main Pipeline
############################################

def main():
    txt_files = sorted(os.listdir(CAPTIONS_DIR))
    dataset = []

    print(f"Found {len(txt_files)} caption files")

    # Load & match
    for fname in txt_files:
        if not fname.endswith(".txt"):
            continue

        base = os.path.splitext(fname)[0]      # e.g., 000123
        txt_path = os.path.join(CAPTIONS_DIR, fname)
        image_path = find_matching_image(base)

        if image_path is None:
            continue  # skip if image missing

        caption = load_caption(txt_path)
        dataset.append({
            "id": base,
            "caption": caption,
            "image_path": image_path
        })

    print(f"Paired captions with images: {len(dataset)}")

    ############################################
    # 1. Heuristic Filtering
    ############################################
    filtered = [
        item for item in dataset
        if passes_heuristics(item["caption"])
    ]
    print(f"After heuristics: {len(filtered)} remaining")

    ############################################
    # 2. Deduplication
    ############################################
    seen = set()
    unique = []
    for item in filtered:
        h = hash_text(item["caption"])
        if h not in seen:
            seen.add(h)
            unique.append(item)

    print(f"After dedupe: {len(unique)} remaining")

    ############################################
    # 3. LLM Scoring
    ############################################
    final = []
    for i in tqdm(range(0, len(unique), BATCH_SIZE)):
        batch = unique[i:i + BATCH_SIZE]
        captions = [item["caption"] for item in batch]
        scores = score_captions_llm(captions)

        for item, s in zip(batch, scores):
            if s >= QUALITY_THRESHOLD:
                item["quality_score"] = s
                final.append(item)

    print(f"Final retained examples: {len(final)}")

    ############################################
    # 4. Save as JSON
    ############################################
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
