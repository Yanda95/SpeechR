#!/usr/bin/env python3
# evaluate_with_gpt4o.py

import json
import time
import argparse
import openai
import os
from typing import Dict, Any, List, Optional

# -----------------------------------------------------------------------------
# 1. Configure OpenAI API and file paths
# -----------------------------------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY") or "sk-..."  # Set your key via env variable if preferred

# -----------------------------------------------------------------------------
# 2. GPT-4o call with fallback JSON parsing
# -----------------------------------------------------------------------------
def call_gpt4o(prompt: str) -> Dict[str, Any]:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )
    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])

# -----------------------------------------------------------------------------
# 3. Evaluate a single QA sample
# -----------------------------------------------------------------------------
def eval_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    rec_id = sample.get("id", "")
    question = sample.get("question", sample.get("dialogue", "")).strip()
    prediction = sample.get("predict_answer", "").strip()

    # Reference answer extraction
    if rec_id.startswith("dilemma-"):
        # Join list of values directly as the reference
        label_list = sample.get("label", [])
        reference = ", ".join(label_list) if isinstance(label_list, list) else str(label_list)
    else:
        options = sample.get("options", [])
        label = sample.get("label")
        reference = ""
        if isinstance(label, int) and 1 <= label <= len(options):
            reference = options[label - 1]
        elif isinstance(label, str):
            reference = label
        else:
            reference = sample.get("gold_answer", "")

    # Chain-of-thought content
    cot = sample.get("predict_cot", [])
    cot_text = "\n".join(cot) if isinstance(cot, list) else ""

    # Compose evaluation prompt
    prompt = f"""
You are an expert evaluator. Given the question, model prediction, and its chain-of-thought (if any), provide the following metrics in JSON:
1. final_correct:
   - For yes/no questions (reference is "Yes" or "No"), interpret the overall stance in the prediction and compare.
   - For classification/list tasks, 1 if the prediction mentions at least one reference item or is semantically equivalent; else 0.
2. logic_relevance: an integer from 1 to 5 indicating how strongly the prediction logically follows the question:
   1 = no relevance
   2 = very weak relevance
   3 = moderate relevance
   4 = strong relevance
   5 = very strong relevance
3. coherence_score: an integer from 1 to 5 assessing the chain-of-thought coherence (only if provided):
   1 = no coherence or no chain-of-thought(disjointed, illogical)
   2 = low coherence (many gaps)
   3 = moderate coherence (some gaps)
   4 = high coherence (clear, minor issues)
   5 = excellent coherence (very clear, logical flow)

Respond exactly with:
{{"final_correct":<0 or 1>,"logic_relevance":<1-5>,"coherence_score":<1-5>}}

Question:
{question}

Reference Answer:
{reference}

Model Prediction:
{prediction}

"""
    print(f"Evaluating {rec_id}...")
    metrics = call_gpt4o(prompt)

    # Safe parse
    try:
        return {
            "final_correct": int(metrics.get("final_correct", 0)),
            "logic_relevance": int(metrics.get("logic_relevance", 1)),
            "coherence_score": int(metrics.get("coherence_score", 0)) if "coherence_score" in metrics else None
        }
    except Exception as e:
        print(f"Parsing failed for {rec_id}: {e}")
        return {"final_correct": 0, "logic_relevance": 1, "coherence_score": None}

# -----------------------------------------------------------------------------
# 4. Batch evaluation with resume support
# -----------------------------------------------------------------------------
def main(input_path: str, output_path: str, pause_s: float):
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f_out:
            for line in f_out:
                try:
                    record = json.loads(line)
                    processed_ids.add(record.get("id", ""))
                except:
                    continue

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out:
        for line in f_in:
            sample = json.loads(line)
            rec_id = sample.get("id", "")
            if rec_id in processed_ids:
                continue

            # Optional: Only evaluate certain categories
            if not any(rec_id.startswith(prefix) for prefix in ["revealcot-", "dilemma-", "gsm8k-", "moral-"]):
                continue

            # Evaluate and write result
            metrics = eval_sample(sample)
            sample.update(metrics)
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"✓ {rec_id} - final_correct={metrics['final_correct']}, "
                  f"logic_relevance={metrics['logic_relevance']}, coherence_score={metrics['coherence_score']}")
            processed_ids.add(rec_id)
            time.sleep(pause_s)

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate reasoning tasks using GPT-4o with strict metrics.")
    parser.add_argument("-i", "--input", default="audio_core_generative.jsonl", help="Input JSONL file path")
    parser.add_argument("-o", "--output", default="generative_result.jsonl", help="Output JSONL file path")
    parser.add_argument("-p", "--pause", type=float, default=1.0, help="Pause in seconds between API calls")
    args = parser.parse_args()
    main(args.input, args.output, args.pause)
