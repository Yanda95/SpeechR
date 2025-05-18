import argparse
import json
import re
import string
import pandas as pd
from typing import List, Optional, Tuple


def strict_map(output: str, options: List[str], label: Optional[int] = None) -> Optional[str]:
    """
    Extract the predicted letter from model output with robust pattern matching.

    Args:
        output (str): Model's output string.
        options (List[str]): List of candidate answer options.
        label (Optional[int]): Ground-truth label (1-based index).

    Returns:
        Optional[str]: The predicted letter (A, B, C, ...) or None if unmatchable.
    """
    text = output.strip()
    text_lower = text.lower()
    letters = list(string.ascii_uppercase[:len(options)])
    normalized_options = [opt.lower().strip().strip('.') for opt in options]

    # Step 1: Exact pattern-based letter extraction
    patterns = [
        r"Answer[:：]?\s*([A-Z])\b",
        r"Final Answer[:：]?\s*([A-Z])\b",
        r"Answer\s+is\s+([A-Z])\b",
        r"The answer is\s+([A-Z])\b",
        r"Choice[:：]?\s*([A-Z])\b",
        r"Selected[:：]?\s*([A-Z])\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in letters:
                return letter

    # Step 2: Loose matching like "A." or "B)"
    matches = re.findall(r"\b([A-Z])[.)]?\b", text.upper())
    filtered = [ch for ch in matches if ch in letters]
    if len(filtered) == 1:
        return filtered[0]

    # Step 3: Match full option text in the output
    for i, opt in enumerate(normalized_options):
        if opt in text_lower:
            return letters[i]

    # Step 4: Fallback for binary questions (yes/no, true/false)
    if set(normalized_options) in [{'yes', 'no'}, {'true', 'false'}] and isinstance(label, int) and 1 <= label <= len(options):
        gold_option = normalized_options[label - 1]
        positive = {"yes", "true"}
        negative = {"no", "false"}

        has_pos = any(kw in text_lower for kw in positive)
        has_neg = any(kw in text_lower for kw in negative)

        if has_pos and has_neg:
            return None  # Ambiguous case

        if gold_option in positive and has_pos:
            return letters[normalized_options.index(gold_option)]
        if gold_option in negative and has_neg:
            return letters[normalized_options.index(gold_option)]

    return None


def evaluate_strict(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate model predictions using strict extraction.

    Args:
        df (pd.DataFrame): DataFrame with columns: dataset, options, label, predict_answer

    Returns:
        Tuple: (summary DataFrame, detailed evaluation DataFrame)
    """
    records = []
    for _, row in df.iterrows():
        options = row['options']
        label = row['label']
        gt_letter = chr(ord('A') + (label - 1)) if isinstance(label, int) and 1 <= label <= len(options) else None
        pred_letter = strict_map(row['predict_answer'], options, label)

        records.append({
            'dataset': row['dataset'],
            'predict_answer': row['predict_answer'],
            'gt_letter': gt_letter,
            'pred_letter': pred_letter,
            'correct': pred_letter == gt_letter
        })

    detailed_df = pd.DataFrame(records)
    summary_df = detailed_df.groupby('dataset').agg(
        total=('correct', 'size'),
        valid=('pred_letter', lambda x: x.notnull().sum()),
        correct=('correct', 'sum')
    ).reset_index()
    summary_df['accuracy'] = summary_df['correct'] / summary_df['total']
    return summary_df, detailed_df


def main(input_file: str):
    # Load input JSONL file
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            parts = item.get('id', '').split('-')
            dataset = f"{parts[0]}-{parts[1]}" if parts[0] == 'moral' and parts[1] in ('easy', 'hard') else parts[0]

            data.append({
                'dataset': dataset,
                'options': item.get('options', []),
                'label': item.get('label'),
                'predict_answer': item.get('predict_answer', '') or ''
            })

    df = pd.DataFrame(data)

    # Evaluate predictions
    summary_df, detailed_df = evaluate_strict(df)

    # Print detailed prediction analysis
    print("=== Detailed Results ===")
    for _, row in detailed_df.iterrows():
        print(f"[{row['dataset']}] predict_answer: {repr(row['predict_answer'])}")
        print(f"    => extracted: {repr(row['pred_letter'])}, ground-truth: {repr(row['gt_letter'])} {'✔' if row['correct'] else '✘'}\n")

    # Print dataset-wise summary
    print("=== Summary by Dataset ===")
    print(summary_df.to_string(index=False))

    # Print overall accuracy
    total = summary_df['total'].sum()
    correct = summary_df['correct'].sum()
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"\nOverall Accuracy: {acc:.2f}% ({correct}/{total})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Strict evaluation for multiple-choice reasoning tasks.")
    parser.add_argument('-i', '--input', default="audio_core_multi-choice.jsonl",
                        help='Path to input JSONL file containing fields: id, options, label, predict_answer')
    args = parser.parse_args()
    main(args.input)
