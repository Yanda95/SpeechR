# 🧠 Audio-CORE

**Audio-CORE** is a multimodal benchmark designed to evaluate the reasoning capabilities of large audio-language models (LALMs) across factual, procedural, and normative dimensions. It supports both structured and open-ended evaluation in realistic, dialogue-driven audio scenarios.

## 📦 Dataset Overview

**🔗 Download Link**: [Google Drive](https://drive.google.com/file/d/1BH2r2idILwUHX0NKsXz6GsSXdO0qWly8/view?usp=sharing)

This dataset includes:
- Audio data for both **multi-choice** and **generative** versions (they share the same audio recordings).
- Three JSONL files:
  - `multi_choice.jsonl`: Multiple-choice questions with options and labels.
  - `generative.jsonl`: Open-ended version for free-form answer generation.
  - `mini.jsonl`: A 10% subset with added prosody annotations (stress, emotion).

## 📊 Evaluation Scripts

### 🔹 Rule-Based Evaluation (for Multi-Choice and Mini Versions)

This script uses symbolic rules to extract and evaluate predicted answers from model outputs.

```bash
python rule-based-evaluate.py --input-file multi_choice.jsonl 
```

### 🔹 LLM-based Evaluation (for Generative Version, take gpt-4o as example)
```
python llm-evaluate.py --input-file generative.jsonl --api-key YOUR_OPENAI_KEY
```
