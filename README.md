# 🧠 SpeechR

**SpeechR** is a multimodal benchmark designed to evaluate the reasoning capabilities of large audio-language models (LALMs) across factual, procedural, and normative dimensions. It supports both structured and open-ended evaluation in realistic, dialogue-driven audio scenarios.

## 📦 Dataset Overview

**🔗 Download Link**: [Google Drive](https://drive.google.com/file/d/1BH2r2idILwUHX0NKsXz6GsSXdO0qWly8/view?usp=sharing)

**Mini-Human Download Link**: [Google Drive](https://drive.google.com/file/d/1dyjkrxJAn8gDC_DVMegFT9vvqhs7N9gt/view?usp=sharing)

This dataset includes:
- Speech data for both **multi-choice** and **generative** versions (they share the same audio recordings).
- Three JSONL files:
  - `multi_choice.jsonl`: Multiple-choice questions with options and labels.
  - `generative.jsonl`: Open-ended version for free-form answer generation.
  - `acoustic-feature.jsonl`: A 10% subset with added prosody annotations (stress, emotion).

## Appendix Summary

The appendix of SpeechR provides detailed insights into dataset construction, evaluation, and analysis. It defines three core reasoning types—factual, procedural, and normative—based on knowledge source, reasoning complexity, and answer objectivity. It also describes data processing steps such as readability enhancement, conversational restructuring, and annotation of acoustic features (e.g., stress and emotion).

Additionally, the appendix introduces source datasets, model baselines, and prompt templates used for data generation and evaluation. It includes human verification results showing high data quality, as well as ablation studies demonstrating that speech-based reasoning is significantly more challenging than text-based reasoning. Finally, it presents qualitative examples, discusses limitations of current benchmarks, and outlines future directions such as improving speech diversity, multilingual coverage, and interactive dialogue settings. :contentReference[oaicite:0]{index=0}

## 📊 Evaluation Scripts

### 🔹 Discrete-choice Evaluation (for Multi-Choice and Acoustic-feature Versions)

This script uses symbolic rules to extract and evaluate predicted answers from model outputs.

```bash
python discrete-choice_eval.py --input speechr_multi_choice.jsonl 
```

### 🔹 LLM-as-a-judge Evaluation (for Generative Version, take gpt-4o as example)
```
python llm-as-a-judge_eval.py --input speechr_generative.jsonl --api-key YOUR_OPENAI_KEY
```
