# Medical Language Model

A specialized language model for medical question answering based on patient symptom descriptions.

## 📌 Project Overview

This project implements a GPT-based language model fine-tuned to provide medical responses to patient queries. The model is trained on structured medical conversation data to offer relevant medical advice based on symptom descriptions.

## 📂 Data Structure

### JSON Format
The training data follows this structured format:

```json
[
  {
    "instruction": "Medical response directive",
    "input": "Patient's symptom description",
    "output": "Doctor's professional response"
  }
]
```

### Installation
```bash
pip install transformers datasets
```

### Key Components

**Model Architecture**
- Base Model: GPT-2
- Tokenizer: GPT-2 with custom padding token
- Training: Hugging Face Trainer API
