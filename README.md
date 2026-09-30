# Decision Prompt Classifier

A zero-shot text classifier that determines whether a user prompt involves **decision-making** or not — without any training or fine-tuning.

## How It Works

Uses a pre-trained NLI (Natural Language Inference) model [`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) from HuggingFace. The model checks if a given prompt entails "making a decision or choice" or is "a general statement or question".

No training data needed — the model understands natural language and reasons about the labels directly.

### Examples

| Prompt | Classification | Confidence |
|---|---|---|
| "I want to buy an Audi" | decision | 0.87 |
| "I've decided to quit my job" | decision | 0.98 |
| "What is the capital of France?" | non-decision | 0.99 |
| "The weather is nice today" | non-decision | 0.99 |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Classify a single prompt

```python
from classifier import classify_prompt

result = classify_prompt("I want to buy an Audi")
print(result)
# {'prompt': 'I want to buy an Audi', 'label': 'making a decision or choice', 'confidence': 0.8756, ...}
```

### Run evaluation on test dataset

```bash
python evaluate.py
```

## Results

Tested on 30 hand-labeled prompts (15 decision + 15 non-decision):

- **Accuracy: 93.3% (28/30)**
- Decision prompts: 15/15 correct
- Non-decision prompts: 13/15 correct

## Files

| File | Description |
|---|---|
| `classifier.py` | Main classifier — loads model, provides `classify_prompt()` |
| `test_dataset.py` | 30 labeled test prompts |
| `evaluate.py` | Runs classifier on test data, prints accuracy report |
| `requirements.txt` | Python dependencies |
