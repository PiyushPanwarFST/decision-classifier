from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
)

LABELS = ["making a decision or choice", "a general statement or question"]


def classify_prompt(prompt):
    result = classifier(
        prompt,
        candidate_labels=LABELS,
        hypothesis_template="The speaker is expressing {}.",
    )
    return {
        "prompt": prompt,
        "label": result["labels"][0],
        "confidence": round(result["scores"][0], 4),
        "all_scores": dict(zip(result["labels"], result["scores"])),
    }


if __name__ == "__main__":
    demo = classify_prompt("I want to buy an Audi")
    print(f"Prompt:     {demo['prompt']}")
    print(f"Label:      {demo['label']}")
    print(f"Confidence: {demo['confidence']}")
    print(f"All scores: {demo['all_scores']}")
