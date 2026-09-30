from classifier import classify_prompt
from test_dataset import TEST_DATA

correct = 0
wrong = 0
wrong_examples = []

print("=" * 70)
print("Running classifier on test dataset...")
print("=" * 70)

for prompt, expected in TEST_DATA:
    result = classify_prompt(prompt)
    predicted = result["label"]
    confidence = result["confidence"]

    if predicted == expected:
        correct += 1
        print(f"[CORRECT] {prompt}")
        print(f"          → {predicted} ({confidence})")
    else:
        wrong += 1
        print(f"[WRONG]   {prompt}")
        print(f"          → {predicted} ({confidence}) | expected: {expected}")
        wrong_examples.append((prompt, predicted, expected, confidence))
    print()

print("=" * 70)
print(f"Total:    {len(TEST_DATA)}")
print(f"Correct:  {correct}")
print(f"Wrong:    {wrong}")
print(f"Accuracy: {correct / len(TEST_DATA) * 100:.1f}%")
print("=" * 70)

if wrong_examples:
    print("\nFailed cases:")
    for prompt, predicted, expected, confidence in wrong_examples:
        print(f"  - \"{prompt}\"")
        print(f"    Predicted: {predicted} ({confidence}) | Expected: {expected}")
