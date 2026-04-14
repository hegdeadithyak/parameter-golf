from datasets import load_dataset

ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

small_text = []
for i, example in enumerate(ds):
    small_text.append(example["text"])
    if i >= 2000:   # ~1–5MB depending on text
        break

with open("tiny.txt", "w") as f:
    f.write("\n".join(small_text))