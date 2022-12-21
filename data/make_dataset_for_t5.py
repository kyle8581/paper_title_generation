import json

for split in ["train", "valid", "test"]:
    with open(f"{split}.json", "r") as f:
        data = json.load(f)
    with open(f"{split}_t5.json", "w") as f:
        json.dump({
            "data": data
        }, f, indent=4)