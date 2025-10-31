import json
from pathlib import Path

source = Path("LearnOpenGL/LearnOpenGL/assets/mlp/mlp_weights.json")
target = Path("LearnOpenGL/LearnOpenGL/assets/mlp/mlp_weights.txt")

with source.open("r", encoding="utf-8") as f:
    data = json.load(f)

layers = data["layers"]
arch = data.get("architecture", {})
input_dim = arch.get("input_dim", len(layers[0]["weights"][0]))
hidden_dim = arch.get("hidden_dim", len(layers[0]["weights"]))
hidden_dim2 = arch.get("hidden_dim2", len(layers[1]["weights"]))
output_dim = arch.get("output_dim", len(layers[2]["weights"]))

with target.open("w", encoding="utf-8") as f:
    f.write("# MLP weight file\n")
    f.write(f"{input_dim} {hidden_dim} {hidden_dim2} {output_dim}\n")
    f.write(" ".join(str(x) for x in data["input_mean"]) + "\n")
    f.write(" ".join(str(x) for x in data["input_std"]) + "\n")
    for layer in layers:
        weights = layer["weights"]
        flat_w = []
        for row in weights:
            flat_w.extend(row)
        f.write(" ".join(str(x) for x in flat_w) + "\n")
        f.write(" ".join(str(x) for x in layer["bias"]) + "\n")

print(f"Converted weights to {target}")
