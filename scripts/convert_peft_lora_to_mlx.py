"""Convert a PEFT LoRA adapter into the adapter format expected by mlx-lm."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open


KEY_RE = re.compile(
    r"^base_model\.model\.model\.(?P<prefix>.+?)\.layers\."
    r"(?P<layer>\d+)\.(?P<module>.+?)\.lora_(?P<ab>[AB])\.weight$"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PEFT LoRA adapter to MLX LoRA adapter.")
    parser.add_argument("peft_adapter", help="Path containing PEFT adapter_model.safetensors")
    parser.add_argument("mlx_adapter", help="Output path for MLX adapters.safetensors")
    args = parser.parse_args()

    peft_path = Path(args.peft_adapter)
    mlx_path = Path(args.mlx_adapter)
    mlx_path.mkdir(parents=True, exist_ok=True)

    peft_config = json.loads((peft_path / "adapter_config.json").read_text())
    rank = int(peft_config["r"])
    alpha = float(peft_config["lora_alpha"])
    scale = alpha / rank
    target_modules = list(peft_config["target_modules"])

    weights = {}
    layer_ids = set()
    module_keys = set()
    with safe_open(peft_path / "adapter_model.safetensors", framework="numpy") as handle:
        for key in handle.keys():
            match = KEY_RE.match(key)
            if not match:
                continue
            layer = int(match.group("layer"))
            module = match.group("module")
            ab = match.group("ab")
            layer_ids.add(layer)
            module_keys.add(module)
            tensor = handle.get_tensor(key)
            mlx_key = f"language_model.model.layers.{layer}.{module}.lora_{ab.lower()}"
            weights[mlx_key] = mx.array(tensor.T)

    if not weights:
        raise SystemExit(f"No PEFT LoRA weights found in {peft_path}")

    mx.save_safetensors(str(mlx_path / "adapters.safetensors"), weights)
    config = {
        "fine_tune_type": "lora",
        "num_layers": max(layer_ids) + 1,
        "lora_parameters": {
            "rank": rank,
            "scale": scale,
            "dropout": float(peft_config.get("lora_dropout", 0.0)),
            "keys": sorted(module_keys),
        },
    }
    (mlx_path / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")

    for filename in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "processor_config.json",
        "README.md",
    ]:
        source = peft_path / filename
        if source.exists():
            shutil.copy2(source, mlx_path / filename)

    print(
        json.dumps(
            {
                "output": str(mlx_path),
                "weights": len(weights),
                "num_layers": config["num_layers"],
                "rank": rank,
                "scale": scale,
                "keys": sorted(module_keys),
                "target_modules": target_modules,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
