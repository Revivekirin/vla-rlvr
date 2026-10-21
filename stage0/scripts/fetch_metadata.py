"""Record immutable Hub revisions and file inventories; does not fetch tensors."""
import json
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
models = ["tanshh97/RIPT_VLA"] + [
    f"moojink/openvla-7b-oft-finetuned-libero-{s}" for s in ("goal", "10", "object", "spatial")
]
lock = {}
for model in models:
    # Exact snapshot embedded in the released Long adapter's base-model path.
    revision_path = '/revision/db188c5eda473a9b81a603759b3444364a887b98' if model.endswith('libero-10') else ''
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{model}{revision_path}?blobs=true", timeout=60) as response:
        data = json.load(response)
    (root / "evidence" / (model.replace("/", "__") + ".json")).write_text(json.dumps(data, indent=2) + "\n")
    lock[model] = {"revision": data["sha"], "files": data["siblings"]}
(root / "models.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
print(json.dumps({k: {"revision": v["revision"], "files": len(v["files"])} for k, v in lock.items()}, indent=2))
