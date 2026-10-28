import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/nfs4/jhkim/repos/vla-rlvr")
OUT = ROOT / "outputs/rlvr_analysis/p3"
BASE = ROOT / "outputs/libero-plus"
EVAL = ROOT / "experiments/libero_plus/evaluate.py"

FIELDS = [
    "repeat", "model", "task_id", "task_name", "shift",
    "historical_case", "seed", "success", "steps",
    "historical_success", "historical_steps",
    "host", "evaluator_sha256"
]

def read_result(model, task_id):
    p = BASE / model / "seed-1" / f"task-{task_id:04d}.json"
    return json.loads(p.read_text())

def select_cases():
    selected = {"lost": [], "gained": [],
                "both_success": [], "both_fail": []}

    for task_id in range(1, 2592):
        s = read_result("scale", task_id)
        r = read_result("ript", task_id)

        assert s["shift"] == r["shift"]
        assert s["task_name"] == r["task_name"]

        if s["shift"] != "Camera Viewpoints":
            continue

        a, b = s["success"], r["success"]

        kind = (
            "lost" if a and not b else
            "gained" if not a and b else
            "both_success" if a and b else
            "both_fail"
        )

        selected[kind].append(task_id)

    assert 698 in selected["lost"]
    assert 705 in selected["gained"]

    lost = [698] + [x for x in selected["lost"] if x != 698][:2]
    gained = [705] + [x for x in selected["gained"] if x != 705][:2]

    ids = (
        lost + gained
        + selected["both_success"][:1]
        + selected["both_fail"][:1]
    )

    assert len(ids) == len(set(ids)) == 8
    assert len(lost) == len(gained) == 3

    return ids, {
        tid: kind
        for kind, values in selected.items()
        for tid in values
        if tid in ids
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()

    assert 1 <= args.repeats <= 20

    OUT.mkdir(parents=True, exist_ok=True)
    result = OUT / "p3_a2_rollouts.csv"
    selection = OUT / "p3_a2_selection.json"

    if result.exists() or selection.exists():
        raise RuntimeError("P3-A2 outputs already exist; refusing overwrite")

    task_ids, labels = select_cases()

    selection.write_text(json.dumps({
        "seed": 1,
        "repeats": args.repeats,
        "task_ids": task_ids,
        "historical_cases": labels,
        "protocol": "Original LIBERO-PLUS Goal evaluator, task filter only"
    }, indent=2))

    print("Selected task IDs:", task_ids, flush=True)
    print("Historical cases:", labels, flush=True)
    print("Planned rollouts:", len(task_ids) * 2 * args.repeats, flush=True)

    if args.prepare_only:
        return

    host = subprocess.check_output(
        ["hostname"], text=True
    ).strip()

    digest = hashlib.sha256(EVAL.read_bytes()).hexdigest()

    with result.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        f.flush()

        for repeat in range(1, args.repeats + 1):
            for model in ("scale", "ript"):
                name = f"repeat_{repeat:02d}_{model}"
                run_dir = OUT / "runs" / name
                log = OUT / f"{name}.log"

                cmd = [
                    "bash", str(ROOT / "experiments/server_env.sh"),
                    "plus", "python", "-u", str(EVAL),
                    "--model", model,
                    "--seed", "1",
                    "--task-ids", *map(str, task_ids),
                    "--output", str(run_dir),
                    "--execute",
                ]

                print("RUN", name, flush=True)

                with log.open("x") as stream:
                    proc = subprocess.run(
                        cmd, cwd=ROOT,
                        stdout=stream, stderr=subprocess.STDOUT,
                        check=False
                    )

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"{name} failed ({proc.returncode}); see {log}"
                    )

                folder = run_dir / model / "seed-1"
                files = sorted(folder.glob("task-*.json"))

                if len(files) != len(task_ids):
                    raise RuntimeError(
                        f"{name}: expected {len(task_ids)} results, got {len(files)}"
                    )

                actual_ids = set()

                for path in files:
                    row = json.loads(path.read_text())
                    tid = int(row["task_id"])

                    assert tid in task_ids and tid not in actual_ids
                    assert row["model"] == model
                    assert row["seed"] == 1
                    assert row["shift"] == "Camera Viewpoints"
                    assert isinstance(row["success"], bool)
                    actual_ids.add(tid)

                    old = read_result(model, tid)

                    writer.writerow({
                        "repeat": repeat,
                        "model": model,
                        "task_id": tid,
                        "task_name": row["task_name"],
                        "shift": row["shift"],
                        "historical_case": labels[tid],
                        "seed": 1,
                        "success": int(row["success"]),
                        "steps": row["steps"],
                        "historical_success": int(old["success"]),
                        "historical_steps": old["steps"],
                        "host": host,
                        "evaluator_sha256": digest,
                    })

                assert actual_ids == set(task_ids)
                f.flush()
                print("COMPLETED", name, flush=True)

    with result.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(task_ids) * 2 * args.repeats
    print("P3-A2 COMPLETE:", result, flush=True)
    print("TOTAL ROLLOUTS:", len(rows), flush=True)

    for tid in task_ids:
        print("TASK", tid, labels[tid])
        for model in ("scale", "ript"):
            subset = [
                r for r in rows
                if int(r["task_id"]) == tid and r["model"] == model
            ]
            successes = sum(int(r["success"]) for r in subset)
            print(" ", model, f"{successes}/{len(subset)}")

if __name__ == "__main__":
    main()
