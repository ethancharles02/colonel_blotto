#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_DIR="$REPO_ROOT/jobs"
RESULTS_DIR="$REPO_ROOT/results"
SBATCH_CMD="${SBATCH_CMD:-sbatch}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage: submit_jobs.sh [--overwrite]

Options:
  --overwrite   Submit all combinations even if the expected CSV already exists
  -h, --help    Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$JOBS_DIR"

SCANCEL_USER="${USER:-$(id -un)}"
scancel -u "$SCANCEL_USER" >/dev/null 2>&1 || true

find "$JOBS_DIR" -mindepth 1 -maxdepth 1 \( -type f -o -type l \) -delete
touch "$JOBS_DIR/.gitkeep"

"$PYTHON_BIN" - "$REPO_ROOT" "$JOBS_DIR" "$RESULTS_DIR" "$OVERWRITE" <<'PY'
import csv
import math
import shlex
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
jobs_dir = Path(sys.argv[2]).resolve()
results_dir = Path(sys.argv[3]).resolve()
overwrite = bool(int(sys.argv[4]))

nonretain_agents = ["random", "even", "mc", "mcts", "dp_nash", "dp_exploit"]
retain_agents = ["random", "even", "retaining_heuristic", "dp_nash", "dp_exploit"]
agent_weights = {
    "mc": 5,
    "mcts": 5,
    "dp_nash": 2,
}

canonical = {
    "sim_iters": "1000",
    "num_steps": "100",
    "n_att": "10",
    "n_def": "10",
    "m": "0.5",
    "p": "1.0",
    "alpha": "0.5",
    "c0": "0.25",
}

parameter_sets = [canonical]
parameter_sets.extend(
    {
        **canonical,
        "n_att": str(value),
    }
    for value in (6, 8, 12, 14)
)
parameter_sets.extend(
    {
        **canonical,
        "n_def": str(value),
    }
    for value in (6, 8, 12, 14)
)
parameter_sets.extend(
    {
        **canonical,
        "alpha": f"{value:.1f}",
    }
    for value in (0.1, 0.3, 0.7, 0.9)
)
parameter_sets.extend(
    {
        **canonical,
        "m": f"{value:.2f}".rstrip("0").rstrip(".") if value != 1.0 else "1.0",
    }
    for value in (0.25, 0.75, 1.0)
)
parameter_sets.extend(
    {
        **canonical,
        "c0": f"{value:.1f}",
    }
    for value in (0.0, 0.5)
)


def build_results_filename(row: dict[str, str]) -> str:
    retain_bool = "True" if row["retain"] == "1" else "False"
    return (
        f"att-{row['attacker']}"
        f"__def-{row['defender']}"
        f"__sims-{row['sim_iters']}"
        f"__steps-{row['num_steps']}"
        f"__natt-{row['n_att']}"
        f"__ndef-{row['n_def']}"
        f"__m-{row['m']}"
        f"__p-{row['p']}"
        f"__alpha-{row['alpha']}"
        f"__c0-{row['c0']}"
        f"__retain-{retain_bool}.csv"
    )


rows = []
sequence = 0
for retain, agents in ((False, nonretain_agents), (True, retain_agents)):
    for attacker in agents:
        for defender in agents:
            score = agent_weights.get(attacker, 1) + agent_weights.get(defender, 1)
            for params in parameter_sets:
                row = {
                    "attacker": attacker,
                    "defender": defender,
                    "retain": "1" if retain else "0",
                    "sim_iters": params["sim_iters"],
                    "num_steps": params["num_steps"],
                    "n_att": params["n_att"],
                    "n_def": params["n_def"],
                    "m": params["m"],
                    "p": params["p"],
                    "alpha": params["alpha"],
                    "c0": params["c0"],
                    "score": str(score),
                    "sequence": str(sequence),
                }
                row["csv_path"] = str(results_dir / build_results_filename(row))
                rows.append(row)
                sequence += 1

selected_rows = []
skipped_rows = []
for row in rows:
    if not overwrite and Path(row["csv_path"]).exists():
        skipped_rows.append(row)
        continue
    selected_rows.append(row)

selected_rows.sort(key=lambda row: (-int(row["score"]), int(row["sequence"])))

batch_count = math.ceil(len(selected_rows) / 5) if selected_rows else 0
batches = [[] for _ in range(batch_count)]
for index, row in enumerate(selected_rows):
    batches[index % batch_count].append(row)

header = [
    "attacker",
    "defender",
    "retain",
    "sim_iters",
    "num_steps",
    "n_att",
    "n_def",
    "m",
    "p",
    "alpha",
    "c0",
    "score",
    "csv_path",
]

for index, batch in enumerate(batches, start=1):
    batch_id = f"{index:04d}"
    manifest_path = jobs_dir / f"blotto_batch_{batch_id}.csv"
    job_path = jobs_dir / f"blotto_batch_{batch_id}.sh"
    log_path = jobs_dir / f"blotto_b{batch_id}.out"
    job_name = f"blotto_b{batch_id}"

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows({key: row[key] for key in header} for row in batch)

    job_script = f"""#!/bin/bash --login
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH -J {shlex.quote(job_name)}
#SBATCH --output={shlex.quote(str(log_path))}

set -euo pipefail

REPO_ROOT={shlex.quote(str(repo_root))}
MANIFEST_PATH={shlex.quote(str(manifest_path))}

cd "$REPO_ROOT"

mamba activate baseball_env

failures=0

while IFS=',' read -r attacker defender retain sim_iters num_steps n_att n_def m p alpha c0 score csv_path; do
  if [[ "$attacker" == "attacker" ]]; then
    continue
  fi

  cmd=(
    python main.py
    --attacker "$attacker"
    --defender "$defender"
    --sim-iters "$sim_iters"
    --num-steps "$num_steps"
    --n-att "$n_att"
    --n-def "$n_def"
    --m "$m"
    --p "$p"
    --alpha "$alpha"
    --c0 "$c0"
    --seed 42
    --results-dir results
    --no-show-plots
  )

  if [[ "$retain" == "1" ]]; then
    cmd+=(--retain)
  fi

  label="attacker=$attacker defender=$defender retain=$retain n_att=$n_att n_def=$n_def alpha=$alpha m=$m p=$p c0=$c0 score=$score"
  echo "START $label"

  if "${{cmd[@]}}"; then
    echo "SUCCESS $label"
  else
    status=$?
    failures=$((failures + 1))
    echo "FAIL status=$status $label" >&2
  fi
done < "$MANIFEST_PATH"

if [[ "$failures" -gt 0 ]]; then
  echo "Batch completed with $failures failed combinations." >&2
  exit 1
fi

echo "Batch completed successfully."
"""

    job_path.write_text(job_script, encoding="utf-8")
    job_path.chmod(0o755)

print(
    "Prepared "
    f"{len(selected_rows)} combinations across {len(batches)} batches; "
    f"skipped {len(skipped_rows)} existing results; "
    f"overwrite={'on' if overwrite else 'off'}."
)
PY

mapfile -t job_files < <(find "$JOBS_DIR" -maxdepth 1 -type f -name 'blotto_batch_*.sh' | sort)
if [[ "${#job_files[@]}" -eq 0 ]]; then
  echo "No pending jobs to submit." >&2
  exit 0
fi

read -r -a sbatch_parts <<< "$SBATCH_CMD"

submit_count=0
for job_file in "${job_files[@]}"; do
  "${sbatch_parts[@]}" "$job_file"
  submit_count=$((submit_count + 1))
done

echo "Submitted $submit_count batch jobs from $JOBS_DIR using: $SBATCH_CMD"
