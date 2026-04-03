#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_DIR="$REPO_ROOT/jobs"
SBATCH_CMD="${SBATCH_CMD:-sbatch}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$JOBS_DIR"
find "$JOBS_DIR" -maxdepth 1 -type f \( -name 'blotto_batch_*.csv' -o -name 'blotto_batch_*.sh' \) -delete

"$PYTHON_BIN" - "$REPO_ROOT" "$JOBS_DIR" <<'PY'
import csv
import shlex
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
jobs_dir = Path(sys.argv[2]).resolve()

nonretain_agents = ["random", "even", "mc", "mcts", "dp_nash", "dp_exploit"]
retain_agents = ["random", "even", "retaining_heuristic", "dp_nash", "dp_exploit"]
heavy_agents = {"mc", "mcts"}

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

rows = []
for retain, agents in ((False, nonretain_agents), (True, retain_agents)):
    for attacker in agents:
        for defender in agents:
            is_heavy = attacker in heavy_agents or defender in heavy_agents
            for params in parameter_sets:
                rows.append(
                    {
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
                        "is_heavy": "1" if is_heavy else "0",
                    }
                )

heavy_rows = [row for row in rows if row["is_heavy"] == "1"]
light_rows = [row for row in rows if row["is_heavy"] == "0"]

batches = []
heavy_index = 0
light_index = 0

while heavy_index < len(heavy_rows):
    batch = []
    for _ in range(2):
        if heavy_index < len(heavy_rows):
            batch.append(heavy_rows[heavy_index])
            heavy_index += 1
    while len(batch) < 5 and light_index < len(light_rows):
        batch.append(light_rows[light_index])
        light_index += 1
    batches.append(batch)

while light_index < len(light_rows):
    batch = light_rows[light_index:light_index + 5]
    light_index += len(batch)
    batches.append(batch)

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
    "is_heavy",
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
        writer.writerows(batch)

    job_script = f"""#!/bin/bash --login
#SBATCH --time=1:00:00
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

while IFS=',' read -r attacker defender retain sim_iters num_steps n_att n_def m p alpha c0 is_heavy; do
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

  label="attacker=$attacker defender=$defender retain=$retain n_att=$n_att n_def=$n_def alpha=$alpha m=$m p=$p c0=$c0"
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

heavy_total = len(heavy_rows)
light_total = len(light_rows)
print(
    "Prepared "
    f"{len(rows)} combinations across {len(batches)} batches "
    f"({heavy_total} heavy, {light_total} light)."
)
PY

mapfile -t job_files < <(find "$JOBS_DIR" -maxdepth 1 -type f -name 'blotto_batch_*.sh' | sort)
if [[ "${#job_files[@]}" -eq 0 ]]; then
  echo "No job files were generated." >&2
  exit 1
fi

read -r -a sbatch_parts <<< "$SBATCH_CMD"

submit_count=0
for job_file in "${job_files[@]}"; do
  "${sbatch_parts[@]}" "$job_file"
  submit_count=$((submit_count + 1))
done

echo "Submitted $submit_count batch jobs from $JOBS_DIR using: $SBATCH_CMD"
