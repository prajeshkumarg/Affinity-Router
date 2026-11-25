# Tasks

A checklist derived from [dev-plan.md](dev-plan.md). Tick items off as they land. Task IDs (e.g. `T2.3`) can be used in commit messages and PR titles.

## Phase 0: Environment
- [ ] T0.1 Accept the Llama 3.1 license on Hugging Face and create a read token
- [ ] T0.2 Build a Singularity + conda overlay with pinned versions of vLLM, peft, trl, bitsandbytes, datasets, httpx, pandas and ruff
- [ ] T0.3 Download `meta-llama/Llama-3.1-8B-Instruct` to HPC scratch once and record the path in `configs/paths.yaml`
- [ ] T0.4 Smoke test on 1 GPU: serve the base model with `--enable-lora`, then send a chat request with and without a dummy adapter
- [ ] T0.5 Check vLLM's `/metrics` names for prefix-cache hits/queries and running/waiting requests, and write them down
- [ ] T0.6 Write a Slurm template (`slurm/template.sbatch`): account, partition, overlay, two vLLM servers pinned to GPU 0 and GPU 1
- [ ] T0.7 Two-replica smoke test on `c24m170-a100-2`: both servers reachable from a script on the node
- [ ] T0.8 Set up the repo skeleton: `finetune/ workload/ router/ bench/ analysis/ dashboard/ configs/ slurm/ tests/`, plus `pyproject.toml` and a ruff config

## Phase 1: Adapters
- [ ] T1.1 Write a finance data prep script that turns Financial PhraseBank, FiQA-SA and ConvFinQA into one chat format, with train/eval splits
- [ ] T1.2 Write a code data prep script that turns the code-instruction dataset into the same chat format
- [ ] T1.3 Write a QLoRA training script (rank 8, fixed steps, config-driven) that saves the adapter and its training metrics (GPU memory, time)
- [ ] T1.4 Train the finance adapter
- [ ] T1.5 Train the coding adapter
- [ ] T1.6 Write an eval script that serves the bf16 base model plus each adapter in vLLM, scoring finance accuracy/F1 and HumanEval pass@1
- [ ] T1.7 Make a score table: base model vs. each adapter on both tasks

## Phase 2: Workload
- [ ] T2.1 Build the finance session corpus: report contexts (~2–4k tokens) with multi-turn questions
- [ ] T2.2 Build the coding session corpus: source files (~1–3k tokens) with multi-turn questions or edits
- [ ] T2.3 Write a trace generator with seed, mix, turn count and output cap that writes JSONL
- [ ] T2.4 Generate traces for the 50:50, 70:30 and 90:10 mixes
- [ ] T2.5 Compute trace statistics (prompt length and shared-prefix length per turn) and document them

## Phase 3: Load generator and calibration
- [ ] T3.1 Write an async streaming client that records TTFT, TPOT, end-to-end latency, status and tokens for each request
- [ ] T3.2 Implement Poisson session arrivals with think time between turns, calibrated to a target request rate
- [ ] T3.3 Write a `/metrics` scraper and an `nvidia-smi` sampler that run alongside each run
- [ ] T3.4 Define the run directory format: config, requests, metrics, GPU samples, router events
- [ ] T3.5 Calibrate on one replica: unloaded TTFT/TPOT, then sweep the rate until saturation
- [ ] T3.6 Fix the SLO and ~5 rate levels in `configs/grid.yaml`

## Phase 4: Router
- [ ] T4.1 Build an OpenAI-compatible async proxy with streaming passthrough
- [ ] T4.2 Track replica state: in-flight counts and the session-to-replica map
- [ ] T4.3 Implement the static split policy
- [ ] T4.4 Implement the least-loaded policy
- [ ] T4.5 Implement the affinity + load hybrid policy (threshold θ)
- [ ] T4.6 Write unit tests with fake backends covering each policy, the threshold behavior and session-map updates
- [ ] T4.7 Write a routing-event log (JSONL) and a live event stream for the dashboard
- [ ] T4.8 Add a reset endpoint that clears router state between runs
- [ ] T4.9 Measure router overhead: TTFT through the router vs. direct
- [ ] T4.10 Fix θ after one calibration check and record it in `configs/grid.yaml`

## Phase 5: Grid runs
- [ ] T5.1 Write a grid driver that, for each run, resets the caches and router, warms up, runs, drains, then writes the run directory
- [ ] T5.2 Make the driver resumable, skipping completed runs
- [ ] T5.3 Write a Slurm batch job for the grid
- [ ] T5.4 Pilot: run the full grid once with a short duration, then sanity-check hit rates and goodput
- [ ] T5.5 Full grid: 45 configurations × 3 repeats
- [ ] T5.6 Validate results: no missing or broken runs

## Phase 6: Analysis
- [ ] T6.1 Write a loader that turns run directories into one DataFrame
- [ ] T6.2 Compute goodput over the steady-state window
- [ ] T6.3 Plot goodput vs. rate, one panel per mix and one line per policy
- [ ] T6.4 Plot the crossover: traffic mix where the hybrid overtakes the static split
- [ ] T6.5 Plot prefix-cache hit rate per policy and mix
- [ ] T6.6 Plot p95/p99 TTFT vs. rate
- [ ] T6.7 Show variance across repeats with error bars

## Phase 7: Dashboard
- [ ] T7.1 Live view of routing decisions per replica, colored by domain
- [ ] T7.2 Running hit rate and rolling goodput panels
- [ ] T7.3 Demo mode: replay a trace while switching policy or mix

## Phase 8: Report and demo
- [ ] T8.1 Write the report: question, setup, method, results, crossover, limitations
- [ ] T8.2 Record a backup demo video
- [ ] T8.3 Make presentation slides
