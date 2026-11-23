# CLAUDE.md: Affinity Router

Context for Claude (and teammates) working in this repo.

## What this project is

A two-person term project (Prajesh + Shashaank) for NYU *Cloud and Machine Learning*, Fall 2026 (Prof. Lee).
It answers **one question**: on 2× A100, does a shared pool with cache-affinity routing beat a static per-domain split for serving finance and coding LoRA adapters, and at what traffic mix does the winner flip?

Read `docs/proposal.md` before making design decisions. Its "Fixed" list is binding.

## Scope rules (from instructor feedback, `.claude/docs/proposal_v1_feedback.png`)

The first proposal swept too many things. **Do not add new sweep dimensions.** Hold these fixed:

- one base model, bf16: `meta-llama/Llama-3.1-8B-Instruct` (no 4-bit or precision study). It is gated on Hugging Face: each teammate must accept Meta's license and set `HF_TOKEN` on the cluster. Fallback if access is delayed: `Qwen/Qwen3-8B` with thinking mode off. Pick one before any adapter training, because adapters are tied to the base model.
- one layout: two replicas, one per GPU (no tensor-parallel split, no four-replica layout)
- vLLM defaults for batching, chunked prefill (on), prefix caching (on), KV-cache size
- Poisson arrivals only; one session workload
- one SLO

The only swept factors are **routing policy (3) × traffic mix (3) × request rate (~5)**.
If something seems worth exploring beyond that, raise it with the team instead of building it.

## Compute: NYU HPC Cloud Bursting

- Never run model training or inference locally (no local CPU or GPU runs). Write code locally and run it on the cluster.
- Access is only through Open OnDemand (NYU VPN off campus). Jobs are submitted from Jupyter terminals on OOD.
- Slurm account (required on every job): `csci_ga_3033_085-2026fa`
- Partitions: `interactive`, `n2c48m24` (CPU), `g2-standard-12` (1× L4), `g2-standard-24` (2× L4), `c12m85-a100-1` (1× A100 40GB), `c24m170-a100-2` (2× A100 40GB, **main experiment partition**)
- Budget: 300 GPU-hours shared per student allocation. Use L4 partitions for debugging and A100 for real runs.
- Instances with low GPU utilization for 20 minutes are shut down automatically. Don't leave idle interactive sessions open, and make benchmark scripts checkpoint results per run.
- Environments use Singularity + conda overlays (templates in `/share/apps/overlay-fs-ext3`, images in `/share/apps/images`).

## Working conventions

- Every benchmark run writes one self-describing result file (config + per-request timings) under `results/`, keyed by `policy_mix_rate_repeat`. Raw results are not committed.
- Configs live in files (YAML), not hard-coded constants, so the full grid can be regenerated from one config.
- The router exposes an OpenAI-compatible API so the load generator can talk to the router or directly to a single vLLM replica.
- Prefer small, testable modules. Router policy logic should be unit-testable without a GPU.
- Don't commit model weights, datasets, adapters, or secrets (`.env`, HF tokens).
- Ask before doing browser actions (e.g. on OOD). Prajesh prefers to do those steps himself.
