# Affinity Router

**Cache-aware request routing for multi-LoRA LLM serving on two GPUs.**

## The question

We serve one 8B base model with two LoRA adapters (finance and coding) on a fixed budget of two A100 GPUs. Should each domain get its own replica (**static split**), or should both replicas serve both domains from a **shared pool** with a router that sends requests where their prefix is already cached?

- A **static split** keeps each replica's prefix cache "pure" but leaves capacity idle when traffic is skewed.
- A **shared pool** uses both GPUs fully but splits each replica's cache between the two domains.

We measure the finance:coding traffic mix at which the shared affinity router overtakes the static split, and how much cache hit rate it gives up to get there.

## Experiment design

**Swept (45 configurations, 3 repeats each):**

| Factor | Levels |
|---|---|
| Routing policy | static split · shared least-loaded · shared affinity + load hybrid |
| Finance:coding mix | 50:50 · 70:30 · 90:10 |
| Request rate | ~5 levels, from light load to saturation |

**Fixed:**

- Base model: Llama 3.1 8B Instruct, bf16 weights
- 2× A100 40GB (NYU HPC Cloud Bursting, `c24m170-a100-2`), one vLLM replica per GPU
- vLLM defaults: chunked prefill on, prefix caching on, default batch limit and KV-cache size
- Multi-turn session workload (finance sessions over a shared report, coding sessions over a shared source file), Poisson arrivals
- One latency SLO on time to first token (TTFT) and time per output token (TPOT)

**Metrics:** goodput under the SLO (the headline), prefix-cache hit rate, p50/p95/p99 TTFT, TPOT, GPU utilization.

## Components

1. **Adapters:** QLoRA (rank 8) finance and coding adapters. We check each adapter's task score against the base model and do no sweeps.
2. **Workload generator:** builds multi-turn finance and coding sessions and replays them at a given rate and mix.
3. **Router:** an OpenAI-compatible proxy in front of the vLLM replicas. It tracks queue depth and session-to-replica affinity, and implements the three policies.
4. **Benchmark harness:** Slurm jobs that run the grid and collect per-request timings and vLLM metrics.
5. **Analysis and dashboard:** crossover plots, plus a live view of routing decisions and goodput.

## Repository layout (planned)

```
finetune/     QLoRA training + eval scripts
workload/     session generation and replay
router/       routing proxy and policies
bench/        load generator, metric collection, Slurm job scripts
analysis/     notebooks and plotting
dashboard/    live demo
docs/         proposal, feedback, report
results/      raw run outputs (git-ignored)
```

## Docs

- [Proposal](docs/proposal.md)
- [Architecture](docs/architecture.md)
- [Development plan](docs/dev-plan.md)
- [Tasks](docs/tasks.md)

## Compute

All runs use NYU HPC Cloud Bursting through Open OnDemand. See `CLAUDE.md` for the Slurm account and partitions.

## Related work

vLLM / PagedAttention (SOSP '23), S-LoRA, DistServe (OSDI '24), Mooncake, QLoRA.
