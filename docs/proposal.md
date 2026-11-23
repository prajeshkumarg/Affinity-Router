# Term Project Proposal (v2): Split or Share? Cache-Aware Routing of Finance and Coding LoRA Adapters on Two GPUs

Cloud and Machine Learning, Fall 2026. Prajesh Kumar Ganesh Kumar, Shashaank R.

*Revised after instructor feedback on v1 (`.claude/docs/proposal_v1_submitted.pdf`): we narrowed the project to one question and hold every other design choice fixed.*

## Question

We serve one 8B base model with two LoRA adapters, one for finance and one for coding, on two A100 GPUs. Two strategies compete:

- **Static split:** one replica per domain. Each replica's prefix cache holds only its own domain's sessions, but one GPU sits underused when traffic is skewed.
- **Shared pool:** both replicas load both adapters, and a router places each request. This uses both GPUs fully, but each replica's cache is shared between domains.

**Which strategy gives higher goodput (requests per second meeting a latency SLO), and at what finance:coding traffic mix does the answer flip?**

## Design

**Swept factors (3 × 3 × ~5 = 45 configurations, 3 repeats each):**

| Factor | Levels |
|---|---|
| Routing policy | (1) static split; (2) shared pool, least-loaded; (3) shared pool, affinity + load hybrid (send a session to the replica holding its prefix unless that replica's queue exceeds the other's by a threshold) |
| Finance:coding mix | 50:50, 70:30, 90:10 |
| Request rate | ~5 levels from light load to saturation, calibrated on a single replica |

**Held fixed:**

- Base model: `meta-llama/Llama-3.1-8B-Instruct` in bf16 (fallback if gated access is delayed: `Qwen/Qwen3-8B`, thinking mode off)
- 2× A100 40GB node (NYU HPC Cloud Bursting), one vLLM replica per GPU
- vLLM defaults: chunked prefill on, prefix caching on, default batch limit and KV-cache size
- Workload: multi-turn finance sessions over a shared report and multi-turn coding sessions over a shared source file, with Poisson arrivals and the sessions' natural context lengths
- SLO: fixed TTFT and TPOT targets, set from unloaded single-replica latency

## Method

1. **Adapters.** Train one QLoRA adapter per domain (finance: Financial PhraseBank, FiQA-SA, ConvFinQA; coding: a code-instruction dataset, evaluated on HumanEval). Report task score against the base model as a sanity check only.
2. **Router.** An OpenAI-compatible proxy that tracks each replica's queue depth and session-to-replica affinity, and implements the three policies.
3. **Calibration.** Measure unloaded latency and the saturation rate on one replica to set the SLO and the rate levels.
4. **Grid.** Run the 45-configuration grid with 3 repeats.
5. **Measure.** Goodput under the SLO, prefix-cache hit rate, p50/p95/p99 TTFT, TPOT, and GPU utilization.

## Expected results

- At 50:50, the static split and the affinity router are close. The static split may win slightly on cache hit rate.
- As traffic skews toward finance, the static split saturates its finance replica while the coding replica idles, and the shared policies pull ahead on goodput.
- Least-loaded routing balances load but scatters sessions across replicas and loses prefix hits. The hybrid keeps most of the hit rate while balancing load.
- **Headline result:** the traffic mix and request rate at which the affinity hybrid overtakes the static split, and the hit rate the shared pool gives up to get there.

## Deliverables

A crossover plot of goodput vs. traffic mix per policy, a table of hit rates and tail latencies, and a live dashboard showing routing decisions and goodput as the traffic mix changes.

## Course relevance

- **Cloud:** multi-replica serving, load balancing, capacity planning on a fixed GPU budget.
- **DNNs:** QLoRA fine-tuning, multi-adapter LLM inference, KV-cache reuse.
- **Performance analysis:** goodput and tail latency under an SLO, crossover analysis.

## Related work

vLLM / PagedAttention (SOSP 2023), S-LoRA (multi-adapter serving), DistServe (OSDI 2024), Mooncake (KV-cache-aware scheduling), QLoRA.
