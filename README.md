# Cost-Aware KV-Cache Preemption in vLLM
**A scheduler-level optimization for vLLM V1 that makes KV-cache preemption decisions based on **reclaimable memory and recomputation cost**, rather than treating eligible running requests uniformly.**

# 1. Why This Matters

LLM serving is heavily constrained by GPU memory because every active request maintains a **KV cache** containing the attention keys and values generated throughout its context.

As requests grow and more sequences are served concurrently, the KV cache can become saturated. When there are not enough free KV-cache blocks to continue scheduling work, the scheduler must **preempt a running request** and reclaim its cache.

The important question is therefore not just **"Which request can I evict?"** but **"Which request should I evict to recover memory while minimizing the work that will have to be redone?"**

That is the problem addressed by this contribution.

# 2. Where This Fits in vLLM

At a high level, vLLM's V1 serving path looks roughly like:

```text
                    Incoming Requests
                           │
                           ▼
                  ┌─────────────────┐
                  │     Engine      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │    Scheduler    │
                  │                 │
                  │ • batching      │
                  │ • token budget  │
                  │ • KV capacity   │
                  │ • preemption    │
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       KV Cache Manager          Model Executor
              │                         │
              ▼                         ▼
       Physical KV Blocks          GPU kernels
              │
              ▼
          GPU HBM
```

The **scheduler** decides which requests can execute in each scheduling iteration.

The **KV-cache manager** tracks the physical KV blocks associated with each request.

When KV capacity becomes insufficient, the scheduler enters the preemption path.

### Existing flow

```text
KV capacity exhausted
        │
        ▼
Identify eligible running requests
        │
        ▼
Existing victim-selection policy
        │
        ▼
Preempt request
        │
        ▼
Release KV blocks
        │
        ▼
Recompute the request when resumed
```

Our change is deliberately located at the **victim-selection step**.

We do **not** change when preemption happens or which requests are eligible.

# 3. The Problem With Treating Victims Equally

Two running requests can have very different costs of preemption.

For example:

```text
Request A
─────────
Large KV footprint
Few tokens generated
Low reconstruction cost

Request B
─────────
Smaller reclaimable KV footprint
Long generation already completed
High reconstruction cost
```

Evicting the request with the larger cache footprint may recover more memory, but it can also destroy more useful computation.

This creates a tradeoff:

```text
              Preemption decision
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Memory recovered       Work destroyed
   by eviction            by recomputation
```

The goal is to make that tradeoff explicit.

# 4. Our Approach: Cost-Aware Victim Selection

For every eligible running request, we compute a **victim desirability score** based on:

### ① Reclaimable KV blocks

We count blocks whose reference count indicates that they can actually be released when the request is preempted.

The reason this counts is:

```Total KV blocks held ≠ KV blocks that can actually be reclaimed.```

Shared/prefix-cache blocks may remain referenced by other requests.

### ② Recompute cost

After preemption, the request must reconstruct work that was previously computed.

We therefore use the request's token state as a proxy for the amount of work that will have to be regenerated.

### ③ Progress protection

A request that is close to completing generation is less attractive to preempt.

The intuition is straightforward:

```text
Early request ────────────────► More work remains
                              ↑
                         easier victim

Nearly finished ─────────────► Little work remains
                              ↑
                         protect it
```

The resulting score favors requests that provide useful KV reclamation relative to their reconstruction cost, while reducing the attractiveness of requests that are already far through generation.

# 5. Scheduler Integration

One design goal was to keep the change **localized**.

The existing scheduler continues to determine:

* when KV pressure requires preemption;
* which requests are eligible;
* how priority scheduling works.

The contribution changes only:

```text
              Eligible running requests
                         │
                         ▼
                 ┌───────────────┐
                 │ Existing      │
                 │ victim choice │
                 └───────┬───────┘
                         │
                         ▼
                  COST-AWARE
                  VICTIM SCORE
                         │
                         ▼
                    Selected
                     victim
```

This keeps the behavioral surface area of the change small.

The priority-preemption path is also left unchanged.

# 6. Implementation

The core scheduler change is in:

```text
vllm/v1/core/sched/scheduler.py
```

The implementation:

1. Retrieves the request's KV-cache blocks through the current KV-cache manager interface.
2. Counts blocks with `ref_cnt == 1` as reclaimable.
3. Estimates recomputation cost from the request's token state.
4. Applies a progress-protection factor.
5. Scores every eligible running request.
6. Selects the highest-scoring victim.

The corresponding scheduler test was updated to use the current:

```text
get_blocks(request_id).blocks
```

interface rather than the older internal `req_to_blocks` representation.

# 7. Correctness Validation

We specifically tested the paths affected by the change:

| Test area                          | Purpose                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------- |
| Cost-aware victim selection        | Verifies the scheduler chooses the intended victim                      |
| Preemption during execution        | Verifies victim selection works during actual scheduler preemption      |
| KV-connector preemption/resumption | Verifies preemption remains compatible with KV-cache connector behavior |

Targeted validation:

```text
10 passed, 157 deselected
```

The full scheduler test file contains many additional tests; the targeted run isolates the tests relevant to this contribution.

# 8. Evaluation Methodology

We evaluated the change against stock vLLM under controlled KV-cache pressure.

The serving configuration intentionally constrains available KV capacity:

```text
Model:              Llama 3.1 8B Instruct
GPU:                A100
KV blocks:          2,000
KV capacity:        32,000 tokens
Max concurrency:    128
Chunked prefill:    enabled
Request rate:       40 req/s
Requests:           500
```

The objective was not simply to maximize throughput.

We measured:

* request throughput;
* output throughput;
* mean TTFT;
* P99 TTFT;
* mean/P99 TPOT;
* mean/P99 ITL;
* preemption frequency.

We also ran repeated baseline and cost-aware experiments rather than relying on a single noisy serving run.


# 9. Stock vs. Cost-Aware

| Metric             |     Stock vLLM |     Cost-Aware |     Δ |
| ------------------ | -------------: | -------------: | ----: |
| Preemptions        |           73.0 |           70.0 | −4.1% |
| Request throughput |     6.64 req/s |     6.65 req/s | +0.1% |
| Output throughput  | 1,418.04 tok/s | 1,411.73 tok/s | −0.4% |
| Mean TTFT          |    3,302.11 ms |    3,239.06 ms | −1.9% |
| P99 TTFT           |   12,152.92 ms |   12,329.85 ms | +1.5% |
| Mean TPOT          |       67.88 ms |       68.92 ms | +1.5% |
| P99 ITL            |      246.44 ms |      226.70 ms | −8.0% |

**Interpretation:**

The cost-aware policy produced **comparable serving performance** to the stock scheduler under the tested KV-constrained workload, while reducing the number of preemptions. The workload did not produce a large enough separation in victim costs to claim a large end-to-end throughput improvement.


# 10. Impact: Making KV-Cache Preemption Cost-Aware

The key impact of this work is that **vLLM's scheduler no longer treats all eligible preemption victims as equivalent**.

Under KV-cache pressure, the scheduler must evict a running request to free GPU memory. But evicting a request has two consequences:

1. **KV memory is recovered**, allowing another request to run.
2. **Previously computed state is lost**, creating recomputation work when the evicted request resumes.

Our policy explicitly considers both sides of this tradeoff.

```text
                 KV cache becomes full
                         │
                         ▼
              Running requests become
              candidates for preemption
                         │
                         ▼
              ┌──────────────────────┐
              │ Cost-aware scoring   │
              │                      │
              │ KV blocks recovered  │
              │        vs.            │
              │ recomputation cost   │
              │        +              │
              │ generation progress  │
              └──────────┬───────────┘
                         │
                         ▼
                Select lower-cost
                victim candidate
                         │
                         ▼
                 Reclaim KV memory
                         │
                         ▼
                Continue scheduling
```

### What changed

Stock vLLM's victim-selection mechanism does not explicitly model the **economic cost of evicting each request**.

This work adds that cost awareness while leaving the rest of the scheduler's preemption mechanism intact:

* the existing KV-capacity trigger remains unchanged;
* the existing eligibility rules remain unchanged;
* priority scheduling remains unchanged;
* only the choice of **which eligible request to preempt** is changed.

### What the evaluation shows

We evaluated stock and cost-aware vLLM under the same KV-constrained serving workload using **Llama 3.1 8B Instruct**, the **ShareGPT** dataset, high request concurrency, and an intentionally restricted KV-cache capacity to induce preemption.

Across repeated runs, the cost-aware scheduler maintained **comparable throughput and latency** to stock vLLM while reducing observed preemptions.

In particular:

* **Preemptions:** 73.0 → 70.0
* **Request throughput:** 6.64 → 6.65 req/s
* **Output throughput:** 1,418.04 → 1,411.73 tok/s
* **Mean TTFT:** 3,302 → 3,239 ms
* **P99 ITL:** 246 → 227 ms

The results therefore demonstrate that introducing cost-aware victim selection **does not require sacrificing overall serving performance** in the evaluated workload, while giving the scheduler a more informed basis for deciding which request to evict.

### Outcome

KV-cache preemption is fundamentally a tradeoff between **recovering scarce GPU memory** and **discarding computation that may need to be regenerated**.

This work moves that decision from:

**"Which request should be evicted?"**

toward:

**"Which eviction gives us useful KV capacity at the lowest reconstruction cost?"**

That makes preemption a **cost-aware scheduling decision**, providing a foundation for future policies that can incorporate additional signals such as KV sharing, request latency objectives, generation progress, or workload characteristics.
