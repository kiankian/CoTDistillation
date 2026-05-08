# Chain-of-Thought Distillation: SmolLM2-360M ← Claude Haiku 4.5

> Distilling chain-of-thought reasoning from Claude Haiku 4.5 onto a 360M-parameter open-weight model, with a focus on understanding what distillation transfers — and what it doesn't — at small scale.

[**Model on HuggingFace**](https://huggingface.co/kianshandi/smollm2-360m-gsm8k-distilled-haiku) · **Trained on:** 1 × Kaggle T4 (free tier) · **API cost:** ~$3

---

## TL;DR

Trained `SmolLM2-360M-Instruct` on 1,435 chain-of-thought solutions distilled from Claude Haiku 4.5 on GSM8K math problems. After training:

| | Accuracy on 200 held-out GSM8K | 95% CI |
|---|---|---|
| **Baseline** | 10.5% | [6.5%, 14.5%] |
| **Fine-tuned** | **12.5%** | [8.0%, 17.5%] |
| **Paired Δ** | +2.0pp (1.19×) | [-3.0pp, 7.0pp] |

The accuracy gain is modest and **not statistically significant** at this sample size (paired bootstrap p ≈ 0.241). However, the qualitative analysis tells a different and more interesting story: the model cleanly adopted Claude's reasoning structure — bolded step headers, explicit arithmetic, `####` answer markers, structured progressions — even when its underlying arithmetic capacity remained the bottleneck.

**The headline finding isn't the +2pp number. It's that distillation at small scale is best understood as *style transfer*, not *capability transfer*.**

---

## Why this project

I'd shipped LLM-API-based projects before — wrappers around Claude and Gemini for various downstream tasks — but I'd never trained a model end-to-end. I wanted to fill that gap with something concrete: real fine-tuning, real eval rigor, and a real artifact.

I chose CoT distillation specifically because it teaches a small student to imitate a frontier teacher's reasoning style, which is a current research direction (e.g., DeepSeek-Math, Phi). I chose SmolLM2-360M as the base because it fits cleanly on a free Kaggle T4 and is small enough to expose the *capability ceiling* of distillation as a method — which turned out to be the most interesting finding.

## Pipeline

1. **Generate** 1,500 CoT solutions to GSM8K training problems with Claude Haiku 4.5 (~$2.70 in API costs)
2. **Filter** via rejection sampling — keep only CoTs whose final answer matches gold (95.7% acceptance, 1,435 kept)
3. **SFT-LoRA fine-tune** SmolLM2-360M on (question, CoT) pairs (LoRA rank 16 on attention projections, 3 epochs, ~20 min on T4)
4. **Evaluate** baseline and fine-tuned models on 200 held-out problems from GSM8K's test split with greedy decoding
5. **Analyze** with paired bootstrap CIs and qualitative side-by-side comparisons

## Training curve

![Loss curve](loss_curve_polished.png)

Train loss decreased monotonically across 3 epochs (0.798 → 0.755 → 0.740) with eval loss tracking closely (0.774 → 0.750 → 0.747), indicating successful learning without overfitting. The plateau in epochs 2–3 is consistent with the model approaching its representational capacity ceiling at 360M scale — additional training would yield diminishing returns.

## What the model actually learned

Even though headline accuracy moved only +2pp, the *qualitative* output transformation is dramatic:

**Baseline (untrained SmolLM2-360M) on a typing-speed problem:**
```
The current measurement is 47 WPM.
The next measurement is 52 WPM.
The third measureme... [rambles, no clear final answer]
```

**Fine-tuned model on the same problem:**
```
I need to find the average of the three measurements.

**Initial measurement:** 47 WPM
**Next measurement:** 52 WPM
**Third measurement:** 52 WPM + 5 WPM = 57 WPM
**Average:** (47 + 52 + 57) / 3 = 150 / 3 = 50 WPM

#### 50
```

The fine-tuned model produces correct **structure** — bolded step headers, explicit calculations, an answer marker — but commits an arithmetic/reading error (it misinterprets "+5 from 52" as a third value of 57 rather than a re-read of 52). It's wrong by 2, in a structurally clean way.

This pattern repeated across the held-out set:

| Metric | Fine-tuned model |
|---|---|
| Outputs ending with `####` marker | 190 / 200 (95%) |
| Median generation length | 416 chars |
| Outputs with bolded step structure | majority |

Distillation transferred the **format** of reasoning reliably. It did not — and at this scale could not — transfer the underlying arithmetic capacity.

## Failure mode analysis

Inspecting the 175 wrong predictions revealed three distinct failure modes:

1. **Off-by-N arithmetic errors.** The model picks a correct framework, walks the steps cleanly, but commits an arithmetic mistake mid-chain. (See typing-speed example above.)

2. **Reading comprehension errors.** The model misinterprets the relationships between entities in the problem (e.g., conflating who-does-what or which value depends on which).

3. **Repetition collapse.** ~5% of outputs entered degenerate loops mid-generation:
   ```
   ...5 footballs
   **Half as many footballs as footballs kept there:**
   5 footballs
   **Half as many footballs as footballs kept there:**
   5 footballs
   ...
   ```
   These exhausted the 512-token budget without producing a final answer. Greedy decoding has no way to escape such loops; sampling with `temperature=0.3` was tested and gave essentially identical accuracy (12.0%), confirming the bottleneck is base-model capacity, not decoding strategy.

## Statistical rigor

Used paired bootstrap (n=2000) on per-problem differences rather than independent CIs, because the same 200 problems were graded under both models. This controls for the fact that some problems are intrinsically harder than others.

```
Baseline:    10.5%  [95% CI: 6.5%, 14.5%]
Fine-tuned:  12.5%  [95% CI: 8.0%, 17.5%]
Paired Δ:    +2.0pp [95% CI: -3.0pp, 7.0pp]
P(fine-tuned ≤ baseline) ≈ 0.241
```

The paired CI crosses zero, so the improvement is **not statistically distinguishable from noise at this sample size**. A larger held-out set (1000+ problems) would be needed to nail down the sign of the effect with confidence.

I chose to report this honestly rather than reach for a more flattering number. The methodology is the artifact, not the outcome.

## Methodology notes worth flagging

A few decisions worth scrutinizing:

- **Teacher contamination.** Claude Haiku 4.5 has very likely seen GSM8K during training, so its CoTs may not be independent of the eval set. The student model never sees the test split directly, but the teacher's "knowledge" of GSM8K is implicitly in the training signal. This caveat applies to most CoT distillation work and is not unique to this project, but it should temper conclusions.
- **Rejection sampling biases the dataset.** Filtering to only-correct CoTs means the student trains exclusively on successful reasoning patterns. This is the standard recipe but it removes the model's exposure to "this looks like reasoning but produces wrong answers" — which is arguably part of what the model needs to internalize to *avoid* hallucinated reasoning.
- **`answers_match` accepts unit conversions** (cents↔dollars, minutes↔hours). Without this, ~5% of correct CoTs would be incorrectly filtered out due to unit mismatches between Claude's reasoning and GSM8K's gold format. The same comparison logic was used at training-data filtering and at eval time — so the comparison is internally consistent.

## What I learned

A handful of takeaways more durable than the +2pp number:

**Examples beat instructions.** I started with a system prompt telling Claude to reason in dollars rather than cents. It didn't work — Claude consistently produced `#### 300` for problems with gold answer `3` (300 cents = $3). Verifying the prompt was actually being delivered confirmed the model was just ignoring the instruction. Adding three few-shot demonstrations that *showed* the cent-to-dollar conversion fixed it on the first try. The lesson: when an LLM ignores an instruction, the fix is almost always more examples, not stronger wording. This generalizes to nearly any LLM-in-the-loop pipeline.

**Distillation has a capacity ceiling.** The same recipe that produces large gains at scale (DeepSeek's R1 distillation onto larger students) produces a ~2pp gain at 360M. The bottleneck isn't the data, the optimizer, or the teacher — it's the student's representational capacity. A 360M model has a hard ceiling on multi-step arithmetic that no amount of CoT scaffolding fixes. Realizing this changed how I think about scaling laws: parameter count isn't just about "more capacity" abstractly, it's the difference between a model that can and cannot perform a specific cognitive operation.

**Eval is half the project.** Building the eval harness *before* the training cell — and verifying the baseline number first — caught a bug in answer extraction that would have invalidated the entire delta. The discipline that matters: never trust a delta where the baseline wasn't measured rigorously, and always test the comparison logic on hand-picked edge cases (cents-vs-dollars, float-vs-int, multiple-numbers-in-output) before trusting it on a thousand examples.

**Greedy decoding has failure modes you should know about.** Repetition collapse is not a model bug; it's a property of greedy sampling on small models with high-confidence ruts. Sampling with low temperature broke some loops but didn't help accuracy, suggesting the loops are symptoms of capacity exhaustion rather than decoding pathology. Useful intuition for any future inference work.

**Model choice matters more than recipe at small scale.** A side experiment showed that the same distillation recipe on Qwen2.5-1.5B started from a 67% baseline — meaning there was almost no headroom for distillation to demonstrate gains. The takeaway: the "right" base model for a distillation experiment is one with measurable headroom on the target benchmark. Picking too capable a starting point hides the recipe's effect; picking too weak a starting point shows you what the recipe *can't* do (which is what this project does).

**Free compute is real but constrained.** The whole project ran on a free Kaggle T4 plus ~$3 in Anthropic API costs. The constraints — 30 hr/week GPU budget, 16GB VRAM, no multi-GPU — forced design decisions (LoRA over full fine-tuning, 360M base, batch size 4 with gradient accumulation) that turned out to be the right ones anyway. I'd recommend the constraints. They make you pick problems where the bottleneck is your understanding, not your wallet.

## Why I'm reporting this honestly

A 360M model going from 10.5% to 12.5% on GSM8K is not a flashy headline. It would be tempting to either drop the project, run it on a bigger base model until I got a prettier number, or fudge the eval. I chose not to because:

1. The methodology is the same — same pipeline, same rigor — whether the headline is +2pp or +20pp.
2. The qualitative findings (style transfer, capability ceiling, failure modes) are *more* interesting than a flattering number.
3. Reporting honestly is the part of ML I most want to be good at. Reaching for impressive numbers is the easiest way to lose a hiring manager who has done this work themselves.

If a smaller story told well is less appealing than a bigger story told sloppily, that's the signal I want to send.

## What I'd do next

Concrete extensions that fit the same infrastructure:

- **GRPO instead of SFT.** With verifiable rewards on GSM8K (binary: did the final answer match gold?), the same 1,500-problem dataset can be used as RL training data. This is the natural follow-on and is an active research area (DeepSeek-R1's recipe).
- **Larger student.** Same recipe on Qwen2.5-1.5B or Llama-3.2-1B would test whether the +2pp ceiling is specifically a 360M-scale phenomenon or a property of the recipe itself.
- **Process reward modeling.** Training a verifier on intermediate reasoning steps (not just final answers) could break the "correct format, wrong arithmetic" failure mode by penalizing bad sub-steps directly.

## Reproducibility

The full pipeline is in [`smollm2_cot_distillation_polished.ipynb`](smollm2_cot_distillation_polished.ipynb). All seeds are fixed (`random.seed(7)` for problem selection, `random.seed(42)` for splits, `np.random.default_rng(42)` for bootstrap). Training reproduces deterministically on a Kaggle T4 with the pinned package versions in the notebook's first cell.

To run inference on the trained model:

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="kianshandi/smollm2-360m-gsm8k-distilled-haiku")
pipe([{"role": "user", "content": "If a pencil costs 30 cents, how much do 4 pencils cost?"}])
```

## Hyperparameters (full)

| | |
|---|---|
| Base model | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| Teacher model | `claude-haiku-4-5` |
| Training data | 1,435 filtered (question, CoT) pairs |
| LoRA rank / alpha | 16 / 32 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Epochs | 3 |
| Effective batch size | 16 (per-device 4 × grad-accum 4) |
| Learning rate | 2e-4 (cosine schedule, 5% warmup) |
| Precision | bf16 |
| Max sequence length | 1024 |
| Held-out test set | 200 problems from GSM8K test split |
| Decoding | greedy (`do_sample=False`) |
| Compute | 1 × NVIDIA T4 (Kaggle free tier) |
| Wall-clock training time | ~20 min |
| API cost | ~$2.70 (Anthropic Haiku 4.5) |

## License & acknowledgements

Base model: SmolLM2-360M-Instruct (Apache 2.0) by HuggingFaceTB. Training data derived from GSM8K (MIT) via Claude Haiku 4.5 (Anthropic). This adapter is released under Apache 2.0.

Built as a portfolio project at UCLA, 2026.
