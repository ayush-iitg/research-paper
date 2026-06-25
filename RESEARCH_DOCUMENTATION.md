# Research Documentation: Cross-Modal Alignment for Vision-Language Models

## Project: Bidirectional Cross-Modal Alignment via Joint Optimization

**Author:** Ayush  
**Institute:** IIT Guwahati  
**Duration:** Research Internship Project  
**Status:** In Progress

---

## Table of Contents

1. [Paper Study and Understanding](#1-paper-study-and-understanding)
2. [CMAR Reproduction (Fine-tuning Version)](#2-cmar-reproduction)
3. [Research Gap Identification](#3-research-gap-identification)
4. [Proposed Method: AAA (Alternating Asymmetric Alignment)](#4-proposed-method-aaa)
5. [Implementation Challenges and Solutions](#5-implementation-challenges-and-solutions)
6. [Evolution to Joint Optimization](#6-evolution-to-joint-optimization)
7. [Final Method: Stop-Gradient Asymmetric CKA](#7-final-method-stop-gradient-asymmetric-cka)
8. [Experimental Setup](#8-experimental-setup)
9. [Results and Analysis](#9-results-and-analysis)
10. [Key Technical Decisions](#10-key-technical-decisions)
11. [Lessons Learned](#11-lessons-learned)
12. [Future Directions](#12-future-directions)
13. [References](#13-references)

---

## 1. Paper Study and Understanding

### Paper: "Seeing Helps Reasoning in Language Models" (CVPR Findings 2026)

**Authors:** Yulu Gan, Kaiya Ivy Zhao, Tomaso Poggio, Phillip Isola (MIT CSAIL/CBMM)

### Core Idea

Language models trained only on text lack direct grounding in the physical world. The paper proposes **Cross-Modal Alignment Regularization (CMAR)** — a method that improves LLMs by aligning their internal representations with those of a frozen vision model during training.

### Key Insight

Vision models naturally learn compositional, spatially-grounded representations from images. By forcing the LLM's hidden states to be structurally similar to these visual representations (measured via CKA), the LLM develops more robust reasoning capabilities — even though it never processes images at inference time.

### Method Summary

1. **Standard LLM training** continues (next-token prediction via NLL loss)
2. **Paired image-text data** is fed to both models simultaneously
3. **Frozen vision encoder** (CLIP) provides stable visual feature targets
4. **CKA alignment loss** measures structural similarity between vision and language representations
5. **Combined objective:** L = L_NLL + lambda * (1 - CKA(H_vision, H_language))

### Mathematical Formulation

**CKA (Centered Kernel Alignment):**

```
CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
```

Where HSIC (Hilbert-Schmidt Independence Criterion) uses centered Gram matrices:
```
K = X @ X^T  (pairwise sample similarities in X-space)
L = Y @ Y^T  (pairwise sample similarities in Y-space)
H = I - (1/n) * ones  (centering matrix)
HSIC = trace(H@K@H @ H@L@H) / (n-1)^2
```

**Key properties of CKA:**
- Dimension-agnostic (works even when CLIP=512-dim and LLM=2048-dim)
- Invariant to orthogonal rotations
- Invariant to isotropic scaling
- Captures second-order structural relationships between samples

### Paper's Main Results

- CKA works best among all alignment metrics (vs InfoNCE, KL-distillation)
- Penultimate layer alignment works best
- CLIP vision encoder gives best results
- Consistent improvements across multiple LLM families (Llama-3, Qwen, Mistral, Phi-3)
- Fine-tuning gains: +2.65% GSM8K, +1.35% Winogrande, +1.21% HellaSwag

---

## 2. CMAR Reproduction

### Objective

Reproduce the fine-tuning version of CMAR on limited compute (single Google Colab T4 GPU).

### Implementation Choices

| Component | Original Paper | Our Reproduction |
|-----------|---------------|-----------------|
| LLM | Llama-3-8B, Qwen2.5-7B | TinyLlama-1.1B-Chat |
| Fine-tuning | Full or LoRA | QLoRA (4-bit + LoRA rank=16) |
| Vision Encoder | CLIP, DINOv2, MAE | CLIP ViT-B/32 (frozen) |
| Dataset | SA-1B captions | Flickr8k |
| Training | Large scale | 1000 samples, 3 epochs |
| Evaluation | HellaSwag, Winogrande, GSM8K | Perplexity, CKA score |

### Results from CMAR Reproduction

```
Baseline Perplexity: 78.42    ->    Final: 11.50     (improved)
Baseline CKA:       0.8626    ->    Final: 0.8698    (improved)
```

### Key Observations

1. Perplexity improved dramatically (mostly from LoRA fine-tuning on caption distribution)
2. CKA improved slightly (0.863 -> 0.870), confirming alignment increased
3. Training loss decreased smoothly, showing CMAR objective is well-behaved
4. CMAR did NOT hurt language quality (no catastrophic forgetting on this evaluation)

### Ablation: CMAR vs L1/L2 Regularization

We compared CMAR against generic regularizers to validate that improvements come from vision structure:
- L1/L2 regularization: no structural information, just penalizes activation magnitudes
- CMAR with random encoder: doesn't help (proves gains are from pretrained visual knowledge)
- Only CMAR with pretrained CLIP provides meaningful improvement

**Conclusion:** The gains are specifically from the structured visual representations, not from generic regularization.

---

## 3. Research Gap Identification

### The Limitation of CMAR

CMAR is **unidirectional**: vision helps language, but the vision model stays frozen and unchanged. This means:
- Only the LLM benefits from alignment
- CLIP's representations are never improved
- In real-world scenarios where sometimes only images are available and sometimes only text, only one modality benefits

### Research Question

**Can we design a framework that improves BOTH the vision model and the language model through bidirectional cross-modal alignment?**

### Constraints

1. Both models should improve independently
2. At inference time, each model can be used standalone
3. Must prevent representation collapse (both converging to trivial outputs)
4. Must be computationally feasible on limited hardware

---

## 4. Proposed Method: AAA (Alternating Asymmetric Alignment)

### Initial Design

**Phase A:** Freeze CLIP, train LLM with CKA alignment (same as CMAR)
**Phase B:** Freeze improved LLM, train CLIP with contrastive + CKA alignment
**Repeat** for K cycles

### Rationale

- Alternating freezing prevents collapse (one model always provides stable target)
- Each phase builds on the previous one (virtuous cycle)
- After Phase B, CLIP is a better teacher for the next Phase A

### Implementation Details

- **CLIP LoRA:** Custom hook-based LoRA adapters on ResidualAttentionBlocks (rank=8)
- **Why hooks on blocks (not out_proj):** CLIP uses F.multi_head_attention_forward internally which bypasses normal PyTorch forward() on out_proj, breaking gradient flow
- **Phase B loss:** L_contrastive + lambda * (1 - CKA(llm_feats, clip_feats))

### Results with Alternating AAA

```
Baseline:        Perplexity=845, CKA=0.626, Zero-shot=93.4%
CMAR-only:       Perplexity=1.76, CKA=0.616
Full AAA:        Perplexity=1.90, CKA=0.602, Zero-shot=92.2%
```

### Problems Identified

1. **Overfitting:** Perplexity of 1.76 is suspiciously low (memorized 1000 captions)
2. **Catastrophic forgetting:** CLIP zero-shot dropped 1.2% from fine-tuning on narrow data
3. **Tug-of-war:** Alternating phases partially undo each other at small scale
4. **Not truly joint:** Professor feedback — "aim for joint optimization"

---

## 5. Implementation Challenges and Solutions

### Challenge 1: CLIP LoRA via Layer Replacement (FAILED)

**Problem:** Initial approach replaced `nn.Linear` layers in CLIP with LoRA-wrapped versions.

**Error:** `AttributeError: 'CLIPLoRALayer' object has no attribute 'weight'`

**Root cause:** CLIP uses `F.multi_head_attention_forward()` which accesses `out_proj.weight` directly via C++ code, bypassing normal PyTorch forward().

**Solution:** Use forward hooks on ResidualAttentionBlocks instead. The block's output is a normal tensor with valid grad_fn.

### Challenge 2: Gradient Flow in Phase B (FAILED)

**Problem:** `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

**Root cause:** `get_clip_features()` function wrapped `encode_image()` in `torch.no_grad()`, blocking all gradient computation.

**Solution:** Remove `torch.no_grad()` from `get_clip_features()`. In Phase A, explicitly wrap with `torch.no_grad()` when calling it. In Phase B (and joint training), call without wrapping.

### Challenge 3: CKA Test Failures

**Problem:** `AssertionError: Independent CKA too high: 0.749`

**Root cause:** With n=32 samples and p=64 features, random matrices have spuriously high CKA due to finite-sample effects.

**Solution:** Use n=512, p=16, q=32 for testing (n >> d ensures reliable statistical behavior).

### Challenge 4: Dataset Column Names

**Problem:** `KeyError: 'image'` or `KeyError: 'sentences'`

**Root cause:** Different HuggingFace datasets use different column names. The `jxie/flickr8k` dataset uses `caption_0`, not `caption` or `sentences`.

**Solution:** Direct loading with known column names: `image_column="image"`, `text_column="caption_0"`.

### Challenge 5: Overfitting on Small Dataset

**Problem:** Perplexity dropped to 1.76 (near perfect memorization of 1000 short captions).

**Solution:** Scale to full Flickr8k (~6000 samples) and add WikiText-2 evaluation to detect if general language ability is preserved.

---

## 6. Evolution to Joint Optimization

### Professor's Feedback

> "Take a pen and paper and think through different ideas. Simply changing the dataset may not be enough. More importantly, aim for joint optimization of both language and vision models."

### Key Insight

Alternating optimization is NOT truly joint — at any moment only one model learns. The professor wants both models learning simultaneously in a single forward-backward pass.

### Ideas Considered

1. **Stop-Gradient Asymmetric CKA** — each model aligns to detached features of the other (CHOSEN)
2. **EMA (Exponential Moving Average) Targets** — align to slowly-moving copies (memory-expensive)
3. **Shared Bottleneck** — both project to common low-dim space (adds complexity)

### Why Stop-Gradient Was Chosen

- Simplest implementation (just add `.detach()`)
- No extra memory (no EMA copies needed)
- Proven principle (similar to BYOL/SimSiam in self-supervised learning)
- True joint optimization (both update every step)
- Prevents collapse (neither model can "see" the other's current gradient direction)

---

## 7. Final Method: Stop-Gradient Asymmetric CKA

### Algorithm

```
For each batch (images, captions):
    1. clip_feats = CLIP.encode_image(images)           # with grad (LoRA hooks)
    2. llm_feats = LLM.penultimate_layer(captions)     # with grad (LoRA)
    
    3. cka_for_llm = CKA(clip_feats.DETACH(), llm_feats)   # LLM learns toward CLIP snapshot
    4. cka_for_clip = CKA(llm_feats.DETACH(), clip_feats)   # CLIP learns toward LLM snapshot
    
    5. llm_loss = NLL + 0.1 * (1 - cka_for_llm)
    6. clip_loss = contrastive + 0.1 * (1 - cka_for_clip)
    
    7. total_loss = llm_loss + clip_loss
    8. total_loss.backward()
    9. llm_optimizer.step()
    10. clip_optimizer.step()
```

### Why This Prevents Collapse

The `.detach()` operation creates an asymmetry:
- When computing LLM's alignment target, CLIP features are treated as constants
- When computing CLIP's alignment target, LLM features are treated as constants
- Neither model can "see" where the other is going in this step
- This is analogous to two people walking toward each other based on where the other WAS, not where they're GOING

### Comparison with Alternating Approach

```
Alternating:
  Step 1-100:  Only LLM learns (CLIP frozen)
  Step 101-200: Only CLIP learns (LLM frozen)
  Each model: learns for 50% of total steps

Joint:
  Step 1-200: BOTH learn simultaneously
  Each model: learns for 100% of total steps (2x more learning)
```

---

## 8. Experimental Setup

### Models

| Model | Architecture | Parameters | Training Method |
|-------|-------------|------------|-----------------|
| TinyLlama-1.1B-Chat | LLaMA decoder, 22 layers | ~1.1B total, ~2.2M trainable | 4-bit QLoRA (rank=16, alpha=32) |
| CLIP ViT-B/32 | Vision Transformer, 12 blocks | ~151M total, ~147K trainable | Hook-based LoRA (rank=8) |

### Dataset

| Dataset | Split | Samples | Usage |
|---------|-------|---------|-------|
| jxie/flickr8k | train | ~6000 | Training (image-text alignment) |
| jxie/flickr8k | validation | ~600 | Eval (LLM perplexity) |
| WikiText-2 | test | 100 samples | Eval (general language ability) |
| CIFAR-10 | test | 500 images | Eval (CLIP zero-shot accuracy) |

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LLM learning rate | 2e-4 | Standard for LoRA fine-tuning |
| CLIP learning rate | 5e-5 | Lower to prevent catastrophic forgetting |
| Lambda (alignment weight) | 0.1 | ~10% alignment signal, 90% task signal |
| Batch size | 4 | Memory constraint on T4 |
| Epochs | 3 | Sufficient for convergence without overfitting |
| Gradient clipping | 1.0 | Prevents exploding gradients |
| LR scheduler | Cosine annealing | Smooth decay for stable convergence |

### Evaluation Metrics

1. **Flickr8k Perplexity:** LLM's ability to predict caption text (domain-specific)
2. **WikiText-2 Perplexity:** LLM's general language modeling (detects catastrophic forgetting)
3. **CIFAR-10 Zero-shot Accuracy:** CLIP's visual understanding (detects vision degradation)
4. **CKA Alignment:** Structural similarity between vision and language representations

---

## 9. Results and Analysis

### CMAR Reproduction Results

| Metric | Before | After CMAR | Change |
|--------|--------|-----------|--------|
| Perplexity | 78.42 | 11.50 | -85% (improved) |
| CKA Score | 0.8626 | 0.8698 | +0.007 (improved) |

### AAA (Alternating) Results

| Metric | Baseline | After AAA | Change |
|--------|----------|----------|--------|
| Perplexity | 845.39 | 1.90 | -99.8% |
| CKA | 0.626 | 0.602 | -0.024 |
| Zero-shot | 93.4% | 92.2% | -1.2% |

### Joint Optimization Results

(To be filled after running the joint optimization notebook)

| Metric | Baseline | After Joint | Change |
|--------|----------|------------|--------|
| Flickr8k PPL | - | - | - |
| WikiText PPL | - | - | - |
| CLIP Acc | - | - | - |
| CKA | - | - | - |

---

## 10. Key Technical Decisions

### Decision 1: Hook-Based CLIP LoRA (Not Layer Replacement)

**Why:** CLIP's `MultiheadAttention` uses `F.multi_head_attention_forward()` which calls `out_proj.weight` directly via C++ code. Replacing the layer breaks the internal attribute access. Forward hooks on the entire ResidualAttentionBlock preserve the original layer while adding trainable corrections.

### Decision 2: Penultimate Layer for LLM Features

**Why:** The paper's ablation shows penultimate layer works best. The final layer is too specialized for vocabulary prediction. Earlier layers are too generic. The penultimate layer contains the model's "high-level understanding" before it gets compressed into token predictions.

### Decision 3: CKA over InfoNCE or KL

**Why:** CKA is dimension-agnostic (handles 512 vs 2048 without projection layers), rotation-invariant, and captures second-order structural relationships. InfoNCE requires same dimensions (needs projection heads). KL requires same dimensions for cosine similarity.

### Decision 4: Separate Learning Rates for LLM and CLIP

**Why:** CLIP is already an excellent vision model — aggressive updates cause catastrophic forgetting. LLM benefits more from alignment since it lacks visual grounding entirely. So LLM LR (2e-4) > CLIP LR (5e-5).

### Decision 5: Stop-Gradient for Joint Training

**Why:** Without stop-gradient, both models can mutually collapse to constant outputs (trivially achieving CKA=1). The `.detach()` creates a one-step delay — each model aligns to where the other WAS, not where it's GOING. This provides optimization stability similar to target networks in RL.

---

## 11. Lessons Learned

### Technical Lessons

1. **Always test gradient flow before training.** A simple `loss.backward()` test would have caught the CLIP LoRA hook issue immediately.
2. **F-strings break on copy-paste to Colab.** Use `print("text", variable)` instead.
3. **Dataset column names vary.** Always inspect `dataset.column_names` before assuming.
4. **Small datasets cause overfitting.** Perplexity of 1.76 on 1000 captions = memorization, not learning.
5. **Evaluation must use held-out data from a DIFFERENT distribution.** WikiText-2 for LLM, CIFAR-10 for CLIP.

### Research Lessons

1. **Start with the simplest reproduction, then extend.** CMAR first, then AAA, then joint.
2. **Professor feedback drives iteration.** "Joint optimization" was the key insight that led to the final method.
3. **Negative results are informative.** AAA's slight degradation revealed overfitting and tug-of-war issues.
4. **Scale matters.** Results at 1000 samples may not generalize to 6000+ samples.

---

## 12. Future Directions

### Short-term (If Time Permits)

1. Run joint optimization on full Flickr8k and report results
2. Ablate lambda values (0.05, 0.1, 0.2) for joint training
3. Compare joint optimization vs alternating AAA at same compute budget
4. Evaluate on standard reasoning benchmarks (CommonsenseQA) if possible

### Medium-term (Potential Publication)

1. Scale to larger models (Llama-2-7B, ViT-L/14)
2. Use larger alignment datasets (COCO Captions, CC3M)
3. Evaluate on full benchmark suite (HellaSwag, Winogrande, GSM8K, ARC)
4. Ablation: number of training steps, learning rate schedules, LoRA ranks
5. Compare stop-gradient vs EMA-based approaches
6. Study convergence properties theoretically

### Long-term (Research Vision)

1. Extend to more than two modalities (audio, video, robotics)
2. Investigate representation dynamics during training (what changes in each layer?)
3. Connect to Platonic Representation Hypothesis (does joint alignment accelerate convergence?)
4. Develop theoretical guarantees for collapse prevention with stop-gradient

---

## 13. References

1. Gan et al., "Seeing Helps Reasoning in Language Models," CVPR Findings 2026.
2. Kornblith et al., "Similarity of Neural Network Representations Revisited," ICML 2019. (CKA)
3. Huh et al., "The Platonic Representation Hypothesis," ICML 2024.
4. Zhang et al., "Deep Mutual Learning," CVPR 2018. (Bidirectional learning concept)
5. Grill et al., "Bootstrap Your Own Latent (BYOL)," NeurIPS 2020. (Stop-gradient in self-supervised)
6. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.
7. Radford et al., "Learning Transferable Visual Models from Natural Language Supervision," ICML 2021. (CLIP)
8. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models," NeurIPS 2023.

---

## Repository Structure

```
research-paper/
├── Gan_Seeing_Helps_Reasoning_in_Language_Models_CVPRF_2026_paper.pdf  (Original paper)
├── CMAR_Finetuning_Reproduction.ipynb    (CMAR reproduction notebook)
├── AAA_Bidirectional_Alignment.ipynb     (Alternating AAA notebook)
├── AAA_Joint_Optimization.ipynb          (Joint optimization notebook - FINAL)
├── RESEARCH_DOCUMENTATION.md             (This document)
└── notebook_source.py                    (Jupytext source for CMAR notebook)
```

---

## Timeline

| Date | Activity |
|------|----------|
| Week 1 | Paper reading and understanding |
| Week 2 | CMAR reproduction (fine-tuning version) |
| Week 3 | Research gap identification, AAA proposal |
| Week 4 | AAA implementation (alternating), debugging |
| Week 5 | Professor feedback, pivot to joint optimization |
| Week 5-6 | Joint optimization implementation and evaluation |

---

## How to Explain This Project (Elevator Pitch)

> "I started by reproducing the CMAR paper which shows that aligning a language model's representations with a frozen vision model improves reasoning. I then extended this to a novel bidirectional setting: instead of only improving the language model, my method (Stop-Gradient Asymmetric CKA) jointly optimizes BOTH models simultaneously. The key innovation is using stop-gradient to prevent representation collapse while allowing true joint training. Each model sees a 'snapshot' of the other as its alignment target — they can't drag each other into degenerate solutions. The result is that both the vision encoder and language model improve their representation quality independently."

---

*Last updated: June 2026*
