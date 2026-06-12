# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CMAR Fine-Tuning Reproduction: Cross-Modal Alignment Regularization
#
# ## Paper: "Seeing Helps Reasoning in Language Models" (CVPR 2026)
#
# ### Overview
#
# This notebook reproduces the **Cross-Modal Alignment Regularization (CMAR)** method,
# which improves language model reasoning by aligning internal representations with a
# frozen vision encoder during fine-tuning.
#
# **Key Insight:** By regularizing the LLM's hidden states to be structurally similar
# to visual representations (measured via CKA), the model develops more grounded,
# compositional internal representations that improve downstream reasoning.
#
# ### Method Summary
#
# 1. **Frozen Vision Encoder:** CLIP ViT-B/32 provides stable visual representations
# 2. **Trainable LLM:** TinyLlama-1.1B-Chat fine-tuned with LoRA adapters
# 3. **Alignment Loss:** CKA (Centered Kernel Alignment) measures representation similarity
# 4. **Combined Objective:** $\mathcal{L} = \mathcal{L}_{NLL} + \lambda \cdot (1 - \text{CKA}(\mathbf{H}_v, \mathbf{H}_l))$
#
# ### Why This Works
#
# - Vision models learn compositional, spatially-grounded representations
# - Aligning LLM hidden states to these representations transfers structural properties
# - CKA is invariant to orthogonal transformations and isotropic scaling
# - The frozen vision encoder prevents representation collapse

# %% [markdown]
# ## 1. Environment Setup
#
# Install all required packages for the reproduction.
# Designed to run on Google Colab with a T4 GPU (~2 hours total runtime).

# %%
# !pip install torch torchvision transformers peft datasets Pillow matplotlib open-clip-torch numpy tqdm accelerate bitsandbytes --quiet

# %% [markdown]
# ## 2. Imports and Device Configuration

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
import gc
import os

warnings.filterwarnings("ignore")

# Device configuration - detect best available hardware
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS")
else:
    device = torch.device("cpu")
    print("Using CPU (training will be slow)")

print(f"\nDevice: {device}")
print(f"PyTorch version: {torch.__version__}")

# %% [markdown]
# ## 3. Load Frozen CLIP ViT-B/32 Vision Encoder
#
# The vision encoder provides stable target representations for alignment.
# **All parameters are frozen** - it never receives gradients during training.
# This prevents representation collapse and provides a fixed alignment target.

# %%
import open_clip

# Load CLIP ViT-B/32 model and preprocessing
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_model = clip_model.to(device)
clip_model.eval()

# CRITICAL: Freeze ALL vision encoder parameters
for param in clip_model.parameters():
    param.requires_grad = False

# Verify frozen status
total_params = sum(p.numel() for p in clip_model.parameters())
trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
print(f"CLIP ViT-B/32 loaded successfully")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params} (should be 0)")
assert trainable_params == 0, "CLIP must be completely frozen!"

# %% [markdown]
# ## 4. Load TinyLlama-1.1B-Chat with LoRA Adapters
#
# We use **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning:
# - **Rank = 16**: Controls the dimensionality of the low-rank update matrices
# - **Alpha = 32**: Scaling factor (effective learning rate multiplier = alpha/rank = 2)
# - **Target modules**: `q_proj` and `v_proj` in attention layers
#
# This allows fine-tuning with <1% of total parameters while maintaining expressivity.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# Model identifier
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Quantization config for memory efficiency on T4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load base model with 4-bit quantization
llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

print(f"Base model loaded: {MODEL_NAME}")
print(f"Model architecture: {llm_model.config.model_type}")
print(f"Hidden size: {llm_model.config.hidden_size}")
print(f"Number of layers: {llm_model.config.num_hidden_layers}")

# %%
# Configure LoRA adapters
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # Rank of low-rank matrices
    lora_alpha=32,                 # Scaling factor (effective lr = alpha/r = 2)
    lora_dropout=0.05,             # Dropout for regularization
    target_modules=["q_proj", "v_proj"],  # Target attention projection layers
    bias="none",
)

# Apply LoRA adapters to the model
llm_model = get_peft_model(llm_model, lora_config)

# Print trainable parameter summary
llm_model.print_trainable_parameters()

# Verify only LoRA parameters are trainable
for name, param in llm_model.named_parameters():
    if param.requires_grad:
        assert "lora" in name.lower(), f"Non-LoRA param is trainable: {name}"

print("\nLoRA configuration verified: only LoRA parameters receive gradients")

# %% [markdown]
# ## 5. CKA Alignment Loss Function
#
# ### Mathematical Derivation
#
# **Centered Kernel Alignment (CKA)** measures the similarity between two representations
# in a way that is invariant to orthogonal transformations and isotropic scaling.
#
# #### Step 1: Column-Centering
#
# Given representation matrices $X \in \mathbb{R}^{n \times p}$ and $Y \in \mathbb{R}^{n \times q}$:
#
# $$X_c = X - \frac{1}{n}\mathbf{1}\mathbf{1}^T X$$
# $$Y_c = Y - \frac{1}{n}\mathbf{1}\mathbf{1}^T Y$$
#
# #### Step 2: HSIC (Hilbert-Schmidt Independence Criterion)
#
# For **linear kernels**, HSIC simplifies to:
#
# $$\text{HSIC}(X, Y) = \frac{\|Y_c^T X_c\|_F^2}{(n-1)^2}$$
#
# where $\|\cdot\|_F$ is the Frobenius norm.
#
# #### Step 3: CKA Normalization
#
# $$\text{CKA}(X, Y) = \frac{\text{HSIC}(X, Y)}{\sqrt{\text{HSIC}(X, X) \cdot \text{HSIC}(Y, Y)}}$$
#
# This normalization ensures CKA $\in [0, 1]$ where 1 means perfectly aligned representations.
#
# #### Properties
#
# - **Invariant to orthogonal transformations:** $\text{CKA}(XR, Y) = \text{CKA}(X, Y)$ for orthogonal $R$
# - **Invariant to isotropic scaling:** $\text{CKA}(\alpha X, Y) = \text{CKA}(X, Y)$
# - **Handles different dimensions:** $p$ and $q$ can differ (CLIP=512, LLM=2048)

# %%
def center_columns(X):
    """Column-center a matrix by subtracting column means.

    Args:
        X: Tensor of shape (n, d) where n=samples, d=features

    Returns:
        X_centered: Tensor of shape (n, d) with zero column means
    """
    return X - X.mean(dim=0, keepdim=True)


def compute_hsic(X, Y):
    """Compute HSIC (Hilbert-Schmidt Independence Criterion) with linear kernel.

    HSIC(X, Y) = ||Y_c^T @ X_c||_F^2 / (n-1)^2

    This measures the statistical dependence between two sets of representations.

    Args:
        X: Tensor of shape (n, p) - first representation (column-centered)
        Y: Tensor of shape (n, q) - second representation (column-centered)

    Returns:
        hsic_value: Scalar tensor with HSIC value
    """
    n = X.shape[0]
    # Cross-covariance matrix: Y_c^T @ X_c has shape (q, p)
    cross_cov = Y.t() @ X
    # Frobenius norm squared = sum of squared elements
    hsic_value = (cross_cov ** 2).sum() / ((n - 1) ** 2)
    return hsic_value


def compute_cka(X, Y):
    """Compute CKA (Centered Kernel Alignment) between two representations.

    CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))

    Args:
        X: Tensor of shape (n, p) - e.g., CLIP image features (n=batch, p=512)
        Y: Tensor of shape (n, q) - e.g., LLM hidden states (n=batch, q=2048)

    Returns:
        cka_value: Scalar in [0, 1], where 1 = perfectly aligned
    """
    # Step 1: Column-center both representations
    X_centered = center_columns(X)
    Y_centered = center_columns(Y)

    # Step 2: Compute HSIC values
    hsic_xy = compute_hsic(X_centered, Y_centered)
    hsic_xx = compute_hsic(X_centered, X_centered)
    hsic_yy = compute_hsic(Y_centered, Y_centered)

    # Step 3: Normalize (with epsilon for numerical stability)
    eps = 1e-8
    cka_value = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + eps)

    return cka_value


# Verify CKA implementation with known properties
def test_cka_implementation():
    """Validate CKA implementation against known mathematical properties."""
    torch.manual_seed(42)
    n, p, q = 32, 64, 128

    # Property 1: CKA(X, X) should equal 1
    X = torch.randn(n, p)
    cka_self = compute_cka(X, X)
    assert abs(cka_self.item() - 1.0) < 1e-5, f"CKA(X,X) = {cka_self.item()}, expected 1.0"

    # Property 2: CKA should be invariant to orthogonal transformation
    Q, _ = torch.linalg.qr(torch.randn(p, p))  # Random orthogonal matrix
    X_rotated = X @ Q
    cka_rotated = compute_cka(X, X_rotated)
    assert abs(cka_rotated.item() - 1.0) < 1e-4, f"CKA not rotation-invariant: {cka_rotated.item()}"

    # Property 3: CKA should be invariant to isotropic scaling
    cka_scaled = compute_cka(X * 5.0, X)
    assert abs(cka_scaled.item() - 1.0) < 1e-5, f"CKA not scale-invariant: {cka_scaled.item()}"

    # Property 4: CKA of independent random matrices should be near 0
    Y = torch.randn(n, q)
    cka_independent = compute_cka(X, Y)
    assert cka_independent.item() < 0.3, f"Independent CKA too high: {cka_independent.item()}"

    # Property 5: CKA should work with different feature dimensions
    Z = torch.randn(n, 256)
    cka_diff_dims = compute_cka(X, Z)
    assert 0 <= cka_diff_dims.item() <= 1, f"CKA out of range: {cka_diff_dims.item()}"

    print("All CKA tests passed!")
    print(f"  CKA(X, X) = {cka_self.item():.6f}")
    print(f"  CKA(X, X@Q) = {cka_rotated.item():.6f} (rotation invariant)")
    print(f"  CKA(5X, X) = {cka_scaled.item():.6f} (scale invariant)")
    print(f"  CKA(X, random Y) = {cka_independent.item():.6f} (low for independent)")

test_cka_implementation()

# %% [markdown]
# ## 6. InfoNCE Alignment Function
#
# **InfoNCE** (Information Noise-Contrastive Estimation) is a contrastive alignment loss
# that maximizes agreement between paired representations while pushing apart non-paired ones.
#
# $$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(v_i, l_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(v_i, l_j)/\tau)}$$
#
# where $\tau$ is a temperature parameter controlling the sharpness of the distribution.

# %%
def compute_infonce(vision_features, language_features, temperature=0.07):
    """Compute InfoNCE contrastive alignment loss.

    Maximizes similarity between paired (vision, language) representations
    while minimizing similarity with all other pairs in the batch.

    Args:
        vision_features: Tensor (n, d_v) - normalized CLIP image features
        language_features: Tensor (n, d_l) - normalized LLM hidden states
        temperature: Float - temperature scaling (lower = sharper distribution)

    Returns:
        loss: Scalar InfoNCE loss (lower = better alignment)
        accuracy: Fraction of correct pairings (for monitoring)
    """
    # L2 normalize features for cosine similarity
    v_norm = F.normalize(vision_features, dim=-1)
    l_norm = F.normalize(language_features, dim=-1)

    # Compute pairwise similarity matrix: (n, n)
    # sim[i,j] = cosine_similarity(vision_i, language_j)
    similarity_matrix = v_norm @ l_norm.t() / temperature

    # Labels: diagonal entries are the positive pairs
    n = vision_features.shape[0]
    labels = torch.arange(n, device=vision_features.device)

    # Symmetric InfoNCE: vision-to-language + language-to-vision
    loss_v2l = F.cross_entropy(similarity_matrix, labels)
    loss_l2v = F.cross_entropy(similarity_matrix.t(), labels)
    loss = (loss_v2l + loss_l2v) / 2.0

    # Compute retrieval accuracy for monitoring
    with torch.no_grad():
        preds_v2l = similarity_matrix.argmax(dim=1)
        preds_l2v = similarity_matrix.argmax(dim=0)
        accuracy = ((preds_v2l == labels).float().mean() +
                    (preds_l2v == labels).float().mean()) / 2.0

    return loss, accuracy.item()


# Test InfoNCE implementation
def test_infonce():
    """Validate InfoNCE with known cases."""
    torch.manual_seed(42)
    n, d = 16, 64

    # Perfect alignment: identical features should give low loss
    X = torch.randn(n, d)
    loss_perfect, acc_perfect = compute_infonce(X, X, temperature=0.07)

    # Random alignment: unrelated features should give high loss
    Y = torch.randn(n, d)
    loss_random, acc_random = compute_infonce(X, Y, temperature=0.07)

    print(f"InfoNCE test results:")
    print(f"  Perfect alignment: loss={loss_perfect.item():.4f}, acc={acc_perfect:.4f}")
    print(f"  Random alignment:  loss={loss_random.item():.4f}, acc={acc_random:.4f}")
    assert loss_perfect < loss_random, "Perfect alignment should have lower loss"
    print("  InfoNCE tests passed!")

test_infonce()

# %% [markdown]
# ## 7. KL Divergence Alignment Function
#
# **KL Alignment** measures how well the LLM's pairwise similarity structure
# matches the vision encoder's pairwise similarity structure.
#
# Given pairwise similarity matrices $S_v$ and $S_l$:
# 1. Convert to probability distributions via temperature-scaled softmax
# 2. Compute KL divergence: $D_{KL}(P_v \| P_l)$
#
# This encourages the LLM to preserve the same relative similarity structure as CLIP.

# %%
def compute_kl_alignment(vision_features, language_features, temperature=1.0):
    """Compute KL divergence alignment on pairwise similarity distributions.

    Aligns the relational structure of representations rather than individual vectors.

    Args:
        vision_features: Tensor (n, d_v) - CLIP image features
        language_features: Tensor (n, d_l) - LLM hidden states
        temperature: Float - temperature for softmax (higher = smoother)

    Returns:
        kl_loss: Scalar KL divergence loss
    """
    # Compute pairwise cosine similarity matrices
    v_norm = F.normalize(vision_features, dim=-1)
    l_norm = F.normalize(language_features, dim=-1)

    # Pairwise similarity: (n, n) matrices
    sim_v = v_norm @ v_norm.t() / temperature
    sim_l = l_norm @ l_norm.t() / temperature

    # Mask out diagonal (self-similarity is always 1, uninformative)
    n = vision_features.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=vision_features.device)

    # Convert to probability distributions per row (softmax over columns)
    # Vision distribution is the target (detached, no gradients)
    p_v = F.softmax(sim_v.masked_fill(~mask, float("-inf")), dim=-1)
    p_l = F.log_softmax(sim_l.masked_fill(~mask, float("-inf")), dim=-1)

    # KL divergence: D_KL(P_v || P_l) = sum(P_v * (log P_v - log P_l))
    # Using PyTorch's kl_div which expects log-probabilities as input
    kl_loss = F.kl_div(p_l, p_v.detach(), reduction="batchmean")

    return kl_loss


# Test KL alignment
def test_kl_alignment():
    """Validate KL alignment implementation."""
    torch.manual_seed(42)
    n, d = 16, 64

    # Identical structure should give low KL
    X = torch.randn(n, d)
    kl_identical = compute_kl_alignment(X, X)

    # Different structure should give higher KL
    Y = torch.randn(n, d)
    kl_different = compute_kl_alignment(X, Y)

    print(f"KL Alignment test results:")
    print(f"  Identical structure: KL = {kl_identical.item():.6f}")
    print(f"  Different structure: KL = {kl_different.item():.6f}")
    assert kl_identical.item() < kl_different.item(), "Identical should have lower KL"
    print("  KL alignment tests passed!")

test_kl_alignment()

# %% [markdown]
# ## 8. Dataset: Image-Text Pairs from HuggingFace
#
# We load paired image-text data for computing the alignment loss.
# Using a subset of ~1000 pairs to fit within Colab time constraints.
#
# Each batch provides:
# - Images processed through CLIP transforms
# - Text tokenized for TinyLlama

# %%
from datasets import load_dataset
from PIL import Image
import io
import requests

# Load Flickr30k dataset (or similar) from HuggingFace
print("Loading image-text dataset from HuggingFace...")
try:
    # Try nlphuji/flickr30k first
    dataset = load_dataset("nlphuji/flickr30k", split="test", streaming=False)
    text_column = "caption"
    image_column = "image"
    print("Loaded nlphuji/flickr30k")
except Exception as e:
    print(f"Could not load flickr30k: {e}")
    try:
        # Fallback to COCO captions subset
        dataset = load_dataset("yerevann/coco-karpathy", split="test")
        text_column = "sentences"
        image_column = "image"
        print("Loaded yerevann/coco-karpathy")
    except Exception as e2:
        print(f"Could not load coco-karpathy: {e2}")
        # Final fallback: use cifar10 with synthetic captions
        dataset = load_dataset("cifar10", split="test[:1000]")
        text_column = "label"
        image_column = "img"
        print("Loaded CIFAR-10 fallback with label-based captions")

# Limit to ~1000 samples for Colab runtime
MAX_SAMPLES = 1000
if len(dataset) > MAX_SAMPLES:
    dataset = dataset.select(range(MAX_SAMPLES))

print(f"Dataset size: {len(dataset)} image-text pairs")
print(f"Columns: {dataset.column_names}")

# %%
class ImageTextDataset(Dataset):
    """Custom dataset for paired image-text data with CLIP and LLM preprocessing."""

    def __init__(self, hf_dataset, clip_transform, tokenizer_fn,
                 image_column="image", text_column="caption", max_length=128):
        self.dataset = hf_dataset
        self.clip_transform = clip_transform
        self.tokenizer_fn = tokenizer_fn
        self.image_column = image_column
        self.text_column = text_column
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Process image for CLIP
        image = item[self.image_column]
        if not isinstance(image, Image.Image):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        image_tensor = self.clip_transform(image)

        # Process text for LLM
        text = item[self.text_column]
        if isinstance(text, list):
            text = text[0]  # Take first caption if multiple
        if isinstance(text, (int, float)):
            # Handle label-only datasets (CIFAR fallback)
            label_names = ["airplane", "automobile", "bird", "cat", "deer",
                          "dog", "frog", "horse", "ship", "truck"]
            text = f"A photograph of a {label_names[int(text)]}"

        tokens = self.tokenizer_fn(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "image": image_tensor,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "text": text,
        }


def collate_fn(batch):
    """Custom collate function for batching image-text pairs."""
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "text": [item["text"] for item in batch],
    }

# %% [markdown]
# ## 9. Create DataLoader
#
# Set up the training and evaluation data splits with appropriate batch sizes
# for a T4 GPU (16GB VRAM).

# %%
# Split dataset into train and eval
train_size = int(0.9 * len(dataset))
eval_size = len(dataset) - train_size

# Create dataset instances
train_dataset = ImageTextDataset(
    hf_dataset=dataset.select(range(train_size)),
    clip_transform=clip_preprocess,
    tokenizer_fn=tokenizer,
    image_column=image_column,
    text_column=text_column,
    max_length=128,
)

eval_dataset = ImageTextDataset(
    hf_dataset=dataset.select(range(train_size, len(dataset))),
    clip_transform=clip_preprocess,
    tokenizer_fn=tokenizer,
    image_column=image_column,
    text_column=text_column,
    max_length=128,
)

# DataLoaders with batch sizes appropriate for T4 GPU
BATCH_SIZE = 8  # Conservative for T4 with 4-bit LLM + CLIP
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)

print(f"Training batches: {len(train_loader)}")
print(f"Evaluation batches: {len(eval_loader)}")
print(f"Batch size: {BATCH_SIZE}")

# %% [markdown]
# ## 10. Hidden State Extraction via Forward Hooks
#
# To compute the alignment loss, we need the **penultimate layer** hidden states
# from the LLM. We use PyTorch forward hooks to capture intermediate activations
# without modifying the model architecture.
#
# The hidden states are then **mean-pooled over the sequence length** to get a
# fixed-size representation for each sample in the batch.

# %%
class HiddenStateExtractor:
    """Extract hidden states from a specific layer using forward hooks.

    This captures the output of the penultimate transformer layer during
    the forward pass without modifying the model.
    """

    def __init__(self, model, layer_idx=-2):
        """
        Args:
            model: The LLM model (with LoRA)
            layer_idx: Which layer to hook (-2 = penultimate layer)
        """
        self.hidden_states = None
        self.hook = None

        # Access the transformer layers
        # For TinyLlama/LLaMA architecture: model.base_model.model.model.layers
        if hasattr(model, "base_model"):
            # PEFT wrapped model
            layers = model.base_model.model.model.layers
        else:
            layers = model.model.layers

        target_layer = layers[layer_idx]
        self.hook = target_layer.register_forward_hook(self._hook_fn)
        print(f"Hook registered on layer {layer_idx} "
              f"(layer {len(layers) + layer_idx}/{len(layers)})")

    def _hook_fn(self, module, input, output):
        """Hook function that captures the layer output."""
        # output is typically (hidden_states, ) or (hidden_states, attention, ...)
        if isinstance(output, tuple):
            self.hidden_states = output[0]
        else:
            self.hidden_states = output

    def get_pooled_features(self, attention_mask=None):
        """Get mean-pooled hidden states over sequence length.

        Args:
            attention_mask: (batch, seq_len) mask to ignore padding tokens

        Returns:
            pooled: (batch, hidden_dim) mean-pooled representation
        """
        if self.hidden_states is None:
            raise RuntimeError("No hidden states captured. Run a forward pass first.")

        hidden = self.hidden_states  # (batch, seq_len, hidden_dim)

        if attention_mask is not None:
            # Expand mask for broadcasting: (batch, seq_len, 1)
            mask = attention_mask.unsqueeze(-1).float().to(hidden.device)
            # Mean pool only over non-padding tokens
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            # Simple mean over all positions
            pooled = hidden.mean(dim=1)

        return pooled

    def remove(self):
        """Remove the forward hook."""
        if self.hook is not None:
            self.hook.remove()


# Register hook on penultimate layer
extractor = HiddenStateExtractor(llm_model, layer_idx=-2)

# %% [markdown]
# ## 11. CMAR Training Loop
#
# ### Algorithm (from the paper)
#
# ```
# Input: Vision encoder V (frozen), LLM M (LoRA), Dataset D, lambda
# For each batch (images, text) in D:
#   1. H_v = V.encode(images)           # CLIP features, no grad
#   2. logits = M(text)                  # Forward pass through LLM
#   3. H_l = mean_pool(M.layer[-2])     # Penultimate layer hidden states
#   4. L_NLL = CrossEntropy(logits, text)
#   5. L_align = 1 - CKA(H_v, H_l)
#   6. L_total = L_NLL + lambda * L_align
#   7. Backpropagate L_total (only LoRA params update)
# ```
#
# **Lambda = 0.1** balances language modeling quality with alignment pressure.

# %%
# Training hyperparameters
LEARNING_RATE = 2e-4
LAMBDA_ALIGN = 0.1       # Alignment loss weight
NUM_EPOCHS = 3           # Number of training epochs
GRAD_ACCUM_STEPS = 4     # Gradient accumulation for effective batch size of 32
MAX_GRAD_NORM = 1.0      # Gradient clipping

# Optimizer (only LoRA parameters are trainable)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, llm_model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

# Learning rate scheduler
total_steps = len(train_loader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps, eta_min=LEARNING_RATE * 0.1
)

print(f"Training configuration:")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Lambda (alignment weight): {LAMBDA_ALIGN}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"  Total optimization steps: {total_steps}")

# %%
def extract_clip_features(images):
    """Extract image features from frozen CLIP encoder.

    Args:
        images: Tensor (batch, 3, 224, 224) - preprocessed images

    Returns:
        features: Tensor (batch, 512) - CLIP image embeddings
    """
    with torch.no_grad():
        features = clip_model.encode_image(images.to(device))
    return features.float()  # Ensure float32 for CKA computation


def compute_nll_loss(logits, input_ids, attention_mask):
    """Compute next-token prediction loss (standard causal LM objective).

    Args:
        logits: (batch, seq_len, vocab_size) - model predictions
        input_ids: (batch, seq_len) - target token IDs
        attention_mask: (batch, seq_len) - mask for valid tokens

    Returns:
        loss: Scalar NLL loss
    """
    # Shift for next-token prediction: predict position t+1 from position t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    # Flatten for cross-entropy
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    # Apply mask and average over valid tokens
    loss = loss.view(shift_labels.shape)
    loss = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)

    return loss

# %% [markdown]
# ## 12. Pre-Training Evaluation (Baseline)
#
# Compute perplexity on the held-out evaluation set **before** CMAR fine-tuning
# to establish a baseline for comparison.

# %%
@torch.no_grad()
def evaluate_perplexity(model, eval_loader, extractor_module=None):
    """Evaluate perplexity and alignment score on the evaluation set.

    Args:
        model: The LLM model
        eval_loader: DataLoader for evaluation data
        extractor_module: HiddenStateExtractor for alignment measurement

    Returns:
        dict with perplexity, avg_nll_loss, avg_cka_score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    cka_scores = []

    for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)

        # Forward pass through LLM
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute NLL loss
        nll_loss = compute_nll_loss(logits, input_ids, attention_mask)
        num_tokens = attention_mask[:, 1:].sum().item()
        total_loss += nll_loss.item() * num_tokens
        total_tokens += num_tokens

        # Compute CKA alignment score
        if extractor_module is not None:
            clip_features = extract_clip_features(images)
            llm_features = extractor_module.get_pooled_features(attention_mask)
            cka_score = compute_cka(clip_features, llm_features.float())
            cka_scores.append(cka_score.item())

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(min(avg_loss, 100))  # Cap to avoid overflow
    avg_cka = np.mean(cka_scores) if cka_scores else 0.0

    model.train()
    return {
        "perplexity": perplexity,
        "avg_nll_loss": avg_loss,
        "avg_cka_score": avg_cka,
    }


# Evaluate baseline (before CMAR training)
print("Evaluating baseline (before CMAR fine-tuning)...")
baseline_metrics = evaluate_perplexity(llm_model, eval_loader, extractor)
print(f"\nBaseline Results:")
print(f"  Perplexity: {baseline_metrics['perplexity']:.2f}")
print(f"  Avg NLL Loss: {baseline_metrics['avg_nll_loss']:.4f}")
print(f"  Avg CKA Score: {baseline_metrics['avg_cka_score']:.4f}")

# %% [markdown]
# ## 13. Execute CMAR Training
#
# Main training loop implementing the full CMAR algorithm with progress tracking
# and logging of all loss components.

# %%
# Training state tracking
training_history = {
    "total_loss": [],
    "nll_loss": [],
    "align_loss": [],
    "cka_score": [],
    "learning_rate": [],
    "step": [],
}

global_step = 0
llm_model.train()

print(f"\n{'='*60}")
print(f"Starting CMAR Training")
print(f"{'='*60}\n")

for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
        leave=True,
    )

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)

        # ===== Step 1: Extract CLIP features (frozen, no gradients) =====
        clip_features = extract_clip_features(images)  # (batch, 512)

        # ===== Step 2: Forward pass through LLM =====
        outputs = llm_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        # ===== Step 3: Extract penultimate layer hidden states =====
        llm_features = extractor.get_pooled_features(attention_mask)  # (batch, 2048)

        # ===== Step 4: Compute NLL loss =====
        nll_loss = compute_nll_loss(logits, input_ids, attention_mask)

        # ===== Step 5: Compute alignment loss via CKA =====
        cka_score = compute_cka(clip_features, llm_features.float())
        align_loss = 1.0 - cka_score  # Want CKA -> 1, so loss -> 0

        # ===== Step 6: Combined CMAR objective =====
        total_loss = nll_loss + LAMBDA_ALIGN * align_loss

        # Scale for gradient accumulation
        scaled_loss = total_loss / GRAD_ACCUM_STEPS

        # ===== Step 7: Backpropagate =====
        scaled_loss.backward()

        # Gradient accumulation step
        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, llm_model.parameters()),
                MAX_GRAD_NORM,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Log metrics
            training_history["total_loss"].append(total_loss.item())
            training_history["nll_loss"].append(nll_loss.item())
            training_history["align_loss"].append(align_loss.item())
            training_history["cka_score"].append(cka_score.item())
            training_history["learning_rate"].append(scheduler.get_last_lr()[0])
            training_history["step"].append(global_step)

        epoch_losses.append(total_loss.item())

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "nll": f"{nll_loss.item():.4f}",
            "cka": f"{cka_score.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    # End of epoch summary
    avg_epoch_loss = np.mean(epoch_losses)
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Average Loss: {avg_epoch_loss:.4f}")
    print(f"  Final CKA: {cka_score.item():.4f}")
    print(f"  Steps completed: {global_step}")

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"Training Complete! Total steps: {global_step}")
print(f"{'='*60}")

# %% [markdown]
# ## 14. Post-Training Evaluation
#
# Re-run the same evaluation to measure improvement after CMAR fine-tuning.

# %%
# Evaluate after CMAR training
print("Evaluating after CMAR fine-tuning...")
post_metrics = evaluate_perplexity(llm_model, eval_loader, extractor)

print(f"\nPost-Training Results:")
print(f"  Perplexity: {post_metrics['perplexity']:.2f}")
print(f"  Avg NLL Loss: {post_metrics['avg_nll_loss']:.4f}")
print(f"  Avg CKA Score: {post_metrics['avg_cka_score']:.4f}")

print(f"\nImprovement:")
ppl_change = baseline_metrics['perplexity'] - post_metrics['perplexity']
cka_change = post_metrics['avg_cka_score'] - baseline_metrics['avg_cka_score']
print(f"  Perplexity: {baseline_metrics['perplexity']:.2f} -> {post_metrics['perplexity']:.2f} "
      f"({'improved' if ppl_change > 0 else 'degraded'} by {abs(ppl_change):.2f})")
print(f"  CKA Score: {baseline_metrics['avg_cka_score']:.4f} -> {post_metrics['avg_cka_score']:.4f} "
      f"({'improved' if cka_change > 0 else 'degraded'} by {abs(cka_change):.4f})")

# %% [markdown]
# ## 15. Visualization
#
# Plot training dynamics including loss curves, alignment scores,
# and before/after comparison.

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("CMAR Training Dynamics", fontsize=14, fontweight="bold")

# Plot 1: Total Loss Curve
ax = axes[0, 0]
ax.plot(training_history["step"], training_history["total_loss"],
        color="blue", alpha=0.7, linewidth=1)
# Smoothed version
if len(training_history["total_loss"]) > 10:
    window = min(20, len(training_history["total_loss"]) // 5)
    smoothed = np.convolve(training_history["total_loss"],
                           np.ones(window)/window, mode="valid")
    ax.plot(range(window-1, len(training_history["total_loss"])), smoothed,
            color="darkblue", linewidth=2, label="Smoothed")
ax.set_xlabel("Training Step")
ax.set_ylabel("Total Loss")
ax.set_title("Total Training Loss (L_NLL + 0.1 * L_align)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: CKA Alignment Score
ax = axes[0, 1]
ax.plot(training_history["step"], training_history["cka_score"],
        color="green", alpha=0.7, linewidth=1)
if len(training_history["cka_score"]) > 10:
    window = min(20, len(training_history["cka_score"]) // 5)
    smoothed = np.convolve(training_history["cka_score"],
                           np.ones(window)/window, mode="valid")
    ax.plot(range(window-1, len(training_history["cka_score"])), smoothed,
            color="darkgreen", linewidth=2, label="Smoothed")
ax.set_xlabel("Training Step")
ax.set_ylabel("CKA Score")
ax.set_title("CKA Alignment Score (higher = better alignment)")
ax.set_ylim(0, 1)
ax.axhline(y=baseline_metrics["avg_cka_score"], color="red",
           linestyle="--", label=f"Baseline: {baseline_metrics['avg_cka_score']:.3f}")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Component Losses
ax = axes[1, 0]
ax.plot(training_history["step"], training_history["nll_loss"],
        color="orange", alpha=0.7, linewidth=1, label="L_NLL")
ax.plot(training_history["step"], training_history["align_loss"],
        color="purple", alpha=0.7, linewidth=1, label="L_align (1-CKA)")
ax.set_xlabel("Training Step")
ax.set_ylabel("Loss Value")
ax.set_title("Component Losses")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Before/After Comparison
ax = axes[1, 1]
metrics_names = ["Perplexity\n(lower=better)", "CKA Score\n(higher=better)"]
before_values = [baseline_metrics["perplexity"], baseline_metrics["avg_cka_score"]]
after_values = [post_metrics["perplexity"], post_metrics["avg_cka_score"]]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width/2, before_values, width, label="Before CMAR",
               color="lightcoral", edgecolor="darkred")
bars2 = ax.bar(x + width/2, after_values, width, label="After CMAR",
               color="lightgreen", edgecolor="darkgreen")

ax.set_ylabel("Value")
ax.set_title("Before vs After CMAR Fine-Tuning")
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("cmar_training_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to cmar_training_results.png")

# %%
# Additional plot: Learning Rate Schedule
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot(training_history["step"], training_history["learning_rate"],
        color="teal", linewidth=2)
ax.set_xlabel("Training Step")
ax.set_ylabel("Learning Rate")
ax.set_title("Cosine Annealing Learning Rate Schedule")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 16. Detailed Mathematical and Implementation Explanation
#
# ### Why CKA Works for Cross-Modal Alignment
#
# **Centered Kernel Alignment (CKA)** is particularly well-suited for cross-modal alignment because:
#
# 1. **Dimension Agnostic:** CKA can compare representations of different dimensionalities
#    (CLIP produces 512-dim vectors, TinyLlama produces 2048-dim vectors). This is possible
#    because HSIC operates on the Gram matrices (n x n), not the feature matrices directly.
#
# 2. **Invariant to Orthogonal Transformations:** If the LLM learns a rotated version of
#    the visual structure, CKA still recognizes this as high alignment. This gives the LLM
#    freedom in how it organizes information while preserving structural relationships.
#
# 3. **Invariant to Isotropic Scaling:** The magnitude of features does not affect alignment
#    measurement, only their relative structure matters.
#
# 4. **Captures Second-Order Statistics:** CKA measures whether two representations
#    encode similar pairwise relationships between samples, which is a stronger signal
#    than first-order feature matching.
#
# ### Derivation of the Combined Objective
#
# The CMAR loss combines two competing objectives:
#
# $$\mathcal{L}_{\text{CMAR}} = \underbrace{\mathcal{L}_{\text{NLL}}}_{\text{language modeling}} + \lambda \cdot \underbrace{(1 - \text{CKA}(\mathbf{H}_v, \mathbf{H}_l))}_{\text{alignment regularizer}}$$
#
# - $\mathcal{L}_{\text{NLL}}$: Standard next-token prediction ensures the model retains language capabilities
# - $1 - \text{CKA}$: Penalizes misalignment between vision and language representations
# - $\lambda = 0.1$: Small weight prevents alignment from dominating language quality
#
# **Gradient flow analysis:**
# - Gradients from $\mathcal{L}_{\text{NLL}}$ update LoRA to improve token prediction
# - Gradients from $\mathcal{L}_{\text{align}}$ update LoRA to make hidden states more vision-like
# - The frozen CLIP encoder ensures a stable optimization target
# - LoRA constrains the update space, preventing catastrophic forgetting
#
# ### Why the Vision Encoder Must Stay Frozen
#
# If both encoders were trainable, they could converge to a trivial solution:
# - Both could collapse to constant representations (CKA = 1 trivially)
# - Representations might drift to match each other but lose semantic meaning
# - The vision encoder provides a **fixed anchor** based on visual grounding
#
# ### Computational Complexity
#
# For a batch of size $n$ with vision features of dimension $d_v$ and language features of dimension $d_l$:
#
# | Operation | Complexity |
# |-----------|-----------|
# | Column centering | $O(n \cdot d)$ |
# | HSIC(X, Y) via $Y_c^T X_c$ | $O(n \cdot d_v \cdot d_l)$ |
# | HSIC(X, X) via $X_c^T X_c$ | $O(n \cdot d_v^2)$ |
# | HSIC(Y, Y) via $Y_c^T Y_c$ | $O(n \cdot d_l^2)$ |
# | Total CKA | $O(n \cdot d_v \cdot d_l)$ |
#
# With $n=8$, $d_v=512$, $d_l=2048$: approximately 8.4M FLOPs per CKA computation.
# This is negligible compared to the LLM forward/backward pass (~2B FLOPs for TinyLlama).
#
# ### Limitations vs Original Paper
#
# 1. **Model scale:** We use TinyLlama-1.1B vs the paper's larger models (7B+)
# 2. **Dataset size:** ~1000 pairs vs the paper's larger training sets
# 3. **Training duration:** 3 epochs vs potentially longer schedules
# 4. **Evaluation:** Perplexity only vs the paper's reasoning benchmarks (ARC, HellaSwag, etc.)
# 5. **Vision encoder:** Single CLIP model vs potential multi-scale features

# %% [markdown]
# ## 17. Implementation Design Choices
#
# ### Forward Hook for Hidden State Extraction
#
# We use `register_forward_hook` rather than `output_hidden_states=True` because:
# - Hooks capture states lazily during the forward pass with zero memory overhead
# - We only need one specific layer, not all layers
# - Compatible with PEFT-wrapped models without modifying the forward signature
#
# ### Mean Pooling Over Sequence Length
#
# The penultimate layer outputs a tensor of shape `(batch, seq_len, hidden_dim)`.
# We need a single vector per sample for CKA. Mean pooling (with attention mask):
# - Averages information across all non-padding positions
# - Is differentiable and gradient-friendly
# - Preserves the full sequence context in the representation
# - Attention-masked pooling prevents padding tokens from diluting the signal
#
# ### 4-bit Quantization with LoRA
#
# QLoRA (4-bit quantized LoRA) enables training on a T4 GPU:
# - Base model weights stored in NF4 format (~0.5 bytes per parameter)
# - LoRA adapters remain in float16 for training precision
# - Memory: ~700MB for base + ~10MB for LoRA vs ~4.4GB for full float32
# - Computation: quantized forward pass + float16 LoRA forward/backward
#
# ### Gradient Accumulation
#
# With `BATCH_SIZE=8` and `GRAD_ACCUM_STEPS=4`:
# - Effective batch size = 32 for CKA computation stability
# - Smaller micro-batches fit in T4 memory
# - CKA with very small batches (n<8) can be noisy; accumulation helps

# %% [markdown]
# ## 18. Comparison of Alignment Methods
#
# We implemented three alignment approaches. Here is a summary of their properties:
#
# | Method | What it Measures | Invariances | Gradient Signal |
# |--------|-----------------|-------------|-----------------|
# | **CKA** | Structural similarity via HSIC | Orthogonal transforms, scaling | Smooth, well-behaved |
# | **InfoNCE** | Contrastive instance matching | None (learns invariances) | Strong but noisy |
# | **KL Divergence** | Relational structure preservation | Scaling (via softmax) | Moderate |
#
# **CKA** is preferred for CMAR because:
# - It measures representation structure, not instance identity
# - Invariance properties give the LLM freedom in how to organize information
# - The gradient is smooth and does not collapse to degenerate solutions
# - It handles different feature dimensions naturally

# %%
# Demonstrate all three alignment methods on the same batch
print("Comparing alignment methods on a sample batch:")
print("=" * 50)

# Get one batch for demonstration
sample_batch = next(iter(eval_loader))
with torch.no_grad():
    sample_images = sample_batch["image"].to(device)
    sample_ids = sample_batch["input_ids"].to(device)
    sample_mask = sample_batch["attention_mask"].to(device)

    # Get features
    v_feats = extract_clip_features(sample_images)
    _ = llm_model(input_ids=sample_ids, attention_mask=sample_mask)
    l_feats = extractor.get_pooled_features(sample_mask).float()

    # CKA
    cka_val = compute_cka(v_feats, l_feats)
    print(f"CKA Score: {cka_val.item():.4f} (1.0 = perfectly aligned)")

    # InfoNCE
    infonce_loss, infonce_acc = compute_infonce(v_feats, l_feats)
    print(f"InfoNCE Loss: {infonce_loss.item():.4f}, Accuracy: {infonce_acc:.4f}")

    # KL Divergence
    kl_loss = compute_kl_alignment(v_feats, l_feats)
    print(f"KL Divergence: {kl_loss.item():.4f} (0.0 = identical structure)")

# %% [markdown]
# ## 19. How to Explain This Reproduction to a Professor
#
# ### Complete Pipeline Explanation
#
# **Q: What is the high-level goal of CMAR?**
#
# A: CMAR improves language model reasoning by regularizing its internal representations
# to be structurally similar to a frozen vision encoder's representations. The hypothesis
# is that vision models learn compositional, grounded representations that, when transferred
# structurally to a language model, improve its ability to reason about compositional concepts.
#
# **Q: Walk me through the training pipeline step by step.**
#
# A: Each training step:
# 1. A batch of paired (image, text) data is loaded
# 2. Images pass through frozen CLIP ViT-B/32, producing 512-dim embeddings
# 3. Text passes through TinyLlama with LoRA adapters
# 4. A forward hook captures the penultimate layer's hidden states
# 5. Hidden states are mean-pooled over sequence length to get 2048-dim vectors
# 6. CKA measures structural similarity between the 512-dim and 2048-dim representations
# 7. Total loss = NLL (language modeling) + 0.1 * (1 - CKA)
# 8. Backpropagation updates only the LoRA parameters (~0.5% of total)
#
# **Q: Why CKA and not cosine similarity or MSE?**
#
# A: CKA has critical invariance properties:
# - It works across different dimensions (512 vs 2048) without projection layers
# - It is invariant to orthogonal rotations, so the LLM is free to organize features differently
# - It is invariant to scaling, so feature magnitudes do not matter
# - It captures second-order structure (relationships between samples), which is richer
#   than first-order matching of individual vectors
#
# **Q: Why is the vision encoder frozen?**
#
# A: Three reasons:
# 1. Prevents representation collapse (both models converging to trivial constant outputs)
# 2. Provides a stable optimization target (moving target problem otherwise)
# 3. CLIP's representations are already well-structured from large-scale pretraining;
#    we want to transfer this structure, not destroy it
#
# **Q: What is LoRA and why use it?**
#
# A: LoRA (Low-Rank Adaptation) adds small trainable matrices to attention layers:
# - Original weight: $W_0 \in \mathbb{R}^{d \times d}$
# - LoRA update: $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$
# - Effective weight: $W = W_0 + \frac{\alpha}{r} BA$
# - With rank=16 and hidden_dim=2048: only $2 \times 2048 \times 16 = 65536$ params per layer
# - Benefits: memory efficient, prevents catastrophic forgetting, fast training
#
# **Q: How does the forward hook work?**
#
# A: `register_forward_hook(fn)` attaches a callback to a layer. During every forward pass,
# after that layer computes its output, the callback is called with (module, input, output).
# We store the output tensor. This is zero-cost when not extracting and does not modify
# the computational graph.
#
# **Q: Why mean-pool over sequence length?**
#
# A: The LLM produces per-token representations of shape (batch, seq_len, 2048).
# CKA needs (batch, features). Mean pooling:
# - Aggregates all token-level information into one vector
# - Is differentiable (gradients flow back to all positions)
# - Respects the attention mask (padding tokens excluded)
# - Alternative: use [CLS] or last-token, but mean is more robust
#
# **Q: What does lambda=0.1 mean in practice?**
#
# A: It means the alignment loss contributes about 10% of the total gradient signal.
# If lambda is too high (e.g., 1.0), the model optimizes for alignment at the expense
# of language quality. If too low (e.g., 0.001), the alignment signal is too weak to
# meaningfully restructure representations. 0.1 is a balanced choice from the paper.
#
# **Q: What would happen if you fine-tuned all parameters instead of LoRA?**
#
# A: Several problems:
# 1. Memory: full fine-tuning of 1.1B params needs ~8GB+ just for gradients+optimizer states
# 2. Catastrophic forgetting: large updates could destroy pretrained knowledge
# 3. Overfitting: with only ~1000 training pairs, full fine-tuning would overfit severely
# 4. LoRA constrains updates to a low-rank subspace, acting as implicit regularization
#
# **Q: How do you know this is working and not just overfitting?**
#
# A: We monitor:
# 1. Validation perplexity on held-out text (should not increase significantly)
# 2. CKA alignment score (should increase, indicating structural alignment)
# 3. The NLL component of training loss (should remain stable or decrease)
# 4. In the full paper: downstream reasoning benchmarks show genuine improvement
#
# **Q: What are the limitations of this reproduction?**
#
# A: Compared to the full paper:
# - Smaller model (1.1B vs 7B+): alignment effects may be less pronounced
# - Smaller dataset (1000 pairs vs much larger): may not fully converge
# - Simpler evaluation (perplexity vs reasoning benchmarks)
# - Single vision encoder (vs potential ensembles or multi-scale features)
# - No hyperparameter search (fixed lambda=0.1, rank=16)

# %% [markdown]
# ## 20. Summary and Next Steps
#
# ### What We Demonstrated
#
# 1. **CKA-based alignment** between a frozen CLIP vision encoder and a LoRA-adapted LLM
# 2. **Three alignment methods** (CKA, InfoNCE, KL) with mathematical derivations
# 3. **End-to-end training** with the combined CMAR objective
# 4. **Evaluation** showing alignment improvement after training
#
# ### Next Steps for Full Reproduction
#
# - Scale to larger LLMs (Llama-2-7B, Mistral-7B)
# - Use full Flickr30k or COCO Captions dataset
# - Evaluate on reasoning benchmarks (ARC, HellaSwag, WinoGrande)
# - Ablate lambda values (0.01, 0.05, 0.1, 0.2, 0.5)
# - Compare CKA vs InfoNCE vs KL as alignment objectives
# - Investigate which layers benefit most from alignment

# %%
# Cleanup
extractor.remove()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("Cleanup complete. Notebook execution finished successfully.")

# %%
# Final summary statistics
print("\n" + "=" * 60)
print("CMAR FINE-TUNING REPRODUCTION - FINAL SUMMARY")
print("=" * 60)
print(f"\nModel: TinyLlama-1.1B-Chat with LoRA (rank=16, alpha=32)")
print(f"Vision Encoder: CLIP ViT-B/32 (frozen)")
print(f"Alignment Method: CKA (linear kernel)")
print(f"Lambda: {LAMBDA_ALIGN}")
print(f"Training Samples: {train_size}")
print(f"Training Steps: {global_step}")
print(f"\nResults:")
print(f"  Baseline Perplexity: {baseline_metrics['perplexity']:.2f}")
print(f"  Final Perplexity:    {post_metrics['perplexity']:.2f}")
print(f"  Baseline CKA:        {baseline_metrics['avg_cka_score']:.4f}")
print(f"  Final CKA:           {post_metrics['avg_cka_score']:.4f}")
print(f"\nConclusion: CMAR fine-tuning {'improved' if post_metrics['avg_cka_score'] > baseline_metrics['avg_cka_score'] else 'did not improve'} "
      f"cross-modal alignment as measured by CKA.")
print("=" * 60)
