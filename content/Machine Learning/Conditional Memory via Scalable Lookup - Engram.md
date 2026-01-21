Paper Link: http://arxiv.org/abs/2601.07372

# Introduction

Modern large language models scales capacity via conditional computations like [[MoE-Mixture-of-Experts]], which increases model size without proportional increases in compute.

Despite the success of this conditional computation paradigm, the intrinsic heterogeneity of linguistic signals suggests significant room for _structural optimization_. Specifically, language modeling entails two main different sub-tasks: **compositional reasoning** and **knowledge retrieval**.

The former task needs deep dynamic computation and text about name entities and formulaic patterns, which is local, static and highly stereotyped. Such information can be represented as computationally inexpensive lookups with classical $N$-gram models. However, current LLMs are forced to **simulate retrieval through computation** (query, key, value, [[Transformer and LLM]]).

To align model with this linguistic duality, we advocate for a complementary axis of sparsity: **conditional memory**. Just like conditional computation sparsely activates parameters to process dynamic logic, conditional memory relies on sparse lookup operations to retrieve static embeddings for fixed knowledge. They revisited $N$-gram embeddings and propose a novel module **Engram**, a conditional memory module grounded in the classic $N$-gram structure but equipped with **modern adaptations** such as tokenizer compression, multi-head hashing, contextualized gating, and multi-branch integration.

# Architecture

## Overview

Engram is a conditional memory module designed to separate static pattern storage from dynamic computation in order to enhance Transformer backbone.

## Sparse Retrieval via Hashed $N$-grams

The first phase maps local context to entries, then retrieves embeddings.

### Tokenizer Compression

Modern tokenizers has different id for tokens like: `apple` and `␣apple`, in order to maximize semantic density, we project them to same id based on normalized textual equivalence (using NFKC, lowercasing, etc.). For raw id $x_{t}$ we have $x'_{t}$ then have $N$-gram $g_{t,n}=(x'_{t-n+1},\dots,x'_{t})$.

### Multi-Head Hashing

It is intractable to parametrize all possible $N$-grams. We adopt a hashing-based approach. We have $K$ distinct hash functions for each $N$-gram order $n$. Each head $k$ maps the compressed context to an index within an embedding table $\mathbf{E}_{n,k}$ of prime size $M_{n,k}$ via a deterministic function $\varphi_{n,k}$:

$$
z_{t,n,k}\triangleq\varphi_{n,k}(g_{t,n}),\quad \mathbf{e}_{t,n,k}=\mathbf{E}_{n,k}[z_{t,n,k}]
$$

In practice, $\varphi$ is implemented as a lightweight multiplicative-XOR hash. We construct the final memory vector $\mathbf{e}_{t}\in \mathbb{R}^{d_{\text{mem}}}$ by concatenating all retrieved embeddings.

## Context Aware Gating

The raw embedding we get knows nothing about context. To enhance expressivity, we apply an Attention-like gating mechanism. We use current hidden state $\mathbf{h}_{t}$ as query, and project embedding to get key and value:

$$
\mathbf{k}_{t}=\mathbf{W}_{K}\mathbf{e}_{t},\quad \mathbf{v}_{t}=\mathbf{W}_{V}\mathbf{e}_{t}
$$

then apply pre-[[RMSNorm]] and compute to get the scalar gate $\alpha_{t}\in(0,1)$:

$$
\alpha_{t}=\sigma\left(  \frac{\text{RMSNorm}(\mathbf{h}_{t})^{\top}\text{RMSNorm}(\mathbf{k}_{t})}{\sqrt{ d }}  \right)
$$

The gated output is defined as $\tilde{\mathbf{v}}_{t}=\alpha_{t}\cdot \mathbf{v}_{t}$.

Finally, to expand the receptive field and enhance the model's non-linearity, we introduce a convolution with kernel size $w$ of 4, dilation $\delta$ of the max $N$-gram order and SiLU activation for sequence of gated values $\tilde{\mathbf{V}}\in \mathbb{R}^{T\times d}$:

$$
Y=\text{SiLU}(\text{Conv1D}(\text{RMSNorm}(\tilde{\mathbf{V}})))+\tilde{\mathbf{V}}
$$

Then Engram is integrated into the backbone via a residual connection: $\mathbf{H}^{(l)}=\mathbf{H}^{(l)}+\mathbf{Y}$, then followed by standard Attention and MoE. Engram is not applied to every layer; its specific placement is governed by the system-level latency constraints detailed in later sections.

## Integration with Multi-branch Architecture
