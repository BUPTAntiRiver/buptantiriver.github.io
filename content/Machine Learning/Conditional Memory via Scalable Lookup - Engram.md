Paper Link: http://arxiv.org/abs/2601.07372

# Introduction

Modern large language models scales capacity via conditional computations like [[MoE-Mixture-of-Experts]], which increases model size without proportional increases in compute.

Despite the success of this conditional computation paradigm, the intrinsic heterogeneity of linguistic signals suggests significant room for _structural optimization_. Specifically, language modeling entails two main different sub-tasks: **compositional reasoning** and **knowledge retrieval**.

The former task needs deep dynamic computation and text about name entities and formulaic patterns, which is local, static and highly stereotyped. Such information can be represented as computationally inexpensive lookups with classical $N$-gram models. However, current LLMs are forced to **simulate retrieval through computation** (query, key, value, [[Transformer and LLM]]).

To align model with this linguistic duality, we advocate for a complementary axis of sparsity: **conditional memory**. Just like conditional computation sparsely activates parameters to process dynamic logic, conditional memory relies on sparse lookup operations to retrieve static embeddings for fixed knowledge. They revisited $N$-gram embeddings and propose a novel module **Engram**, a conditional memory module grounded in the classic $N$-gram structure but equipped with **modern adaptations** such as tokenizer compression, multi-head hashing, contextualized gating, and multi-branch integration.
