---
title: Resume
---

| Name        | Email                  | GitHub                                            | Personal Website                                       |
| ----------- | ---------------------- | ------------------------------------------------- | ------------------------------------------------------ |
| Qianhe Xiao | lapis_xiao@outlook.com | [BUPTAntiRiver](https://github.com/BUPTAntiRiver) | [Qianhe Xiao's Blog](https://buptantiriver.github.io/) |

# Education

## Undergraduate

Beijing University of Posts and Telecommunications

Major: Artificial Intelligence | 2023 - Present

GPA: 3.53 / 4.00 | Top 3% (Year 1)

CET4: 596 | CET6: 635

## Selected Advanced Coursework & Training

### Machine Learning

- Stanford CS336 Large Language Model from Scratch: Implemented transformer training, scaling laws, data filtering, RLHF, 5 Assignment Project repos: [a1](https://github.com/BUPTAntiRiver/assignment1-basics.git), [a2](https://github.com/BUPTAntiRiver/assignment2-systems.git), [a3](https://github.com/BUPTAntiRiver/assignment3-scaling.git), [a4](https://github.com/BUPTAntiRiver/assignment4-data.git), [a5](https://github.com/BUPTAntiRiver/assignment5-alignment.git)
- CMU Deep Learning Systems: Implemented Autodiff engine, NDArray, Lab Project saving: [needle](https://drive.google.com/drive/folders/1iAh2aKaB1Q3xCzC3cuPvNTI3uU21Dnpi?usp=drive_link)
- UC Berkeley CS 285 Deep Reinforcement Learning, [notes](https://buptantiriver.github.io/Machine-Learning/RL/CS-285-Reinforcement-Learning/)
- MIT 6.S184 Flow Matching and Diffusion Models Introduction, [notes](https://buptantiriver.github.io/Machine-Learning/MIT-6.S184/)
- MIT 6.S978 Deep Generative Models
- CS224n NLP Introduction
- MIT TinyML: modern techniques on large models, distributed training, distillation, pruning, quantization

### Computer Science

- CS144 Computer Network, Lab [repo](https://github.com/BUPTAntiRiver/minnow.git)
- CSAPP Computer Systems A Programmer Perspective, Lab [repo](https://github.com/BUPTAntiRiver/CSAPP.git), [notes](https://buptantiriver.github.io/Computer-Science/CSAPP/)

# Skills

- Programming: Mainly Python, C++, know about bash, AscendC, CUDA, Java
- Tools: Linux, Git, Docker, nvim

# Projects

## AI Infra Intern - Huawei CANN TileLang-Ascend

GitHub Repo: [tile-ai/tilelang-ascend](https://github.com/tile-ai/tilelang-ascend.git)

- Implemented examples with TileLang-Ascend including online softmax, tailed GEMM, layer norm, RMS norm
- Enhanced `T.Parallel` feature to support discrete and complex nested parallel patterns, enabling auto copy to GM, frequently used in other operators
- Fixed multiple compiler and usability bugs; contributed patches merged into the main repository
- Certificated: Ascend C Operator Development (Intermediate)

## CMU Course Project - Deep Learning Systems

- Implemented a minimal deep learning framework (autograd, tensor ops, NDArray on cuda backend)
- Built CNN, Transformer model with the framework

## Stanford Course Project - Large Language Model from Scratch

Full procedure to build LLM from scratch:

- Implemented BPE tokenizer from scratch, optimized with cache and multi-process
- Implemented Transformer and utils from scratch, Attention layer, RoPE, etc
- Profiled with Nsight systems
- Scaled to find best model size and number of training tokens under given budget like [[Chinchilla]]
- Filtered and preprocessed training data from Common Crawl
- Post trained on Qwen2.5-Math-1.5B with SFT, Expert Iteration and RL
