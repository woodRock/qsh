# 🐚 qsh: Qwen Shell

[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![Model](https://img.shields.io/badge/Model-Qwen--3.5--0.8B-blue.svg)](https://huggingface.co/Qwen/Qwen3.5-0.8B)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Built%20with-Candle-lightgrey.svg)](https://github.com/huggingface/candle)

**AI-powered Coreutils for the modern terminal.**

`qsh` is a high-performance, multimodal terminal assistant that brings the power of **Qwen 3.5-0.8B** directly to your command line. Built in Rust for speed and safety, it acts as a suite of "smart pipes" for your existing Unix workflows.

---

## 🚀 Quick Install (macOS / Linux)

Ensure you have [Rust](https://rustup.rs/) installed, then run:

```bash
git clone https://github.com/yourusername/qsh.git && cd qsh/qsh && cargo install --path . && qsh setup
```

---

## ✨ Features

- **English to Bash:** Natural language command generation with instant execution and line-by-line explanation.
- **Semantic Text Filter:** `grep` on steroids. Filter lines based on meaning rather than just regex patterns.
- **Multimodal Vision Filter:** Use visual reasoning to filter files. Analyze a stream of image paths with natural language queries.
- **Hardware Accelerated:** Optimized for **Metal** (macOS) and **CUDA** (Linux) via the Candle inference framework.
- **Hybrid Architecture:** Custom implementation of **Gated DeltaNet** and **Gated Attention** for efficient local inference.

---

## 🛠️ Usage

### 1. Interactive Command Assistant
Convert your intent to valid Bash commands instantly.
```bash
qsh "Find all files larger than 100MB and list them by date"
```
> `[E]xecute? [e]xplain? [A]bort?`

### 2. Semantic Pipe (`filter`)
Filter text data based on high-level concepts.
```bash
cat server.log | qsh filter "unusual security activity"
```

### 3. Vision Pipe (`vision`)
Filter image paths using visual intelligence.
```bash
ls screenshots/*.png | qsh vision "is there code visible in this image?"
```

---

## 🏗️ Technical Implementation

`qsh` implements the full **Qwen 3.5-0.8B** architecture from scratch in Rust:
- **Unified Vision-Language Foundation:** Early-fusion multimodal processing.
- **Mamba-2 / SSD Evolution:** Implementation of Gated DeltaNet recurrent layers.
- **RoPE & QKNorm:** Accurate positional embeddings and head normalization for stable output.
- **Greedy Decoding:** Optimized for reliability in small-parameter models.

## 📦 Setup from Source

1. **Clone & Build:**
   ```bash
   git clone https://github.com/yourusername/qsh.git
   cd qsh/qsh
   cargo build --release
   ```

2. **Initialize Weights:**
   ```bash
   ./target/release/qsh setup
   ```

---

Built with ❤️ and 🦀 by the community. Powered by [Qwen](https://github.com/QwenLM/Qwen).
