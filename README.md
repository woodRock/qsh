# 🐚 qsh: Qwen Shell

[![Rust CI](https://github.com/woodRock/qsh/actions/workflows/rust.yml/badge.svg)](https://github.com/woodRock/qsh/actions/workflows/rust.yml)
[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/language-Python-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Qwen--3.5--0.8B-blue.svg)](https://huggingface.co/Qwen/Qwen3.5-0.8B)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**AI-powered Coreutils for the modern terminal.**

`qsh` is a high-performance terminal assistant that brings the power of **Qwen 3.5-0.8B** directly to your command line. It uses a hybrid architecture: a lightning-fast **Rust CLI** wrapper around a high-performance **Python inference engine** (PyTorch/MPS), providing the best of both worlds: speed, safety, and cutting-edge multimodal capabilities.

---

## 🚀 Quick Install (macOS / Linux)

Ensure you have [Rust](https://rustup.rs/) and **Python 3.10+** installed, then run:

```bash
curl -sSL https://raw.githubusercontent.com/woodRock/qsh/main/setup.sh | sh
```

---

## ✨ Features

- **English to Bash:** Natural language command generation with instant execution and line-by-line explanation.
- **Semantic Text Filter:** `grep` on steroids. Filter lines based on meaning rather than just regex patterns.
- **Multimodal Vision Filter:** Use visual reasoning to filter files. Analyze a stream of image paths with natural language queries.
- **Hardware Accelerated:** Automatically leverages **MPS** (Apple Silicon) or **CUDA** (NVIDIA) via PyTorch for ultra-fast inference.
- **Hybrid Architecture:** Optimized Rust binary for the CLI interface and a dedicated Python virtual environment for heavy-lifting inference.

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

`qsh` utilizes a robust hybrid approach:
- **Rust Frontend:** Handles the CLI arguments, pipes, interactive loops, and process management.
- **Python Backend:** A dedicated `qenv` virtual environment running the **Qwen 3.5-0.8B** model.
- **Efficient Bridge:** High-speed communication via JSON-RPC over local pipes.
- **Multimodal Foundation:** Full support for Qwen 3.5's vision-language features, including interleaved text and image processing.
- **Smart Resizing:** Automatically manages image dimensions to optimize memory usage on hardware accelerators like MPS.

## 📦 Setup from Source

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/woodRock/qsh.git
   cd qsh
   ```

2. **Run Setup Script:**
   ```bash
   ./setup.sh
   ```

---

Built with ❤️ by the community. Powered by [Qwen](https://github.com/QwenLM/Qwen).
