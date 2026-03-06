# QSH Examples & Showcase

Welcome to the `qsh` examples folder! These files are here to help you test the capabilities of the Qwen Shell ("Smart Pipe") architecture.

## Getting Started

First, ensure you have set up the model weights:

```bash
qsh setup
```

---

## 1. The Commander Mode (English → Bash)

Ask `qsh` to perform complex tasks using standard Unix tools.

```bash
# Example: Find all files modified in the last 24 hours
qsh "find all files modified in the last 24 hours"

# Example: Count total lines of all Rust files in this project
qsh "count the total lines in all .rs files in this folder recursively"
```

---

## 2. The Semantic Filter (`qsh filter`)

Filter text based on meaning, not just exact keywords.

```bash
# Filter notes that mention a specific time or deadline
cat examples/notes.txt | qsh filter "is this about a deadline or due date?"
```

---

## 3. The Vision Filter (`qsh vision`)

Process and filter images based on what's *inside* them.

### A. Sorting by Content
```bash
# Find only the cat in the examples folder
ls examples/*.jpg | qsh vision "is there a cat in this photo?"
```

### B. Finding Code Screenshots
```bash
# Identify if an image is a screenshot of software code
ls examples/*.png | qsh vision "is this a screenshot of code?"
```

### C. Filtering Blurry Photos
```bash
# Find photos that are out of focus or blurry
ls examples/*.jpg | qsh vision "is this photo blurry or out of focus?"
```

---

## Combined "God Mode" (Coming Soon)

In the future, you will be able to chain these seamlessly:

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
