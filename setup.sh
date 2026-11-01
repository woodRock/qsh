#!/bin/bash
set -e

echo "--- 🐚 qsh Installation ---"

# 1. Check for Rust/Cargo
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust/Cargo not found. Please install it first: https://rustup.rs"
    exit 1
fi

# 2. Check for Python 3.10+
PYTHON_CMD=""
for cmd in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v $cmd &> /dev/null; then
        VERSION=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [ "$(echo "$VERSION >= 3.10" | bc -l)" -eq 1 ]; then
            PYTHON_CMD=$(command -v $cmd)
            echo "Found compatible Python: $PYTHON_CMD (version $VERSION)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.10 or higher is required."
    exit 1
fi

# 3. Setup Wizard (Model)
echo ""
echo "--- Setup Wizard ---"
echo "Select your preferred Qwen 3.5 model size:"
echo "1) 0.8B - ~0.5GB disk, ~1GB RAM. Fast, for basic shell commands. Fits any M-series Mac."
echo "2) 9B   - ~5.5GB disk, ~8GB RAM (16GB recommended). Balanced for most tasks."
echo "3) 27B  - ~16GB disk, ~20GB RAM (32GB+ recommended). Expert reasoning, complex scripting."
read -p "Model [1-3, default: 1]: " MODEL_CHOICE
MODEL_CHOICE=${MODEL_CHOICE:-1}

case $MODEL_CHOICE in
    1) MODEL="Qwen/Qwen3.5-0.8B" ;;
    2) MODEL="Qwen/Qwen3.5-9B" ;;
    3) MODEL="Qwen/Qwen3.5-27B" ;;
    *) MODEL="Qwen/Qwen3.5-0.8B" ;;
esac

echo ""
echo "Model: $MODEL"
echo "Starting installation..."
echo ""

# 4. Determine Installation Directory
# We install everything into ~/.local/share/qsh to keep it self-contained
INSTALL_ROOT="$HOME/.local/share/qsh"
BIN_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_ROOT/src"
mkdir -p "$BIN_DIR"

# 5. Clone and Build
TEMP_DIR=$(mktemp -d)
echo "Cloning repository..."
git clone https://github.com/woodRock/qsh.git "$TEMP_DIR"
cd "$TEMP_DIR/qsh"

echo "Building Rust CLI (this may take a minute)..."
cargo build --release --quiet

# 6. Install Components
echo "Installing components to $INSTALL_ROOT..."
cp target/release/qsh "$INSTALL_ROOT/qsh-bin"
cp src/inference.py "$INSTALL_ROOT/src/inference.py"

# Create a wrapper script in ~/.local/bin/qsh
cat << EOF > "$BIN_DIR/qsh"
#!/bin/bash
exec "$INSTALL_ROOT/qsh-bin" "\$@"
EOF
chmod +x "$BIN_DIR/qsh"

# 7. Setup Python Virtual Environment
echo "Setting up Python environment (qenv)..."
cd "$INSTALL_ROOT"
$PYTHON_CMD -m venv qenv
source qenv/bin/activate
echo "Installing Python dependencies (transformers, torch, peft, datasets, etc.)..."
pip install --upgrade pip --quiet
pip install --quiet git+https://github.com/huggingface/transformers.git qwen-vl-utils torch torchvision accelerate pillow peft datasets

# Create initial config
CONFIG_DIR="$HOME/Library/Application Support/com.qwen.qsh"
mkdir -p "$CONFIG_DIR"
CONFIG_FILE="$CONFIG_DIR/config.toml"

cat << EOF > "$CONFIG_FILE"
default_model = "$MODEL"
safety_check = true
EOF

# Cleanup
rm -rf "$TEMP_DIR"

echo "----------------------------------------"
echo "✅ Installation successful!"
echo ""
echo "qsh is installed at $BIN_DIR/qsh"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo "⚠️ Warning: $BIN_DIR is not in your PATH."
    echo "Add this to your shell profile (e.g., .bashrc or .zshrc):"
    echo "  export PATH=\"\$PATH:$BIN_DIR\""
fi
echo ""
echo "Try it out:"
echo "  qsh 'list files by size'"
echo "  ls *.jpg | qsh vision 'is there a cat?'"
