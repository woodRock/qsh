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

# 3. Determine Installation Directory
# We install everything into ~/.local/share/qsh to keep it self-contained
INSTALL_ROOT="$HOME/.local/share/qsh"
BIN_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_ROOT/src"
mkdir -p "$BIN_DIR"

# 4. Clone and Build
TEMP_DIR=$(mktemp -d)
echo "Cloning repository..."
git clone https://github.com/woodRock/qsh.git "$TEMP_DIR"
cd "$TEMP_DIR/qsh"

echo "Building Rust CLI (this may take a minute)..."
cargo build --release --quiet

# 5. Install Components
echo "Installing components to $INSTALL_ROOT..."
cp target/release/qsh "$INSTALL_ROOT/qsh-bin"
cp src/inference.py "$INSTALL_ROOT/src/inference.py"

# Create a wrapper script in ~/.local/bin/qsh
cat << EOF > "$BIN_DIR/qsh"
#!/bin/bash
exec "$INSTALL_ROOT/qsh-bin" "\$@"
EOF
chmod +x "$BIN_DIR/qsh"

# 6. Setup Python Virtual Environment
echo "Setting up Python environment (qenv)..."
cd "$INSTALL_ROOT"
$PYTHON_CMD -m venv qenv
source qenv/bin/activate
echo "Installing Python dependencies (transformers, torch, etc.)..."
pip install --upgrade pip --quiet
pip install --quiet git+https://github.com/huggingface/transformers.git qwen-vl-utils torch torchvision accelerate pillow

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
