#!/bin/bash
set -e

echo "--- 🐚 qsh Installation ---"

# 1. Check for Rust/Cargo
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust/Cargo not found. Please install it first: https://rustup.rs"
    exit 1
fi

# 2. Check for Python 3.10+, cmake (for llamacpp)
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

if ! command -v cmake &> /dev/null; then
    echo "Warning: 'cmake' not found. It will be required if you choose the LlamaCpp engine."
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
echo "Installing Python dependencies (transformers, torch, peft, datasets, etc.)..."
pip install --upgrade pip --quiet
pip install --quiet git+https://github.com/huggingface/transformers.git qwen-vl-utils torch torchvision accelerate pillow peft datasets

# 7. Setup Wizard (Engine & Model)
echo ""
echo "--- 🧙 qsh Setup Wizard ---"
echo "Select your preferred inference engine:"
echo "1) Python (Transformers) - Default, easiest to setup."
echo "2) Rust (Candle) - Fast, lightweight, but experimental."
echo "3) LlamaCpp (TurboQuant) - Highest performance on Apple Silicon (M2+), requires manual build of the turboquant_plus fork."
read -p "Engine [1-3, default: 1]: " ENGINE_CHOICE
ENGINE_CHOICE=${ENGINE_CHOICE:-1}

case $ENGINE_CHOICE in
    1) ENGINE="python" ;;
    2) ENGINE="rust" ;;
    3) ENGINE="llamacpp" ;;
    *) ENGINE="python" ;;
esac

echo ""
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

# Create initial config
CONFIG_DIR="$HOME/Library/Application Support/com.qwen.qsh"
mkdir -p "$CONFIG_DIR"
CONFIG_FILE="$CONFIG_DIR/config.toml"

cat << EOF > "$CONFIG_FILE"
default_engine = "$ENGINE"
default_model = "$MODEL"
safety_check = true

[llama_cpp]
server_url = "http://localhost:8080"
turbo_k = "q8_0"
turbo_v = "turbo4"
flash_attn = true
EOF

if [ "$ENGINE" == "llamacpp" ]; then
    if ! command -v cmake &> /dev/null; then
        echo "Error: 'cmake' is required to build the LlamaCpp engine. Please install it (e.g., 'brew install cmake') and run setup again."
        exit 1
    fi

    echo ""
    echo "🏗️  Setting up TurboQuant+ (llama.cpp fork)..."
    TURBO_DIR="$INSTALL_ROOT/turboquant_plus"
    if [ ! -d "$TURBO_DIR" ]; then
        git clone https://github.com/TheTom/turboquant_plus "$TURBO_DIR"
    fi
    
    cd "$TURBO_DIR"
    mkdir -p build
    cd build
    echo "Building llama-server with Metal support (this may take a few minutes)..."
    cmake .. -DGGML_METAL=ON
    cmake --build . --config Release --target llama-server -j$(sysctl -n hw.ncpu)

    SERVER_BIN="$TURBO_DIR/build/bin/llama-server"
    
    # 7.2 Download GGUF Model
    MODEL_FILENAME=$(echo "$MODEL" | sed 's/.*\///')-Q4_K_M.gguf
    # Note: Unsloth uses a different repo naming convention
    REPO_NAME="unsloth/$(echo "$MODEL" | sed 's/.*\///')-GGUF"
    MODEL_URL="https://huggingface.co/$REPO_NAME/resolve/main/$MODEL_FILENAME"
    MODEL_DEST="$INSTALL_ROOT/models/$MODEL_FILENAME"
    
    mkdir -p "$INSTALL_ROOT/models"
    
    if [ ! -f "$MODEL_DEST" ]; then
        echo ""
        echo "📥 Downloading $MODEL_FILENAME (~$(case $MODEL_CHOICE in 1) echo "0.5GB" ;; 2) echo "5.5GB" ;; 3) echo "16GB" ;; esac))..."
        curl -L "$MODEL_URL" -o "$MODEL_DEST"
    else
        echo ""
        echo "✅ Model $MODEL_FILENAME already exists at $MODEL_DEST"
    fi

    # Update config with binary and model path
    cat << EOF > "$CONFIG_FILE"
default_engine = "$ENGINE"
default_model = "$MODEL"
safety_check = true

[llama_cpp]
server_url = "http://localhost:8080"
server_binary = "$SERVER_BIN"
model_path = "$MODEL_DEST"
turbo_k = "q8_0"
turbo_v = "turbo4"
flash_attn = true
EOF

    echo ""
    echo "✅ TurboQuant+ built successfully at $SERVER_BIN"
    echo "✅ Model downloaded and configured at $MODEL_DEST"
fi

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
