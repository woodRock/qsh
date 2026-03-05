#!/bin/bash
set -e

echo "--- qsh Installation ---"

# Check for Cargo
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust/Cargo not found. Please install it first: https://rustup.rs"
    exit 1
fi

# Create a temporary directory for the build
TEMP_DIR=$(mktemp -d)
echo "Cloning repository..."
git clone https://github.com/woodRock/qsh.git "$TEMP_DIR"
cd "$TEMP_DIR/qsh"

echo "Building qsh in release mode (this may take a minute)..."
cargo build --release --quiet

# Determine installation path
if [ -w /usr/local/bin ]; then
    INSTALL_DIR="/usr/local/bin"
else
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
fi

echo "Installing binary to $INSTALL_DIR/qsh..."
cp target/release/qsh "$INSTALL_DIR/qsh"

# Cleanup
rm -rf "$TEMP_DIR"

echo "Running initial setup to download model weights..."
"$INSTALL_DIR/qsh" setup

echo "----------------------------------------"
echo "Installation successful!"
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo "Warning: $INSTALL_DIR is not in your PATH."
    echo "Add this to your shell profile (e.g., .bashrc or .zshrc):"
    echo "export PATH=\"\$PATH:$INSTALL_DIR\""
fi
echo "You can now run qsh by typing: qsh 'your prompt'"
