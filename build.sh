#!/usr/bin/env bash

set -e  # Exit on first error

# Install system dependencies if needed
echo "Installing system dependencies..."

# Update package list
apt-get update || true

# Install basic build tools
apt-get install -y build-essential || true

echo "Build script completed successfully"
