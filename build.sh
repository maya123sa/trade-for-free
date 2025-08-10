#!/usr/bin/env bash

set -e  # Exit on first error

# Install system dependencies if needed
echo "Installing system dependencies..."

# Update package list
apt-get update || true

# Install basic build tools
apt-get install -y build-essential || true

# Install Python and pip
apt-get install -y python3 python3-pip || true

echo "Build script completed successfully"
