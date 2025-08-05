#!/usr/bin/env bash

set -e  # Exit on first error

# Download TA-Lib source code
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

# Extract it
tar -xvzf ta-lib-0.4.0-src.tar.gz

# Build and install
cd ta-lib
./configure --prefix=/usr
make
make install

# Go back to root for pip install
cd ..
