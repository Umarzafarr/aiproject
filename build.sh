#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories and copy files
mkdir -p build
cp -r api build/
cp -r templates build/
cp -r predict.py build/
cp -r model.pth build/ # Ensure model file is copied
cp requirements.txt build/ 