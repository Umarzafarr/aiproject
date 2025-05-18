#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip and install build dependencies
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel

# Install gunicorn first
pip install gunicorn

# Install other requirements
pip install -r requirements.txt

# Create build directory if it doesn't exist
mkdir -p build
cp -r api build/
cp -r templates build/
cp -r predict.py build/
cp requirements.txt build/ 