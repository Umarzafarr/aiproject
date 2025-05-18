#!/usr/bin/env bash
# exit on error
set -o errexit

python -m pip install --upgrade pip
pip install gunicorn
pip install -r requirements.txt

# Create build directory if it doesn't exist
mkdir -p build
cp -r api build/
cp -r templates build/
cp -r predict.py build/
cp requirements.txt build/ 