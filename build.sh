#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
python -m pip install --upgrade pip

# Install requirements with pre-built wheels
pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Create necessary directories and copy files
mkdir -p build
mkdir -p build/uploads  # Create uploads directory
mkdir -p build/temp_uploads  # Create temp uploads directory
mkdir -p build/static  # Create static directory
cp -r api build/
cp -r templates build/
cp -r static build/ 2>/dev/null || echo "No static files to copy"
cp -r predict.py build/
cp -r model.pth build/
cp requirements.txt build/

# Set proper permissions
chmod -R 755 build/
chmod -R 777 build/uploads
chmod -R 777 build/temp_uploads 