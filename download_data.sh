#!/bin/bash
# Download datasets from Zenodo
# This script should be run from /workspace/data directory

set -e

echo "Downloading datasets from Zenodo (https://doi.org/10.5281/zenodo.17840656)..."
echo "Total download size: ~100GB"
echo "This may take 30-60 minutes depending on your internet connection."
echo ""

# Download all datasets
echo "[1/5] Downloading full_list.zip..."
wget https://zenodo.org/records/17840656/files/full_list.zip

echo "[2/5] Downloading reduced_list.zip..."
wget https://zenodo.org/records/17840656/files/reduced_list.zip

echo "[3/5] Downloading traffic_captures.zip..."
wget https://zenodo.org/records/17840656/files/traffic_captures.zip

echo "[4/5] Downloading train_test_WF.zip..."
wget https://zenodo.org/records/17840656/files/train_test_WF.zip

echo "[5/5] Downloading overheads.zip..."
wget https://zenodo.org/records/17840656/files/overheads.zip

echo ""
echo "Extracting datasets..."

unzip -q full_list.zip && echo "  ✓ full_list extracted"
unzip -q reduced_list.zip && echo "  ✓ reduced_list extracted"
unzip -q traffic_captures.zip && echo "  ✓ traffic_captures extracted"
unzip -q train_test_WF.zip && echo "  ✓ train_test_WF extracted"
unzip -q overheads.zip && echo "  ✓ overheads extracted"

echo ""
echo "Cleaning up ZIP files..."
rm -f *.zip

echo ""
echo "✓ Dataset download and extraction complete!"
echo "Total extracted size: ~100GB"
