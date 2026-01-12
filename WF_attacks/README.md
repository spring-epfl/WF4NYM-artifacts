# Website Fingerprinting Attacks

This directory contains the integration of the ExplainWF framework for evaluating Website Fingerprinting attacks on our captured traffic data.

## Overview

We use the ExplainWF framework ([Jansen et al., PoPETs 2023](https://explainwf-popets2023.github.io/)) which implements multiple state-of-the-art WF attacks:
- **k-FP** (k-Fingerprinting): Random Forest-based classifier
- **DF** (Deep Fingerprinting): Deep neural network classifier
- **Tik-Tok**: Timing-aware deep learning classifier  
- **SVM**: Support Vector Machine classifier

## Setup

**Important: Python Version Requirement**

The ExplainWF framework requires **Python 3.9 or 3.10** due to TensorFlow 2.10 dependency constraints (TensorFlow 2.10 does not support Python 3.11+).


### 1. Download ExplainWF Framework

```bash
cd WF_attacks

# Clone the ExplainWF repository
git clone https://github.com/explainwf-popets2023/explainwf-popets2023.github.io.git
```

### 2. Apply Modifications

Apply our patch to enable compatibility with our dataset and add improvements:

```bash
cd explainwf-popets2023.github.io/ml/code

# Apply previous patches
./download.bash
./patch.bash

# Apply the patch
patch -p3 < ../../../explainwf_modifications.patch

cp ../../../train_test.py ./
```

**Key Modifications:**
- **F1 Score**: Added F1 score computation to metrics
- **Zero Division Handling**: Added `zero_division=0` to prevent errors with empty classes
- **Sequence Length**: Increased to 10,000 packets for DF and Tik-Tok (from default 5,000)
- **Feature Importance Export**: Save Random Forest feature importances for k-FP
- **Temp Directory**: Use local `tmp/` folder instead of system temp
- **API Updates**: Modified `test_models.py` for batch testing multiple models

### 3. Install Dependencies

```bash
# Create Python virtual environment (make sure you're using Python 3.9 or 3.10)
python3 -m venv venv
source venv/bin/activate

# Verify Python version
python --version  # Should show Python 3.9.x or 3.10.x

# Install dependencies
pip install --upgrade pip
pip install -r ../requirements.txt
```



## Usage

### Training Models with 5-Fold Cross-Validation


**Note:** The current `train_test.py` implementation only trains and evaluates the Tik-Tok model. k-FP, DF, and SVM are not included in this script.

The `train_test.py` script performs **5-fold cross-validation** training and evaluation. This splits the data into 5 folds and trains on 4 folds while testing on the remaining fold, repeating for each fold.

**Basic Usage:**

```bash

python train_test.py ../../../../data/train_test_WF/configuration00_default.pkl ./
```

**Parameters:**
- **Input file**: Path to the pickle file (e.g., `configuration00_default.pkl`)
- **Output directory**: Where to save trained models and results
- **`--open_world_nmon_pages`**: Number of monitored pages (default: 95)
  - Use 95 for full dataset
  - Use 55 for reduced dataset
  - Omit for closed-world scenario

**What Happens During 5-Fold CV:**

1. Data is split into 5 equal folds
2. For each fold (fold 0-4):
   - Train on 4 folds
   - Test on remaining 1 fold
   - Save model as `<model>_fold<N>.tar.gz`
   - Save results as `output_<model>_fold<N>.json`
3. Aggregate results across all 5 folds

**Output Files:**

```
output_dir/
├── kfp_fold0.tar.gz          # Trained k-FP model (fold 0)
├── kfp_fold1.tar.gz          # Trained k-FP model (fold 1)
├── ...
├── nn_fold0.tar.gz           # Trained DF model (fold 0)
├── ...
├── output_kfp_fold0.json     # Results for k-FP fold 0
├── output_kfp_fold1.json     # Results for k-FP fold 1
├── ...
├── output_nn_fold0.json      # Results for DF fold 0
├── ...
└── aggregated_results.json   # Summary across all folds
```

### Testing Pre-Trained Models

If you have pre-trained models and want to test on new data:

```bash
python test_models.py \
    /path/to/test_data.pkl \
    /path/to/output_folder/ \
    /path/to/models/
```

This will test all models (kfp, nn, tt, svm) found in the models folder.

## Understanding Results

Each output JSON file contains:

```json
{
  "acc": 0.85,           // Overall accuracy
  "recall": 0.83,        // Recall (sensitivity)
  "precision": 0.84,     // Precision
  "f1": 0.835,           // F1 score
  "fpr": 0.05,           // False positive rate (open-world only)
  "tp": 1234,            // True positives
  "fp": 56,              // False positives
  "tn": 789,             // True negatives
  "fn": 123,             // False negatives
  "confusion": [[...]],  // Confusion matrix
  "training_set": "...", // Training data used
  "test_set": "..."      // Test data used
}
```

## Notes

- **Training Time**: 
  - k-FP: ~10-30 minutes per fold
  - DF/Tik-Tok: ~1-3 hours per fold (with GPU)
  - Complete 5-fold CV: ~5-15 hours per configuration
  
- **Memory Requirements**:
  - k-FP: ~32GB RAM
  - DF/Tik-Tok: ~64GB RAM

- **Reproducibility**: Neural network results may vary slightly between runs due to random initialization, even with fixed seeds
