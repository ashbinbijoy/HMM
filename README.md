# Hidden Markov Model (HMM) for Driver Activity Recognition

## Overview
This project implements a **Hidden Markov Model (HMM)** for recognizing driver activities based on sensor data. The model processes pre-recorded data, applies preprocessing techniques, and predicts driver states using a probabilistic approach.

## Dataset
- **Training Data:** `driver_activity_train.csv`
- **Testing Data:** `driver_activity_test.csv`
- Each dataset contains **10 feature columns** representing sensor readings and **1 target column** representing driver activity labels.

## Preprocessing Steps
1. **Load Data**: Read CSV files and extract features (`X`) and labels (`y`).
2. **Label Encoding**: Convert categorical activity labels into numerical values.
3. **Standardization**: Scale the feature values using `StandardScaler` to improve model performance.

## Model Implementation
- The **HMM** is implemented using the `hmmlearn` library.
- The number of hidden states is set based on the unique activity labels in the training set.
- A **Gaussian HMM** (`hmm.GaussianHMM`) with a diagonal covariance matrix is used.
- The model is trained for `100 iterations` with a `random_state` for reproducibility.

## Performance Metrics
After training, the model is evaluated using:
- **Accuracy**: Measures the overall correctness of predictions.
- **Precision**: Indicates the proportion of correctly classified positive cases.
- **Recall**: Measures the ability to detect true positive cases.
- **F1-Score**: Harmonic mean of precision and recall.

### Results
| Metric  | Score |
|---------|-------|
| Train Accuracy | 0.20 |
| Test Accuracy  | 0.19 |
| Precision  | 0.20 |
| Recall  | 0.19 |
| F1-Score  | 0.18 |

## Dependencies
- `numpy`
- `hmmlearn`
- `scikit-learn`

## Installation
```sh
pip install numpy hmmlearn scikit-learn
```

## Running the Model
To execute the model, run:
```sh
python hmm_driver_activity.py
```
Ensure that the dataset files are present in the working directory.

## Limitations & Future Work
- The current accuracy is **low**; further tuning of parameters and additional feature engineering may improve performance.
- Exploring **deep learning** models such as LSTMs or CNNs could enhance recognition accuracy.

---
This implementation provides a **baseline approach** for driver activity recognition using HMMs. Future work can involve **hybrid models** integrating HMM with **DBN** or **CNN** for better results.
