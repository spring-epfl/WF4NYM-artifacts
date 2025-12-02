# Feature Importance Analysis

This folder contains scripts for extracting and analyzing feature importance in website fingerprinting attacks. The analysis uses Random Forest classifiers to evaluate how different feature groups contribute to classification accuracy.

## Overview

Most of the code in this folder is adapted from **reWeFDE** [1], which is a re-implementation of the WeFDE framework to measure information leakage for the paper "Tik-Tok: The Utility of Packet Timing in Website Fingerprinting Attacks" [2] accepted in Privacy Enhancing Technologies Symposium (PETS) 2020.

## Main Components

- **`feature_importance.ipynb`**: Jupyter notebook for running the complete feature extraction and classification pipeline

## Feature Groups

The analysis evaluates the following feature groups:
- **Packet Count**: Includes multiple statistics of incoming, outgoing and global packet counts.
- **Time**: Includes the 25th, 50th, 75th and 100th quartiles of packet transmission time, relative to the start of the capture.
- **Inter-arrival Time**: Includes the mean, variance, and standard deviation of packet inter-arrival times.
- **N-gram**: Includes the frequency of all possible sequences of n packets, with n between 2 and 6 (e.g., counts how many times the sequence [-1, +1, -1] or [+1, +1, +1, -1, -1, -1] appears on the trace).
- **Transposition**: Includes statistics of the precise ordering (position) of outgoing and incoming packets.
- **Bursts**: Includes statistics of burst of incoming data, which corresponds to the number of consecutive incoming packets received without outgoing packet in between.

## References

[1] reWeFDE - https://github.com/notem/reWeFDE

[2] Rahman, M. S., Imani, M., Mathews, N., & Wright, M. (2020). Tik-Tok: The Utility of Packet Timing in Website Fingerprinting Attacks. Privacy Enhancing Technologies Symposium (PETS) 2020.
