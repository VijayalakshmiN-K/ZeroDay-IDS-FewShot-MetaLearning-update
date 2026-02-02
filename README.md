# ZeroDay-IDS-FewShot-MetaLearning-update
Few-shot meta-learning based intrusion detection framework for zero-day attack detection using multiple benchmark datasets
# Zero-Day Intrusion Detection using Few-Shot Meta-Learning

This repository presents a few-shot meta-learning based intrusion detection framework
designed to detect zero-day attacks using limited labeled samples.

## Key Contributions
- Few-shot learning formulation for zero-day intrusion detection
- Episodic training using meta-learning principles
- Integration of Prototypical Networks, Relation Networks, and MAML
- Robust detection under limited labeled data scenarios

## Datasets
The following benchmark datasets are used:
- CICIDS 2017
- UNSW-NB15
- NSL-KDD
- CICIOT2023

Due to dataset size and licensing restrictions, datasets are not included.
Please download datasets from their official sources.

## Experimental Setup
- N-way K-shot classification framework
- Episodic meta-training strategy
- Evaluation using Accuracy, Precision, Recall, and F1-score

## Reproducibility
All experiments are conducted with a fixed random seed to ensure reproducibility.

## Usage
```bash
python main.py
