# Multi-label-Classification-with-k-Nearest-Neighbor
A modified K-Nearest Neighbor (KNN) approach for multi-label classification, specifically applied to predicting EC-classes of substrate molecules. Our implementation explores weighted neighbor voting based on predicted Hamming loss.

## Overview
Traditional classification typically assigns a single label to each example. However, real-world scenarios often require multiple labels per instance. In this project, we implement a modified KNN algorithm that:
- Predicts multiple EC-classes for substrate molecules
- Uses neighbor trustworthiness weighting
- Compares performance against traditional KNN

## Key Features
- Custom weighting system based on predicted Hamming loss
- Multi-label prediction capabilities
- Integrated distance metrics (Euclidean, Cosine)
- Performance evaluation using Hamming loss

## Dataset
- 1,039 substrate molecules
- 196 features per molecule
- Multiple EC-class labels per molecule
- Label imbalance (1-248 instances per class)

## Requirements
Python>=3.8
pandas
numpy
scikit-learn
scipy

## Key findings:
- Traditional KNN outperforms modified KNN at lower k values
- Performance converges at higher k values
- Best performance achieved with k=25 (Hamming Loss: 0.28)

## Limitations & Future Work
- High computational complexity for large datasets
- Current implementation lacks neighbor reranking
- Room for improvement in weight calculation method
- Need for cross-validation and more extensive testing
- Potential for ranking model implementation

## Contributors
1. Branley Mmasi
2. David Liu

## Acknowledgments
- Based on Chiang et al.'s 2012 approach.
- Created as part of CS66 Machine Learning (Fall 2023) at Swarthmore College.
- Special thanks to Dr. Ben Mitchell for consultation. 

## License
This project is licensed under the MIT License — see the LICENSE file for details.
