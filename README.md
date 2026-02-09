# Skeleton-based Gesture Recognition with a Lightweight Transformer

---

## Motivation

Skeleton-based gesture recognition is a fundamental problem in human–computer interaction (HCI).

Traditional approaches often rely on frame-level features or sliding-window–based sampling combined with classical classifiers.  
Such methods have limited ability to explicitly model temporal dynamics.

This project revisits a VR-based hand gesture recognition task originally implemented using scikit-learn classifiers, and explores whether a lightweight Transformer encoder can better capture **temporal motion patterns** under a **low-data regime**.

---

## Dataset

The dataset consists of VR hand gesture recordings captured as 3D hand joint coordinates.

- Each frame contains 25 hand joints, represented by (x, y, z) coordinates  
- Each gesture video contains 100–300 frames, with variable sequence lengths  
- Gesture classes include static gestures and simple dynamic gestures 
- Due to slow hand motion and high frame redundancy, raw sequences contain substantial temporal overlap

---

## Data Preprocessing

To enable consistent temporal modelling and reduce redundancy:

- Each gesture sequence is uniformly subsampled
- The subsampled sequence is divided into 3 temporal segments
- Each segment contains 30 frames
- Joint coordinates are normalized relative to a reference joint

This preprocessing strategy preserves coarse temporal structure while avoiding highly correlated sliding-window samples.

---

## Baseline: Classical Machine Learning

As a baseline, gesture sequences are flattened and classified using classical scikit-learn models.

- Temporal structure is implicitly encoded via feature concatenation  
- Models include traditional classifiers such as logistic regression and k-NN  
- No explicit temporal modelling is performed  

---

## Proposed Method: Lightweight Transformer Encoder

A compact Transformer encoder is used to explicitly model temporal dynamics.

Key design choices:

- Input tokens correspond to subsampled skeleton frames
- Self-attention operates along the temporal dimension
- Model capacity is deliberately constrained to avoid overfitting
- Mean pooling is applied over temporal features for classification

This design enables temporal modelling while remaining suitable for small datasets.

---

## Experiments

- Data split is performed at the video level to avoid cross-video leakage  
- Models are trained using cross-entropy loss
- Performance is evaluated using classification accuracy
- Both training and validation loss / accuracy are monitored  
- Each experiment is repeated 5 times with different random seeds  

---

## Results

| Method | Accuracy |
|------|------|
| Classical scikit-learn baseline | Best: 67.0% |
| Transformer (temporal modelling) | 69.72% ± 2.06% (5 runs) |

Although absolute performance gains are modest, the Transformer consistently improves recognition accuracy by explicitly modelling gesture dynamics.

In later training epochs, validation loss may increase slightly while accuracy continues to improve, indicating increased confidence on a small number of misclassified samples — a common behaviour in **low-data regimes**.

---

## Discussion

This project demonstrates that even a lightweight Transformer encoder can outperform classical classifiers for skeleton-based gesture recognition when temporal structure is modelled explicitly.

More importantly, the study highlights the importance of:

- Appropriate temporal abstraction
- Avoiding highly correlated sliding-window samples
- Careful regularisation in low-data settings
