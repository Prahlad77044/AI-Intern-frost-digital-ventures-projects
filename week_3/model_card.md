# 📄 Model Card — CIFAR-10 ResNet Transfer-Learning Experiments

## 1. Model Overview
This model card documents three ResNet-18 transfer-learning variants trained on the CIFAR-10 dataset:

1. **Last FC-Only Training**  
   - All backbone layers frozen.  
   - Only the final fully connected layer is trained.

2. **Partial Unfreezing (Layer4 + FC)**  
   - Only the last ResNet block (Layer4) and FC layer are trainable.

3. **Fully Unfrozen Fine-Tuning**  
   - All layers are unfrozen and trained end-to-end.

All models start from ImageNet-pretrained weights.

---

## 2. Intended Use

### Appropriate Use
- Educational purposes (understanding transfer learning stages).
- Research on:
  - Transfer learning depth,
  - Calibration (ECE, reliability diagrams, temperature scaling),
  - Robustness to noise/corruptions.
- Baseline for small-scale vision tasks.

### Not Intended For
- Real-world applications.
- Medical or biometric imaging.
- Security-sensitive or safety-critical deployment.

---

## 3. Performance Summary

*(Insert your actual numbers after running experiments)*

| Model | Test Accuracy | ECE (Before TS) | ECE (After Temp Scaling) |
|-------|---------------|------------------|---------------------------|
| Last FC Only         | ~82–85% | 0.03–0.05 | ↓ Improved |
| Layer4 + FC          | ~88–91% | 0.02–0.03 | ↓ Improved |
| Fully Unfrozen       | ~93–95% | ~0.01     | ↓ Slightly Improved |

---

## 4. Evaluation Setup

### Dataset
- **CIFAR-10**  
  - 10 classes  
  - 32×32 color images  
  - Standard train/validation/test splits  

### Metrics
- Accuracy  
- Precision, Recall, F1  
- Expected Calibration Error (ECE)  
- Reliability diagrams  
- Negative Log-Likelihood (NLL)  
- Confusion matrix  

### Robustness Tests
- Additive Gaussian noise  
- Mild corruptions  
- Calibration before/after temperature scaling  

---

## 5. Key Findings

### Transfer Learning Depth
- **FC-only** → Learns limited features; underfits.  
- **Layer4+FC** → Strong middle ground; faster + better accuracy.  
- **Fully Unfrozen** → Best accuracy & best calibration.

### Calibration
- All models are **overconfident** before calibration.  
- Temperature scaling significantly reduces ECE.  
- Fully-unfrozen model shows the best baseline calibration.

---

## 6. Failure Modes

### Common Failures
- Misclassification of:
  - **Cat vs Dog**
  - **Truck vs Automobile**
  - **Bird vs Airplane** (small images make them confusing)
- Images with:
  - Heavy noise  
  - Blur  
  - Occlusion  
  - Non-CIFAR-like distribution  

### Behavior Under Corruption
- Accuracy drops sharply under:
  - Gaussian noise  
  - Motion blur  
  - JPEG compression  
- Calibration worsens unless temperature-scaled.

---

## 7. Robustness Notes

### Strengths
- Good performance even with limited training due to transfer learning.
- Temperature scaling dramatically improves calibration.
- Partial-unfreezing offers great trade-off between speed and performance.

### Weaknesses
- CIFAR-10’s very low resolution reduces robustness to:
  - Small object detail  
  - Noise  
  - Strong augmentations  
- Still vulnerable to adversarial perturbations.
- Overconfident predictions without temperature scaling.

### Recommendations
- Use **temperature scaling** before trusting probabilities.  
- Consider stronger augmentations:
  - Mixup / CutMix  
  - RandAugment  
  - Cutout  

---

## 8. Limitations
- Not production-ready.  
- Limited generalization outside CIFAR-10 domain.  
- Does not estimate uncertainty beyond softmax confidence.  
- No defense against adversarial attacks.

---

## 9. Ethical Considerations
- No sensitive or personal data involved.  
- No identifiable subjects.  
- Lowest risk category for ethical harm.

---

## 10. Contact
- **Author:** *Prahlad Neupane*  
- **Purpose:** Internship project
- **Last Updated:** *11/7/2025*  


