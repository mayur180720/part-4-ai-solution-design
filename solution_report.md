# AI Solution Design Report
**Domain:** Healthcare | **Problem:** Medical Image Triage
**Author:** AI Business Analyst | **Reference:** ai_usecase_reference_catalog.csv

---

## Task 1: Business Domain

**Selected Domain: Healthcare**

Healthcare was selected due to its high-impact, time-sensitive nature and strong alignment with AI capabilities in image analysis. Medical imaging is one of the most data-rich, well-researched areas of applied AI, and automation here can directly save lives by reducing diagnostic delays.

---

## Task 2: Business Problem Definition

**What problem is being solved?**
Radiologists and clinical staff face overwhelming volumes of medical images (X-rays, CT scans, MRIs) daily. The current process requires a radiologist to manually review every scan in queue order, meaning a critical case (e.g., a pneumothorax or early-stage tumour) may wait hours behind routine scans.

**Who are the users and stakeholders?**
Primary users are radiologists and attending physicians. Secondary stakeholders include hospital administrators, patients, and health insurance providers who depend on timely and accurate diagnostics.

**Current process and its limitations:**
Today, scans are reviewed in FIFO (first-in, first-out) order. This creates two key limitations: (1) critical findings are not fast-tracked, increasing risk of adverse outcomes; and (2) radiologist burnout is high due to repetitive review of normal scans. There is no automated pre-screening layer to prioritise the queue intelligently.

---

## Task 3: AI Task Type

**Selected AI Task: Image Classification**

The problem maps directly to image classification — given a medical scan, the model must output a priority label such as `Critical`, `Moderate`, or `Routine`. This is a multi-class classification over image inputs. Image classification using convolutional neural networks is well-established in medical imaging literature (e.g., CheXNet for chest X-rays) and delivers strong performance with appropriate training data. Object detection could be a future extension for localising abnormalities, but classification is the right starting point for triage.

---

## Task 4: Data Requirement Plan

**Type of data needed:** Medical images — primarily chest X-rays and CT scan slices in DICOM or JPEG format, paired with radiologist-labelled reports.

| Attribute | Details |
|---|---|
| **Data type** | Unstructured (images) + Semi-structured (DICOM metadata) |
| **Input features** | Raw pixel arrays, patient age, scan type, acquisition date |
| **Target variable** | Triage label: `Critical` / `Moderate` / `Routine` |
| **Collection method** | Hospital PACS (Picture Archiving and Communication System), anonymised public datasets (NIH ChestX-ray14, CheXpert) |
| **Data quality risks** | Class imbalance (critical cases are rare), label disagreement between radiologists, equipment variation across hospitals, missing or corrupted DICOM metadata |

A minimum of 50,000 labelled images across classes is recommended for training, with deliberate oversampling or augmentation of the `Critical` class to counter imbalance.

---

## Task 5: Model Recommendation

**Recommended Architecture: CNN with Transfer Learning (EfficientNet-B4 or ResNet-50)**

A pre-trained CNN backbone is the most pragmatic choice. Training from scratch on medical images requires millions of samples; transfer learning from ImageNet significantly reduces data requirements and training time. EfficientNet-B4 offers an excellent accuracy-to-compute trade-off and has demonstrated strong performance on chest X-ray benchmarks. The final classification head is replaced with a 3-class softmax layer fine-tuned on hospital data.

Fine-tuning strategy: freeze early convolutional layers (low-level edge detectors transfer well), and unfreeze deeper layers during training. Apply class-weighted loss to handle the critical-case imbalance. Grad-CAM visualisations should be generated to produce heatmaps that radiologists can use to verify model attention regions.

---

## Task 6: Evaluation Plan

**Technical Metrics:**

| Metric | Target |
|---|---|
| Recall (Critical class) | ≥ 0.95 — missing a critical case is the highest-risk failure |
| Precision (Critical class) | ≥ 0.80 — minimise false alarms that waste radiologist time |
| Overall AUC-ROC | ≥ 0.92 |
| Review time reduction | ≥ 30% faster queue clearance |

**Business Metrics:** Average time-to-diagnosis for critical cases; radiologist satisfaction scores; rate of adverse outcomes attributable to delayed triage.

**Failure cases to monitor:** The model may underperform on scan types underrepresented in training (e.g., paediatric scans, non-standard positioning). It may also degrade on images from scanners with different calibration.

**Human review process:** All `Critical` predictions must be immediately surfaced to a radiologist for confirmation. The model operates in an assistive, not autonomous, capacity. A monthly audit of 500 randomly sampled predictions by senior radiologists is recommended to detect drift.

---

## Task 7: Responsible AI Considerations

**Bias in data:** Training data from a single hospital or geography may not generalise. Scans from older equipment or patients with rare demographics may be systematically underrepresented, leading the model to perform worse on those subgroups. Bias audits should be run across patient age, sex, and equipment type before deployment.

**Incorrect predictions and privacy:** A false negative (classifying a critical case as routine) carries life-threatening risk. A false positive creates unnecessary urgency and erodes clinician trust over time. Both must be tracked. All patient data must be fully anonymised (de-identified per HIPAA/GDPR) before ingestion; the model must never store identifiable information.

**Over-reliance on AI:** Clinical staff may begin to defer entirely to model output, reducing their own diagnostic vigilance. Clear UI design must frame the model as a "second opinion" tool, not a decision authority. Training for radiologists on interpreting and overriding AI output is mandatory.

**Human oversight:** A radiologist must always make the final diagnostic decision. The model's confidence score and Grad-CAM heatmap should always be visible to aid interpretation. A fallback protocol must exist for when the model is unavailable or confidence is low.

---

## Task 8: Final Solution Summary

| Field | Details |
|---|---|
| **Problem** | Radiologists cannot prioritise urgent cases in high-volume scan queues, causing diagnostic delays |
| **Proposed AI Solution** | CNN-based image classifier that triages incoming scans as Critical / Moderate / Routine and re-orders the radiologist's worklist automatically |
| **Required Data** | 50,000+ labelled medical images (X-rays, CT scans) with DICOM metadata, sourced from hospital PACS and public datasets |
| **Model Recommendation** | EfficientNet-B4 with transfer learning, fine-tuned with class-weighted loss and Grad-CAM explainability layer |
| **Expected Business Impact** | 30%+ reduction in time-to-diagnosis for critical cases; lower radiologist burnout; improved patient outcomes for time-sensitive conditions |
| **Risks & Mitigation** | False negatives mitigated by high-recall optimisation and mandatory human confirmation; bias mitigated by diverse training data and subgroup audits; privacy protected through full de-identification pipeline |

---

### Architecture Diagram

See `diagrams/solution_architecture.png` for the end-to-end system pipeline.

> **Diagram description:** The pipeline flows left to right across five stages:
> 1. **Data Ingestion** — Hospital PACS system exports DICOM scans → anonymisation and de-identification layer → image pre-processing (resize, normalise, augment)
> 2. **Model Inference** — Pre-processed image tensor fed into EfficientNet-B4 fine-tuned classification head → outputs triage label + confidence score + Grad-CAM heatmap
> 3. **Triage Queue** — Predictions written to a priority queue database; `Critical` cases surfaced immediately to the on-call radiologist dashboard
> 4. **Radiologist Review** — Clinician views scan + heatmap + AI confidence; confirms or overrides the label
> 5. **Feedback Loop** — Override decisions logged back to training pipeline for periodic model re-training and drift monitoring
