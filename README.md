
# EFM-Net Implementation

This anonymized repository provides the core model implementation for the submitted paper:

**EFM-Net: An Electrode Field to Muscle Network for Spinal Cord Stimulation Response Prediction**

The complete training pipeline, experiment scripts, and releasable dataset files will be made available upon publication.

## Repository Structure

```text
.
├── efm_net.py          # Main EFM-Net model and field-map token encoder
├── field_map.py        # Unit field-map construction and electrode layouts
├── token_response.py   # Muscle-query cross-attention and current-modulated response head
└── README.md
