# Printed-label dataset

This directory stores the novel printed-label anomaly-inspection dataset.

- `reference/label_master.png`: digital reference for alignment checks.
- `manifests/capture_manifest.csv`: provenance and split record for every photograph.
- `raw/train_normal` and `raw/val_normal`: normal-only learning data.
- `raw/test`: held-out normal and defective images.
- `masks/test`: binary ground-truth masks for defective test images only.

Follow `docs/LABEL_DATA_COLLECTION.md` before adding photographs.
