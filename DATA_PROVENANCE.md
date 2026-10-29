# Dataset provenance - V4

## Medical video

Default extension videos are the complete `example1.mp4` and `example2.mp4` echocardiography examples published in the public University of Tokyo Hospital / University of Tokyo repository:

- Repository: `UTcardiology/video-echo-clip`
- Project: *Video CLIP Model for Multi-View Echocardiography Interpretation*
- Local names: `echo_example1.mp4`, `echo_example2.mp4`
- Policy: the full downloaded clip is used; no short demo derivative is created.

The repository README identifies these files as sample echocardiography videos for testing. Retain the repository/paper citation when using them in academic material.

## 3-D brain MRI

The project downloads one structural T1-weighted MRI from the Amsterdam Open MRI Collection (AOMIC), ID1000 dataset, hosted by OpenNeuro:

- Dataset: OpenNeuro `ds003097`
- Subject: `sub-0001`
- File: `sub-0001_run-1_T1w.nii.gz`
- Modality: real 3-D T1-weighted structural brain MRI
- Download host: public OpenNeuro S3 object

This AOMIC-ID1000 dataset is publicly distributed through OpenNeuro; retain the AOMIC/OpenNeuro provenance and cite the dataset/paper in publications.

The raw NIfTI is preserved. `dataset_manager.py` creates a deterministic uint8 `.npy` volume for the byte-based paper cipher and writes the exact preprocessing metadata beside it.
