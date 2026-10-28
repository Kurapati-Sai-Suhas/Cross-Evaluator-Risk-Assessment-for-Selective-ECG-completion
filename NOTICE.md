# Notices and attribution

## Data

The models were trained, calibrated and thresholded on **PTB-XL v1.0.3** (PhysioNet), released under the Creative Commons
Attribution 4.0 International licence (CC BY 4.0):

* P. Wagner, N. Strodthoff, R.-D. Bousseljot, D. Kreiseler, F. I. Lunze, W. Samek, T. Schaeffter, "PTB-XL, a large
  publicly available electrocardiography dataset," *Scientific Data*, vol. 7, art. 154, 2020,
  doi:10.1038/s41597-020-0495-6.
* P. Wagner et al., PTB-XL v1.0.3, PhysioNet, doi:10.13026/kfzx-aw45.

No PTB-XL recording, CPSC 2018 recording or other patient data is included in this repository. The example inputs in
`examples/synthetic/` are generated signals, not ECG recordings.

## Reconstructor architecture

The reconstructor G v0.1 is this project's own PyTorch implementation (`src/models/l2g.py`) of the generator described
in:

* A. Srivastava, D. Sheet, A. Patra, "L2G-ECG: Learning to Generate Missing Leads in ECG Signals using Adversarial
  Autoencoder," *IEEE Journal of Biomedical and Health Informatics*, 2025, doi:10.1109/JBHI.2025.3587428.

The project adds 12 lead-mask input channels and exact copy-through of observed samples. No code from the authors' public
repository was copied (that repository states no licence).

## Licence of this repository

**No licence has been chosen yet.** Until the authors add one, all rights are reserved. Viewing the code is permitted by
GitHub's terms; reuse needs the authors' permission.

## Status

Research prototype only: **not a medical device**, not clinically validated, and not for diagnosis or treatment.
