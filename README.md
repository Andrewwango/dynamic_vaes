# VAEs

## Contents

`/`

- `run.py` - main training script
- `kvae.py` - KVAE class defining training and eval
- `kvae_model.py` - class defining KVAE model structure and defines forward pass
- `test.ipynb` - visualising latent space, parameters etc. from results
- `auxiliary.py` - data and vis functions

`/bouncing_ball_data/`

- `box.py` - script for generating bouncing ball video data using PyMunk
- `box.npz`, `box_test.npz` - train and val data

`/nonlinear_ball_data/`

- `circular.py` - script for generating circular/elliptical video data using PyMunk
- `circle.npz`, `circle_test.npz` - train and val data
- `elliptical.npz`, `elliptical_test.npz` - train and val data

`/pytorch_cvae_test/`

- implementations of classical static convolutional VAEs online tutorials

`/kvae_original_paper/`

- core components of original implementation of KVAE model by Fraccaro et al.

## Introduction