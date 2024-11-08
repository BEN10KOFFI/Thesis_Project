# Generative Feature Training of Thin 2-Layer Networks

This repository contains the implementations for the paper ["Generative Feature Training of Thin 2-Layer Networks"](https://arxiv.org/abs/2411.xxxxx). The code is written in PyTorch (version 2.4).

Do not hesitate to contact us, if you have any questions.

## Usage

Below, we describe the usage of the code for reproducing the results from the paper.

### Generative Feature Transforms

The script `run_experiments.py` starts the generative feature training. It takes three input arguments for selecting the experiment:

- the argument `functions` specifies whether to run the examples for function approximation (`True`) or for regression on the UCI datasets (`False`). Default is `False`.

- the argument `setting` specifies which exact setting. For function approximation, the settings 0 to 5 refer to the settings from
Table 1, the settings 6 to 8 refer to the visualizations from Section 4.2 and settings 9 to 11 refer to the settings from Table 2.
For regression the settings 0 to 5 refer to the different dataset (in the same order as in Table 3).

- the argument `activation` selects the Phi (choices `Fourier` and `sigmoid`)

Some examples:

```
python run_experiments.py --functions True --setting 0 --activation Fourier
```
for reproducing the results from the first column Table 1 for GFT and GFT-p with Fourier activation

```
python run_experiments.py --setting 0 --activation sigmoid
```
for reproducing the results from Table 2 for GFT and GFT-p with propulsion dataset and sigmoid activation

### Neural Network Comparison

The script `neural_network.py` starts the comparison for neural networks with the same input arguments as the generative feature transforms.


## Citation

```
@article{HN2024,
  title={Generative Feature Training of Thin 2-Layer Networks},
  author={Hertrich, Johannes and Neumayer, Sebastian},
  journal={arXiv preprint arXiv:2411.xxxxx},
  year={2024}
}
```
