# runs the experiments for GFT and GFT-p from the numerical section

from src.generative_features_m import parameter_search, train_generative_features, phi_gabor
import torch
import argparse
from src.get_setting_m import getSetting
import numpy as np
import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parser
parser = argparse.ArgumentParser(description="Choose parameters")
parser.add_argument("--functions", type=bool, default=False)
parser.add_argument("--setting", type=int, default=0)
parser.add_argument("--activation", type=str, default="Fourier")
inp = parser.parse_args()
synthetic_functions = inp.functions
setting = inp.setting

# Seeds
np.random.seed(1)
torch.manual_seed(1)

# Dataset names
ds_names = [
    "propulsion",
    "galaxy",
    "airfoil",
    "CCPP",
    "telemonitoring-total",
    "skillcraft",
]

# Print dataset name
if not synthetic_functions:
    print(ds_names[setting])

# Activation definition
if inp.activation == "Fourier":
    act = lambda x, w, xi, sigma: torch.exp(2j * torch.pi * torch.sum(x[:, None, :] * w[None, :, :], dim=-1))
    bias = False
    sigma = 1.0
elif inp.activation == "sigmoid":
    act = lambda x, w, xi, sigma: torch.sigmoid(torch.sum(x[:, None, :] * w[None, :, :], dim=-1))
    bias = True
    sigma = 1.0
elif inp.activation == "cosine":
    act = lambda x, w, xi, sigma: torch.cos(torch.pi * torch.sum(x[:, None, :] * w[None, :, :], dim=-1))
    bias = True
    sigma = 1.0
elif inp.activation == "gabor":
    act = phi_gabor
    bias = False
    sigma = 1.0
else:
    raise ValueError("Unknown activation")

# Load dataset
points, fun_vals, test_points, test_vals = getSetting(synthetic_functions, setting)

# Hyperparameter search
gen_params, post_params = parameter_search(
    points,
    fun_vals,
    act=act,
    sigma=sigma,
    bias=bias
)

equal_params = gen_params[0] == post_params[0] and gen_params[1] == post_params[1]

gen_mses = []
post_mses = []
n_trials = 5

# Main experiment loop
for trial in range(n_trials):
    points, fun_vals, test_points, test_vals = getSetting(synthetic_functions, setting)
    n_train_points = int(0.9 * points.shape[0])
    val_points = points[n_train_points:]
    val_fun_vals = fun_vals[n_train_points:]
    points = points[:n_train_points]
    fun_vals = fun_vals[:n_train_points]

    # GFT training
    _, model, _, w_finder = train_generative_features(
        points,
        fun_vals,
        n_features_train=gen_params[1],
        prior_lam=gen_params[0],
        val_points=val_points,
        val_fun_vals=val_fun_vals,
        bias=bias,
        act=act,
        sigma=sigma
    )

    # GFT-p postprocessing if needed
    if not equal_params:
        _, _, _, w_finder = train_generative_features(
            points,
            fun_vals,
            n_features_train=post_params[1],
            prior_lam=post_params[0],
            val_points=val_points,
            val_fun_vals=val_fun_vals,
            bias=bias,
            act=act,
            sigma=sigma
        )

    # Test GFT
    with torch.no_grad():
        preds = model(test_points) 
        gen_mse = torch.mean((preds - test_vals) ** 2, -1)
    print(
        "MSE of the generative model for regularization strength {0:.2E} and {2} features is: {1:.2E}".format(
            gen_params[0], gen_mse, gen_params[1]
        )
    )

    # Test GFT-p
    with torch.no_grad():
        preds = w_finder(test_points)
        post_mse = torch.mean((preds - test_vals) ** 2, -1)
    print(
        "MSE of the postprocessing for regularization strength {0:.2E} and {2} features is: {1:.2E}".format(
            post_params[0], post_mse, post_params[1]
        )
    )

    gen_mses.append(gen_mse.item())
    post_mses.append(post_mse.item())

# Logging
title = (
    "Setting {0}, functions {1}, activation ".format(setting, synthetic_functions)
    + inp.activation
    + "\n"
)
write_string1 = "Avg MSE of the generative model for regularization strength {0:.2E} and {2} features is: {1:.2E}\n".format(
    gen_params[0], np.mean(gen_mses), gen_params[1]
)
write_string2 = "Avg MSE of the postprocessing for regularization strength {0:.2E} and {2} features is: {1:.2E}\n".format(
    post_params[0], np.mean(post_mses), post_params[1]
)

print(write_string1)
print(write_string2)
with open("./log.txt", "a") as f:
    f.write(title)
    f.write(write_string1)
    f.write(write_string2)
    f.write("\n\n")
