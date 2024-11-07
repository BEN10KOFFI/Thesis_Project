from src.generative_features import parameter_search, train_generative_features
import torch
import argparse
from src.get_setting import getSetting
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Choose parameters")
parser.add_argument("--functions", type=bool, default=False)
parser.add_argument("--setting", type=int, default=0)
parser.add_argument("--activation", type=str, default="Fourier")
inp = parser.parse_args()
synthetic_functions = inp.functions
setting = inp.setting

np.random.seed(1)
torch.manual_seed(1)
ds_names = [
    "propulsion",
    "galaxy",
    "airfoil",
    "CCPP",
    "telemonitoring-total",
    "skillcraft",
]

if not synthetic_functions:
    print(ds_names[setting])


if inp.activation == "Fourier":
    act = lambda x: torch.exp(2j * torch.pi * x)
    bias = False
elif inp.activation == "sigmoid":
    act = torch.nn.functional.sigmoid
    bias = True
elif inp.activation == "cosine":
    act = lambda x: torch.cos(torch.pi * x)
    bias=True
else:
    raise ValueError("Unknown activation")


points, fun_vals, test_points, test_vals = getSetting(synthetic_functions, setting)
gen_params, post_params = parameter_search(points, fun_vals, bias=bias, act=act)

equal_params = gen_params[0] == post_params[0] and gen_params[1] == post_params[1]
gen_mses = []
post_mses = []
n_trials = 5

for trial in range(n_trials):
    points, fun_vals, test_points, test_vals = getSetting(synthetic_functions, setting)
    n_train_points = int(0.9 * points.shape[0])
    val_points = points[n_train_points:]
    val_fun_vals = fun_vals[n_train_points:]
    points = points[:n_train_points]
    fun_vals = fun_vals[:n_train_points]

    _, model, _, w_finder = train_generative_features(
        points,
        fun_vals,
        n_features_train=gen_params[1],
        prior_lam=gen_params[0],
        val_points=val_points,
        val_fun_vals=val_fun_vals,
        bias=bias,
        act=act,
    )
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
        )

    # test
    with torch.no_grad():
        preds = model(test_points)
        gen_mse = torch.mean((preds - test_vals) ** 2, -1)
    print(
        "MSE of the generative model for regularization strength {0:.2E} and {2} features is: {1:.2E}".format(
            gen_params[0], gen_mse, gen_params[1]
        )
    )

    # test
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


title = "Setting {0}, functions {1}\n".format(setting, synthetic_functions)
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
