# Main script for the comparison of a plain neural network training


import torch
import argparse
from src.get_setting import getSetting
import numpy as np
import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Choose parameters")
parser.add_argument("--setting", type=int, default=0)
parser.add_argument("--activation", type=str, default="Fourier")
parser.add_argument("--functions", type=bool, default=False)
inp = parser.parse_args()
synthetic_functions = inp.functions
setting = inp.setting

np.random.seed(1)
torch.manual_seed(1)


if inp.activation == "Fourier":
    act = lambda x: torch.exp(2j * torch.pi * x)
    bias = False
    cplx = True
elif inp.activation == "sigmoid":
    act = torch.nn.functional.sigmoid
    bias = True
    cplx = False
elif inp.activation == "relu":
    act = torch.nn.functional.relu
    bias = True
    cplx = False
else:
    raise ValueError("Unknown activation")


class TwoLayerNN(torch.nn.Module):
    def __init__(self, activation, n_hidden, dim, bias, cplx):
        super().__init__()
        self.activation = activation
        self.layer1 = torch.nn.Linear(dim, n_hidden, bias=bias)
        self.layer2 = torch.nn.Linear(
            n_hidden, 1, bias=bias, dtype=torch.complex64 if cplx else torch.float
        )

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x))).real


points, fun_vals, test_points, test_vals = getSetting(synthetic_functions, setting)
n_train_points = int(0.9 * points.shape[0])
val_points = points[n_train_points:]
val_fun_vals = fun_vals[n_train_points:]
points = points[:n_train_points]
fun_vals = fun_vals[:n_train_points]
n_prior_points = max(1000, points.shape[0])
points_shift = torch.min(points, 0)[0]
points_scale = torch.max(points - points_shift, 0)[0]

lam_choices = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
best_lam = None
best_mse = 1e8
for lam in lam_choices:
    model = TwoLayerNN(act, 100, points.shape[1], bias, cplx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    iterations = 100000
    val_mse = -1

    for i in (progress_bar := tqdm.tqdm(range(iterations))):
        preds = model(points).squeeze()
        mse = torch.mean((preds - fun_vals) ** 2)
        smoothness_prior = 0
        if lam > 0:
            prior_points = (
                torch.rand((n_prior_points, points.shape[1]), device=device)
                * points_scale
                + points_shift
            )
            prior_points.requires_grad_(True)
            prior_preds = model(prior_points).squeeze()
            pred_diffs = torch.autograd.grad(
                torch.sum(prior_preds), prior_points, create_graph=True
            )[0]
            smoothness_prior = torch.mean(torch.sum(torch.abs(pred_diffs), -1))
        loss = mse + lam * smoothness_prior
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 200 == 0:
            preds = model(val_points).squeeze()
            val_mse = torch.mean((preds - val_fun_vals) ** 2).item()
        progress_bar.set_description(
            "Loss: {0:.2E}, MSE: {1:.2E}, Validation MSE {2:.2E}".format(
                loss.item(), mse.item(), val_mse
            )
        )

    preds = model(val_points).squeeze()
    val_mse = torch.mean((preds - val_fun_vals) ** 2).item()
    if val_mse < best_mse:
        best_mse = val_mse
        best_lam = lam


n_trials = 5

nn_mses = []

for trial in range(n_trials):
    points, fun_vals, test_points, test_vals = getSetting(synthetic_functions, setting)
    n_train_points = int(0.9 * points.shape[0])
    val_points = points[n_train_points:]
    val_fun_vals = fun_vals[n_train_points:]
    points = points[:n_train_points]
    fun_vals = fun_vals[:n_train_points]

    model = TwoLayerNN(act, 100, points.shape[1], bias, cplx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    iterations = 100000
    val_mse = -1
    print(best_lam)

    for i in (progress_bar := tqdm.tqdm(range(iterations))):
        preds = model(points).squeeze()
        mse = torch.mean((preds - fun_vals) ** 2)
        smoothness_prior = 0
        if best_lam > 0:
            prior_points = (
                torch.rand((n_prior_points, points.shape[1]), device=device)
                * points_scale
                + points_shift
            )
            prior_points.requires_grad_(True)
            prior_preds = model(prior_points).squeeze()
            pred_diffs = torch.autograd.grad(
                torch.sum(prior_preds), prior_points, create_graph=True
            )[0]
            smoothness_prior = torch.mean(torch.sum(torch.abs(pred_diffs), -1))
        loss = mse + best_lam * smoothness_prior
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 200 == 0:
            preds = model(val_points).squeeze()
            val_mse = torch.mean((preds - val_fun_vals) ** 2).item()
        progress_bar.set_description(
            "Loss: {0:.2E}, MSE: {1:.2E}, Validation MSE {2:.2E}".format(
                loss.item(), mse.item(), val_mse
            )
        )

    # test
    with torch.no_grad():
        preds = model(test_points).squeeze()
        nn_mse = torch.mean((preds - test_vals) ** 2)
    print(nn_mse)

    nn_mses.append(nn_mse.item())

title = (
    "Setting {0}, lam {1:.2E}, functions {2}, activation ".format(
        setting, best_lam, synthetic_functions
    )
    + inp.activation
    + "\n"
)
write_string1 = "Avg MSE is: {0:.2E}\n".format(np.mean(nn_mses))
print(write_string1)
with open("./log_nn.txt", "a") as f:
    f.write(title)
    f.write(write_string1)
    f.write("\n\n")
