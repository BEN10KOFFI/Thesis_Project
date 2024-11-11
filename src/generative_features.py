# main implmenetation of the generative feature training and
# corresponding postprocessing

import torch
import numpy as np
import tqdm
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"


def dense_generator(input_dim, output_dim, hidden_dim=512):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    )


phi = lambda x: torch.exp(2j * torch.pi * x)


class LearnedRFFDual(torch.nn.Module):
    def __init__(
        self, points, fun_vals, Nw, prior_lam, n_prior_points, reg, act=phi, bias=False
    ):
        super().__init__()
        self.bias = bias
        if self.bias:
            points = torch.cat((points, torch.ones_like(points[:, :1])), 1)
        self.points_shift = torch.min(points, 0)[0]
        self.points_scale = torch.max(points - self.points_shift, 0)[0]
        self.points = points
        self.fun_vals = fun_vals
        self.dim = points.shape[-1]
        self.generator = dense_generator(self.dim, self.dim).to(device)
        self.Nw = Nw
        self.prior_lam = prior_lam
        self.n_prior_points = n_prior_points
        self.reg = reg
        self.act = act

    def sample_latent(self, N):
        return torch.randn((N, self.dim), device=device)

    def sample_w(self, N):
        latent = self.sample_latent(N)
        w = self.generator(latent)
        return w

    def get_wb(self, with_prior=False):
        w = self.sample_w(self.Nw)
        A = self.act(torch.sum(w[None, :, :] * self.points[:, None, :], -1)) / self.Nw
        if self.bias:
            A = torch.cat((A, torch.ones_like(A[:, :1])), 1)
        AtA = torch.matmul(torch.adjoint(A), A) + self.reg * torch.eye(
            self.Nw + 1 * self.bias, device=device
        )
        Aty = torch.mv(torch.adjoint(A), self.fun_vals.to(A))
        b = torch.linalg.solve(AtA, Aty)
        return w, b, A

    def forward(self, x):
        w, b, _ = self.get_wb(True)
        if x.shape[1] < self.points.shape[1]:
            x = torch.cat((x, torch.ones_like(x[:, :1])), 1)
        A = self.act(torch.sum(w[None, :, :] * x[:, None, :], -1)) / self.Nw
        if self.bias:
            A = torch.cat((A, torch.ones_like(A[:, :1])), 1)
        return torch.real(torch.mv(A, b))

    def loss(self, reg=0.01):
        w, b, A = self.get_wb()
        preds = torch.real(torch.mv(A, b))
        mse = torch.mean((preds - self.fun_vals) ** 2, -1)
        if self.prior_lam != 0:
            prior_points = (
                torch.rand((self.n_prior_points, self.points.shape[1]), device=device)
                * self.points_scale
                + self.points_shift
            )
            prior_points.requires_grad_(True)
            A_prior = (
                self.act(torch.sum(w[None, :, :] * prior_points[:, None, :], -1))
                / self.Nw
            )
            if self.bias:
                A_prior = torch.cat((A_prior, torch.ones_like(A_prior[:, :1])), 1)
            prior_preds = torch.real(torch.mv(A_prior, b))
            pred_diffs = torch.autograd.grad(
                torch.sum(prior_preds), prior_points, create_graph=True
            )[0]
            smoothness_prior = torch.mean(torch.sum(torch.abs(pred_diffs), -1))
        else:
            smoothness_prior = 0.0
        return mse, self.prior_lam * smoothness_prior


class RFF_w(torch.nn.Module):
    def __init__(
        self,
        points,
        fun_vals,
        w,
        generator,
        prior_lam,
        n_prior_points,
        reg,
        act=phi,
        bias=False,
    ):
        super().__init__()
        self.bias = bias
        if self.bias:
            points = torch.cat((points, torch.ones_like(points[:, :1])), 1)
        self.points_shift = torch.min(points, 0)[0]
        self.points_scale = torch.max(points - self.points_shift, 0)[0]
        self.points = points
        self.fun_vals = fun_vals
        self.latent_w = torch.nn.Parameter(w)
        self.generator = generator
        self.Nw = w.shape[0]
        self.prior_lam = prior_lam
        self.n_prior_points = n_prior_points
        self.reg = reg
        self.act = act

    def get_b(self, with_prior=False):
        w = self.generator(self.latent_w)
        A = self.act(torch.sum(w[None, :, :] * self.points[:, None, :], -1)) / self.Nw
        if self.bias:
            A = torch.cat((A, torch.ones_like(A[:, :1])), 1)
        AtA = torch.matmul(torch.adjoint(A), A) + self.reg * torch.eye(
            self.Nw + 1 * self.bias, device=device
        )
        Aty = torch.mv(torch.adjoint(A), self.fun_vals.to(A))
        b = torch.linalg.solve(AtA, Aty)
        return b, A, w

    def forward(self, x):
        b, _, w = self.get_b(True)
        if x.shape[1] < self.points.shape[1]:
            x = torch.cat((x, torch.ones_like(x[:, :1])), 1)
        A = self.act(torch.sum(w[None, :, :] * x[:, None, :], -1)) / self.Nw
        if self.bias:
            A = torch.cat((A, torch.ones_like(A[:, :1])), 1)
        return torch.real(torch.mv(A, b))

    def loss(self):
        b, A, w = self.get_b()
        preds = torch.real(torch.mv(A, b))
        mse = torch.mean((preds - self.fun_vals) ** 2, -1)
        if self.prior_lam:
            prior_points = (
                torch.rand((self.n_prior_points, self.points.shape[1]), device=device)
                * self.points_scale
                + self.points_shift
            )
            prior_points.requires_grad_(True)
            A_prior = (
                self.act(torch.sum(w[None, :, :] * prior_points[:, None, :], -1))
                / self.Nw
            )
            if self.bias:
                A_prior = torch.cat((A_prior, torch.ones_like(A_prior[:, :1])), 1)
            prior_preds = torch.real(torch.mv(A_prior, b))
            pred_diffs = torch.autograd.grad(
                torch.sum(prior_preds), prior_points, create_graph=True
            )[0]
            smoothness_prior = torch.mean(torch.sum(torch.abs(pred_diffs), -1))
        else:
            smoothness_prior = 0.0
        return mse, self.prior_lam * smoothness_prior


def train_generative_features(
    points,
    fun_vals,
    val_points=None,
    val_fun_vals=None,
    n_features_train=None,
    prior_lam=0.0,
    n_prior_points=None,
    lam=1e-7,
    test_points=None,
    test_vals=None,
    act=phi,
    bias=False,
):
    n_features_train = 100 if n_features_train is None else n_features_train
    n_prior_points = 1000 

    model = LearnedRFFDual(
        points,
        fun_vals,
        n_features_train,
        prior_lam,
        n_prior_points,
        lam,
        act=act,
        bias=bias,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("\nTrain Generator:")
    iterations = 40000
    batch_size = 1
    val_mse = -torch.tensor(1, device=device)
    best_validation = None
    for i in (progress_bar := tqdm.tqdm(range(iterations))):
        if i % batch_size == 0:
            optimizer.zero_grad()
        mse, reg = model.loss()
        loss = mse + reg
        loss.backward()
        if (i + 1) % batch_size == 0:
            optimizer.step()
        if val_points is not None:
            if (i + 1) % 200 == 0:
                val_mse_mean = 0
                for _ in range(10):
                    with torch.no_grad():
                        preds = model(val_points)
                        val_mse_mean += torch.mean(
                            (preds - val_fun_vals) ** 2, -1
                        ).item()
                val_mse = val_mse_mean / 10
                if not test_points is None:
                    with torch.no_grad():
                        preds = model(test_points)
                        print(torch.mean((preds - test_vals) ** 2, -1).item(), val_mse)
                        print(torch.max((preds - test_vals) ** 2) / test_vals.shape[0])
                if best_validation is None or best_validation > val_mse:
                    best_validation = val_mse
                    best_parameters = copy.deepcopy(model.state_dict())
        progress_bar.set_description(
            "Loss: {0:.2E}, MSE: {1:.2E}, Validation MSE {2:.2E}".format(
                loss.item(), mse.item(), val_mse
            )
        )
    gen_mse = -1
    if val_points is not None:
        gen_mse_sum = 0
        for _ in range(10):
            with torch.no_grad():
                preds = model(val_points)
                gen_mse_sum += torch.mean((preds - val_fun_vals) ** 2, -1).item()
        gen_mse = gen_mse_sum / 10
        print("Generative output MSE: {0:.2E}".format(gen_mse))

    print(f"\nGradient descent in the latent space:")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    w_finder = RFF_w(
        points,
        fun_vals,
        torch.randn_like(model.sample_latent(n_features_train)),
        model.generator,
        prior_lam,
        n_prior_points,
        lam,
        act=act,
        bias=bias,
    )

    optimizer = torch.optim.Adam(w_finder.parameters(), lr=1e-4)

    iterations = 40000
    val_mse = -torch.tensor(1, device=device)
    best_validation = None
    for i in (progress_bar := tqdm.tqdm(range(iterations))):
        lam = lam
        optimizer.zero_grad()
        mse, reg = w_finder.loss()
        loss = mse + reg
        loss.backward()
        optimizer.step()
        if (i + 1) % 200 == 0 and val_points is not None:
            with torch.no_grad():
                preds = w_finder(val_points)
                val_mse = torch.mean((preds - val_fun_vals) ** 2, -1)
            if best_validation is None or best_validation > val_mse:
                best_validation = val_mse
                best_parameters = copy.deepcopy(w_finder.state_dict())
        progress_bar.set_description(
            "Loss: {0:.2E}, MSE: {1:.2E}, Validation MSE {2:.2E}".format(
                loss.item(), mse.item(), val_mse.item()
            )
        )
    post_mse = -1
    if val_points is not None:
        w_finder.load_state_dict(best_parameters)
        with torch.no_grad():
            preds = w_finder(val_points)
            post_mse = torch.mean((preds - val_fun_vals) ** 2, -1)
            print("Postprocessing validation MSE: {0:.2E}".format(post_mse.item()))

    return gen_mse, model, post_mse, w_finder


def model_selection(
    points,
    fun_vals,
    val_points,
    val_fun_vals,
    n_features_train_choices=None,
    prior_lam_choices=None,
    n_prior_points=None,
    lam=1e-7,
    test_points=None,
    test_vals=None,
    act=phi,
    bias=False,
):

    if prior_lam_choices is None:
        prior_lam_choices = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    if n_features_train_choices is None:
        n_features_train_choices = [100]

    best_gen_mse = 1e8
    best_post_mse = 1e8
    best_gen_lam = None
    best_post_lam = None
    best_gen_nf = None
    best_post_nf = None
    best_gen_model = None
    best_post_model = None
    for n_features_train in n_features_train_choices:
        for prior_lam in prior_lam_choices:
            print(prior_lam, n_features_train)
            gen_mse, model, post_mse, w_finder = train_generative_features(
                points,
                fun_vals,
                val_points,
                val_fun_vals,
                n_features_train=n_features_train,
                prior_lam=prior_lam,
                n_prior_points=n_prior_points,
                lam=lam,
                test_points=test_points,
                test_vals=test_vals,
                act=act,
                bias=bias,
            )
            if gen_mse < best_gen_mse:
                best_gen_mse = gen_mse
                best_gen_nf = n_features_train
                best_gen_model = model
                best_gen_lam = prior_lam
            if post_mse < best_post_mse:
                best_post_mse = post_mse
                best_post_model = w_finder
                best_post_nf = n_features_train
                best_post_lam = prior_lam

    return [best_gen_mse, best_gen_model, best_gen_lam, best_gen_nf], [
        best_post_mse,
        best_post_model,
        best_post_lam,
        best_post_nf,
    ]


def parameter_search(
    points,
    fun_vals,
    n_features_train_choices=[100],
    prior_lam_choices=[0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    n_prior_points=None,
    lam=1e-7,
    act=phi,
    bias=False,
):
    n_train_points = int(0.9 * points.shape[0])
    val_points = points[n_train_points:]
    val_fun_vals = fun_vals[n_train_points:]
    points = points[:n_train_points]
    fun_vals = fun_vals[:n_train_points]
    generative, postprocessing = model_selection(
        points,
        fun_vals,
        val_points,
        val_fun_vals,
        prior_lam_choices=prior_lam_choices,
        n_features_train_choices=n_features_train_choices,
        n_prior_points=n_prior_points,
        lam=lam,
        test_points=None,
        test_vals=None,
        act=act,
        bias=bias,
    )
    return [generative[2], generative[3]], [postprocessing[2], postprocessing[3]]
