# generates training and test datasets for the different settings

import torch
import src.get_dataset as get_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getSetting(synthetic_functions, setting, n_points=None):
    if synthetic_functions:
        def Friedmann1(x):
            return (
                10 * torch.sin(torch.pi * x[:, 0] * x[:, 1])
                + 20 * (x[:, 2] - 0.5) ** 2
                + 10 * x[:, 3]
                + 5 * x[:, 4]
            )

        def Ishigami(x):
            return (
                torch.sin(x[:, 0])
                + 7 * torch.sin(x[:, 1]) ** 2
                + 0.1 * x[:, 2] ** 4 * torch.sin(x[:, 0])
            )

        def polynomial(x):
            return x[:, 3] ** 2 + x[:, 1] * x[:, 2] + x[:, 0] * x[:, 1] + x[:, 3]

        def kreuz(x):
            return torch.sin(4 * torch.pi * x[:, 0] ** 2 + 1) + torch.cos(
                4 * torch.pi * (x[:, 1] ** 4 + x[:, 1])
            )

        def example(x):
            return torch.sin(torch.sum(x, -1)) + torch.sum(x**2, -1) ** 2

        def shifted_norm(x):
            return torch.sqrt(torch.sum(torch.abs(x - 0.5), -1))

        def sqrt_Friedmann1(x):
            return torch.sqrt(Friedmann1(x))

        def benchmark_highly_oscillatory(x):
            return torch.sin(130 * torch.pi / 3 * (2 * x[:, 0] - 1) ** 2 + 80 * torch.pi * (2 * x[:, 0] - 1))
    
        n_test_points = None
        n_points_ = n_points

        if setting == 0:
            fun = polynomial
            dim = 5
            n_points = 300
        elif setting == 1:
            fun = polynomial
            dim = 10
            n_points = 500
        elif setting == 2:
            fun = Ishigami
            dim = 5
            n_points = 500
        elif setting == 3:
            fun = Ishigami
            dim = 10
            n_points = 1000
        elif setting == 4:
            fun = Friedmann1
            dim = 5
            n_points = 500
        elif setting == 5:
            fun = Friedmann1
            dim = 10
            n_points = 200
        elif setting == 6:
            fun = kreuz
            dim = 2
            n_points = 2000
        elif setting == 7:
            dim = 2
            n_points = 2000
            theta = torch.tensor(torch.pi / 4)
            A = torch.tensor(
                [
                    [torch.cos(theta), -torch.sin(theta)],
                    [torch.sin(theta), torch.cos(theta)],
                ]
            ).to(device)
            fun = lambda x: kreuz(torch.matmul(x, A))
        elif setting == 8:
            dim = 2
            n_points = 2000
            theta = torch.tensor(torch.pi / 4)
            A = torch.tensor([[1.0, 0.3], [0.3, 1.0]]).to(device)
            fun = lambda x: kreuz(torch.matmul(x, A))
        elif setting == 9:
            dim = 10
            n_points = 1000
            fun = example
        elif setting == 10:
            dim = 20
            n_points = 1000
            fun = shifted_norm
        elif setting == 11:
            dim = 5
            n_points = 500
            fun = sqrt_Friedmann1
        elif setting == 99:
            dim = 1
            n_points = 10000
            fun = benchmark_highly_oscillatory

        if n_points_:
            n_points = n_points_
        n_test_points = n_points

        def sample_support_points(N):
            return torch.rand((N, dim), device=device)


        points = sample_support_points(n_points)
        fun_vals = fun(points)
        test_points = sample_support_points(n_test_points)
        test_vals = fun(test_points)

    else:
        # Data sets from the SHRIMP paper
        ds_names = [
            "propulsion",
            "galaxy",
            "airfoil",
            "CCPP",
            "telemonitoring-total",
            "skillcraft",
        ]
        points, points_test, fun_vals, fun_vals_test = get_dataset.getDataset(
            ds_names[setting]
        )
        points = torch.tensor(points, device=device, dtype=torch.float)
        fun_vals = torch.tensor(fun_vals.squeeze(), device=device, dtype=torch.float)
        test_points = torch.tensor(points_test, dtype=torch.float, device=device)
        test_vals = torch.tensor(
            fun_vals_test.squeeze(), dtype=torch.float, device=device
        )
        print(fun_vals.shape, test_vals.shape, points.shape, test_points.shape)
    return points, fun_vals, test_points, test_vals
