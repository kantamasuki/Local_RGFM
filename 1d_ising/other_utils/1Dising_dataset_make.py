import argparse
import json
from pathlib import Path

import numpy as np


import matplotlib.pyplot as plt


def plot_corr_check(C: np.ndarray, K: float, xi: float, out_path: str):
    r = np.arange(len(C))

    # Exact result: C(r) = (tanh K)^r = exp(-r/xi)
    C_exact = np.exp(-r / xi)

    plt.figure(figsize=(6, 4))

    # A logarithmic plot requires positive correlation values.
    mask = C > 0

    plt.semilogy(r[mask], C[mask], "o", markersize=3, label="sample")
    plt.semilogy(r, C_exact, "-", label=r"$\exp(-r/\xi)$")

    plt.xlabel("r")
    plt.ylabel(r"$C(r)$")
    plt.title(rf"1D Ising correlation, $K={K:.4f}$, $\xi={xi:.2f}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def K_from_xi(xi: float) -> float:
    return np.arctanh(np.exp(-1.0 / xi))


def xi_from_K(K: float) -> float:
    return -1.0 / np.log(np.tanh(K))


def exact_sample_open_ising_1d(
    L: int,
    num_samples: int,
    K: float,
    seed: int = 0,
    chunk_size: int = 10000,
):
    """
    Exact samples from 1D ferromagnetic Ising with free/open boundary:

        H = -K sum_{i=1}^{L-1} s_{i-1} s_{i}

    Returns:
        data: int8 array, shape (num_samples, L), values in {-1, +1}.
    """
    rng = np.random.default_rng(seed)

    p_same = 1.0 / (1.0 + np.exp(-2.0 * K))

    data = np.empty((num_samples, L), dtype=np.int8)

    start = 0
    while start < num_samples:
        end = min(start + chunk_size, num_samples)
        n = end - start

        s0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n, 1))

        same = rng.random((n, L - 1)) < p_same
        tau = np.where(same, 1, -1).astype(np.int8)

        spins = np.empty((n, L), dtype=np.int8)
        spins[:, :1] = s0
        spins[:, 1:] = (s0 * np.cumprod(tau, axis=1)).astype(np.int8)

        data[start:end] = spins
        start = end

    return data


def domain_wall_density(data: np.ndarray) -> float:
    return float(np.mean(data[:, :-1] != data[:, 1:]))


def mean_corr(data: np.ndarray, max_r: int = 256):
    """
    Estimate C(r)=<s_i s_{i+r}> with open boundary.
    Returns array of length max_r+1.
    """
    C = np.empty(max_r + 1, dtype=np.float64)
    C[0] = 1.0

    x = data.astype(np.float32)
    for r in range(1, max_r + 1):
        C[r] = np.mean(x[:, :-r] * x[:, r:])

    return C


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--L", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=100000)

    # Give either --K_list or --xi_list.
    parser.add_argument("--K_list", type=float, nargs="*", default=None)
    parser.add_argument("--xi_list", type=float, nargs="*", default=[32, 64, 128, 256])

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--out_dir", type=str, default="./dataset_ising1d_open")
    parser.add_argument("--max_corr_r", type=int, default=256)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.K_list is not None and len(args.K_list) > 0:
        K_list = args.K_list
        labels = [f"K{K:.6f}" for K in K_list]
    else:
        K_list = [K_from_xi(xi) for xi in args.xi_list]
        labels = [f"xi{int(xi)}" if float(xi).is_integer() else f"xi{xi:g}" for xi in args.xi_list]

    for idx, (K, label) in enumerate(zip(K_list, labels)):
        print("i, K = ", idx, K)
        xi = xi_from_K(K)
        p_dw = 1.0 / (1.0 + np.exp(2.0 * K))

        data = exact_sample_open_ising_1d(
            L=args.L,
            num_samples=args.num_samples,
            K=K,
            seed=args.seed + idx,
            chunk_size=args.chunk_size,
        )

        data_path = out_dir / f"ising1d_open_L{args.L}_N{args.num_samples}_{label}.npy"
        meta_path = out_dir / f"ising1d_open_L{args.L}_N{args.num_samples}_{label}.json"
        corr_path = out_dir / f"ising1d_open_L{args.L}_N{args.num_samples}_{label}_corr.npy"

        np.save(data_path, data)

        C = mean_corr(data, max_r=min(args.max_corr_r, args.L - 1))
        np.save(corr_path, C)
        plot_path = out_dir / f"ising1d_open_L{args.L}_N{args.num_samples}_{label}_corr.png"

        plot_corr_check(
            C=C,
            K=K,
            xi=xi,
            out_path=str(plot_path),
        )
        
        meta = {
            "model": "1D ferromagnetic Ising",
            "boundary_condition": "free",
            "Hamiltonian": "H = -K sum_{i=1}^{L-1} s_{i-1} s_{i}",
            "spin_values": [-1, 1],
            "L": args.L,
            "num_samples": args.num_samples,
            "K": float(K),
            "xi_corr_exact": float(xi),
            "seed": args.seed + idx,
            "sampler": "exact bond sampler",
            "data_path": str(data_path),
            "corr_path": str(corr_path),
            "mean_spin": float(data.mean()),
            "mean_abs_magnetization": float(np.abs(data.mean(axis=1)).mean()),
            "p_domain_wall_exact": float(p_dw),
            "domain_wall_density_empirical": domain_wall_density(data),
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print("saved:", data_path)
        print(f"  K = {K:.6f}")
        print(f"  xi_exact = {xi:.3f}")
        print(f"  p_domain_wall_exact = {p_dw:.6f}")
        print(f"  p_domain_wall_empirical = {meta['domain_wall_density_empirical']:.6f}")


if __name__ == "__main__":
    main()
