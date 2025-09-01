"""Minimal runtime smoke test for core deps (torch CPU + msprime)."""

import msprime
import torch


def main() -> None:
    print("Torch version:", torch.__version__)
    x = torch.randn(16, 32)
    lin = torch.nn.Linear(32, 8)
    out = lin(x).mean()
    out.backward()
    print("Torch basic forward/backward OK")
    ts = msprime.sim_ancestry(samples=4, sequence_length=1000, recombination_rate=1e-8)
    mts = msprime.sim_mutations(ts, rate=1e-8)
    print("msprime sites:", mts.num_sites)
    print("Smoke test PASS")


if __name__ == "__main__":
    main()
