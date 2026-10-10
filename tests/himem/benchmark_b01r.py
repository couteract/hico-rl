#!/usr/bin/env python3
"""
B01R Performance Benchmark Suite

Requirements:
- CUDA synchronization before/after timing
- Proper warmup runs
- Complete metrics (mean, std, p50, p95, p99)
- Separate CPU/GPU benchmarks
"""

import time
import statistics
import torch
import numpy as np
from typing import Dict, List
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot.memory.himem.config import HiMemConfig, TokenSchema
from lerobot.memory.himem.encoder import HiMemEncoder


class PerformanceBenchmark:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.warmup_runs = 10
        self.benchmark_runs = 100

    def cuda_sync(self):
        """Synchronize CUDA operations if on GPU."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def run_warmup(self, fn, *args, **kwargs):
        """Run warmup iterations."""
        print(f"  Warming up ({self.warmup_runs} runs)...")
        for _ in range(self.warmup_runs):
            fn(*args, **kwargs)
        self.cuda_sync()

    def benchmark_function(self, fn, *args, **kwargs) -> Dict[str, float]:
        """
        Benchmark a function with proper CUDA sync and statistics.

        Returns:
            Dict with mean, std, p50, p95, p99 in milliseconds
        """
        self.run_warmup(fn, *args, **kwargs)

        timings = []
        print(f"  Benchmarking ({self.benchmark_runs} runs)...")

        for _ in range(self.benchmark_runs):
            self.cuda_sync()
            start = time.perf_counter()

            fn(*args, **kwargs)

            self.cuda_sync()
            end = time.perf_counter()

            timings.append((end - start) * 1000)  # Convert to ms

        return {
            "mean_ms": statistics.mean(timings),
            "std_ms": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "p50_ms": np.percentile(timings, 50),
            "p95_ms": np.percentile(timings, 95),
            "p99_ms": np.percentile(timings, 99),
            "min_ms": min(timings),
            "max_ms": max(timings),
            "runs": len(timings)
        }

    def benchmark_encoder_forward(self, config: HiMemConfig, seq_len: int) -> Dict:
        """Benchmark encoder forward pass."""
        encoder = HiMemEncoder(config).to(self.device)
        encoder.eval()

        # Create dummy input
        token_dim = config.token_schema.token_dim
        x = torch.randn(1, seq_len, token_dim, device=self.device)

        with torch.no_grad():
            def forward_fn():
                return encoder(x, return_all=False)

            results = self.benchmark_function(forward_fn)
            results["seq_len"] = seq_len
            results["backend"] = config.backend
            results["memory_dim"] = config.memory_dim

        return results

    def run_full_benchmark_suite(self) -> Dict:
        """Run complete benchmark suite."""
        print(f"\n{'='*60}")
        print(f"Performance Benchmark Suite - Device: {self.device.upper()}")
        print(f"{'='*60}")

        results = {
            "device": self.device,
            "warmup_runs": self.warmup_runs,
            "benchmark_runs": self.benchmark_runs,
            "backends": {}
        }

        # Test configurations
        backends = ["gru_reference"]
        if self.device == "cuda":
            try:
                import mamba_ssm
                backends.append("mamba2")
            except ImportError:
                print("  Warning: mamba_ssm not available, skipping mamba2 benchmarks")

        seq_lengths = [1, 4, 8]  # Rolling window sizes

        for backend in backends:
            print(f"\n--- Backend: {backend} ---")
            results["backends"][backend] = {}

            for seq_len in seq_lengths:
                print(f"\nSequence length: {seq_len}")

                # Create config
                if backend == "mamba2":
                    config = HiMemConfig(
                        backend="mamba2",
                        memory_dim=128,
                        mamba2_d_model=128,
                        mamba2_num_layers=2
                    )
                else:
                    config = HiMemConfig(
                        backend="gru_reference",
                        memory_dim=128,
                        gru_hidden_dim=128,
                        gru_num_layers=1
                    )

                bench_results = self.benchmark_encoder_forward(config, seq_len)
                results["backends"][backend][f"seq_{seq_len}"] = bench_results

                print(f"  Mean: {bench_results['mean_ms']:.3f}ms ± {bench_results['std_ms']:.3f}ms")
                print(f"  P50: {bench_results['p50_ms']:.3f}ms")
                print(f"  P95: {bench_results['p95_ms']:.3f}ms")
                print(f"  P99: {bench_results['p99_ms']:.3f}ms")

        return results


def main():
    import json

    # CPU Benchmarks (always run)
    cpu_bench = PerformanceBenchmark(device="cpu")
    cpu_results = cpu_bench.run_full_benchmark_suite()

    # GPU Benchmarks (conditional)
    gpu_results = None
    if torch.cuda.is_available():
        gpu_bench = PerformanceBenchmark(device="cuda")
        gpu_results = gpu_bench.run_full_benchmark_suite()
    else:
        print("\n" + "="*60)
        print("GPU Benchmarks: SKIPPED (CUDA not available)")
        print("="*60)

    # Save results
    report = {
        "cpu": cpu_results,
        "gpu": gpu_results if gpu_results else {"status": "skipped", "reason": "CUDA not available"}
    }

    output_path = Path(__file__).parent.parent.parent / "B01R_PERFORMANCE_REPORT.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Performance report saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
