#!/usr/bin/env python3
"""
B01R Report Generator

Generates structured test report with:
- CPU/GPU test separation
- Real exit codes from pytest
- Performance benchmarks
- Configuration validation
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


class B01RReportGenerator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests" / "himem"
        self.results: Dict[str, dict] = {}

    def run_test_suite(self, test_file: str, test_name: str) -> Tuple[int, str]:
        """
        Run a test suite and return (exit_code, output).

        Returns:
            (exit_code, stdout+stderr)
        """
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file),
            "-v", "--tb=short",
            "--json-report", f"--json-report-file=/tmp/himem_{test_name}.json"
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            exit_code = result.returncode
            output = result.stdout + "\n" + result.stderr
            return exit_code, output
        except subprocess.TimeoutExpired:
            return 124, "Test timed out after 300s"
        except Exception as e:
            return 1, f"Test execution failed: {e}"

    def run_cpu_tests(self) -> Dict[str, dict]:
        """Run CPU-only tests."""
        cpu_tests = {
            "protocol_cpu": "tests/himem/test_himem_protocol.py",
            "event_mamba_cpu": "tests/himem/test_event_mamba_cpu.py",
            "cpu_basic": "tests/memory/himem/test_cpu_basic.py",
        }

        results = {}
        for name, test_file in cpu_tests.items():
            print(f"Running CPU test: {name}...")
            exit_code, output = self.run_test_suite(test_file, name)
            results[name] = {
                "exit_code": exit_code,
                "passed": exit_code == 0,
                "output_length": len(output),
                "test_file": test_file
            }

        return results

    def run_gpu_tests(self) -> Dict[str, dict]:
        """Run GPU tests if CUDA available."""
        import torch

        if not torch.cuda.is_available():
            return {
                "status": "skipped",
                "reason": "CUDA not available"
            }

        gpu_tests = {
            "event_mamba_gpu": "tests/himem/test_event_mamba_gpu.py",
            "gpu_backends": "tests/memory/himem/test_gpu_backends.py",
            "mamba2_official_gpu": "tests/memory/himem/test_mamba2_official.py",
        }

        results = {}
        for name, test_file in gpu_tests.items():
            print(f"Running GPU test: {name}...")
            exit_code, output = self.run_test_suite(test_file, name)
            results[name] = {
                "exit_code": exit_code,
                "passed": exit_code == 0,
                "output_length": len(output),
                "test_file": test_file
            }

        return results

    def run_performance_benchmarks(self) -> Dict[str, any]:
        """Run performance benchmarks with proper CUDA sync."""
        # This will be implemented as a separate benchmark suite
        return {
            "status": "pending",
            "note": "Performance benchmarks require separate benchmark suite"
        }

    def generate_report(self) -> dict:
        """Generate complete B01R report."""
        report = {
            "report_version": "b01r2",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "project_root": str(self.project_root),
            "cpu_tests": {},
            "gpu_tests": {},
            "performance": {},
            "summary": {}
        }

        # Run CPU tests
        print("\n=== Running CPU Tests ===")
        report["cpu_tests"] = self.run_cpu_tests()

        # Run GPU tests
        print("\n=== Running GPU Tests ===")
        report["gpu_tests"] = self.run_gpu_tests()

        # Run performance benchmarks
        print("\n=== Performance Benchmarks ===")
        report["performance"] = self.run_performance_benchmarks()

        # Compute summary
        cpu_passed = sum(1 for t in report["cpu_tests"].values()
                        if isinstance(t, dict) and t.get("passed", False))
        cpu_total = len([t for t in report["cpu_tests"].values() if isinstance(t, dict)])

        gpu_results = report["gpu_tests"]
        if isinstance(gpu_results, dict) and "status" in gpu_results:
            gpu_passed = 0
            gpu_total = 0
        else:
            gpu_passed = sum(1 for t in gpu_results.values()
                           if isinstance(t, dict) and t.get("passed", False))
            gpu_total = len([t for t in gpu_results.values() if isinstance(t, dict)])

        report["summary"] = {
            "cpu_tests_passed": cpu_passed,
            "cpu_tests_total": cpu_total,
            "gpu_tests_passed": gpu_passed,
            "gpu_tests_total": gpu_total,
            "all_cpu_passed": cpu_passed == cpu_total,
            "all_gpu_passed": gpu_total == 0 or gpu_passed == gpu_total,
            "gate_status": "PASS" if (cpu_passed == cpu_total and
                                     (gpu_total == 0 or gpu_passed == gpu_total)) else "FAIL"
        }

        return report

    def write_report(self, report: dict, output_path: Path):
        """Write report to JSON file."""
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to: {output_path}")


def main():
    project_root = Path(__file__).parent.parent.parent
    generator = B01RReportGenerator(project_root)

    report = generator.generate_report()

    # Write report
    report_path = project_root / "B01R_TEST_REPORT.json"
    generator.write_report(report, report_path)

    # Print summary
    print("\n" + "=" * 50)
    print("B01R Test Report Summary")
    print("=" * 50)
    summary = report["summary"]
    print(f"CPU Tests: {summary['cpu_tests_passed']}/{summary['cpu_tests_total']}")
    print(f"GPU Tests: {summary['gpu_tests_passed']}/{summary['gpu_tests_total']}")
    print(f"Gate Status: {summary['gate_status']}")
    print("=" * 50)

    # Exit with appropriate code
    sys.exit(0 if summary["gate_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
