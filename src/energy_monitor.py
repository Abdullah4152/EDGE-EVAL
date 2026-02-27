"""
energy_monitor.py
-----------------
GPU energy monitoring using pynvml.
Supports both single-GPU and multi-GPU setups.
Used during training, merging, quantization, and inference.
"""

import time
import threading
import statistics

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False


class EnergyMonitor:
    """
    Single-GPU energy monitor (used during single-GPU training and evaluation).
    Polls GPU power at a fixed interval and accumulates Joules.
    """

    def __init__(self, interval: float = 0.5, gpu_index: int = 0):
        self.interval = interval
        self.running = False
        self.total_joules = 0.0
        self.peak_watts = 0.0
        self.min_watts = float("inf")
        self.power_samples = []
        self.duration = 0.0
        self.available = PYNVML_AVAILABLE

        if self.available:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self.available = False
                print("⚠️  pynvml: could not get GPU handle. Energy tracking disabled.")

    def _monitor(self):
        start_time = time.time()
        while self.running and self.available:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                power_w = power_mw / 1000.0
                self.power_samples.append(power_w)
                self.peak_watts = max(self.peak_watts, power_w)
                self.min_watts = min(self.min_watts, power_w)
                self.total_joules += power_w * self.interval
            except Exception:
                pass
            time.sleep(self.interval)
        self.duration = time.time() - start_time

    def start(self):
        if self.available:
            self.running = True
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()

    def stop(self) -> dict:
        self.running = False
        if self.available and hasattr(self, "thread"):
            self.thread.join(timeout=5)

        if not self.available or not self.power_samples:
            return {
                "total_joules": 0.0,
                "avg_watts": 0.0,
                "peak_watts": 0.0,
                "min_watts": 0.0,
                "median_watts": 0.0,
                "std_watts": 0.0,
                "duration": 0.0,
            }

        avg_watts = self.total_joules / self.duration if self.duration > 0 else 0.0
        return {
            "total_joules": self.total_joules,
            "avg_watts": avg_watts,
            "peak_watts": self.peak_watts,
            "min_watts": self.min_watts if self.min_watts != float("inf") else 0.0,
            "median_watts": statistics.median(self.power_samples),
            "std_watts": statistics.stdev(self.power_samples) if len(self.power_samples) > 1 else 0.0,
            "duration": self.duration,
        }


class DualGPUEnergyMonitor:
    """
    Multi-GPU energy monitor. Aggregates power across all detected GPUs.
    Used during dual-GPU training (7B/8B models) and merging/quantization.
    Returns per-GPU breakdowns in addition to totals.
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.running = False
        self.total_joules = 0.0
        self.peak_watts = 0.0
        self.duration = 0.0
        self.gpu_handles = []
        self.gpu_names = []
        self.per_gpu_joules = []
        self.per_gpu_peak_watts = []
        self.available = PYNVML_AVAILABLE

        if self.available:
            try:
                num_gpus = pynvml.nvmlDeviceGetCount()
                for i in range(num_gpus):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
                    self.gpu_names.append(pynvml.nvmlDeviceGetName(handle))
                    self.per_gpu_joules.append(0.0)
                    self.per_gpu_peak_watts.append(0.0)
                print(f"⚡ Energy monitoring enabled for {num_gpus} GPU(s): {self.gpu_names}")
            except Exception as e:
                self.available = False
                print(f"⚠️  pynvml init failed: {e}. Energy tracking disabled.")

    def _monitor(self):
        start_time = time.time()
        while self.running and self.available:
            try:
                total_power = 0.0
                for idx, handle in enumerate(self.gpu_handles):
                    power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    total_power += power_w
                    self.per_gpu_joules[idx] += power_w * self.interval
                    if power_w > self.per_gpu_peak_watts[idx]:
                        self.per_gpu_peak_watts[idx] = power_w
                if total_power > self.peak_watts:
                    self.peak_watts = total_power
                self.total_joules += total_power * self.interval
            except Exception:
                pass
            time.sleep(self.interval)
        self.duration = time.time() - start_time

    def start(self):
        if self.available:
            self.running = True
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.available and hasattr(self, "thread"):
            self.thread.join(timeout=5)
        avg_watts = (self.total_joules / self.duration) if self.duration > 0 else 0.0
        if self.gpu_handles:
            print(
                f"   📊 Energy: {self.total_joules:.2f}J | "
                f"Avg: {avg_watts:.2f}W | Peak: {self.peak_watts:.2f}W | "
                f"{self.duration:.2f}s"
            )
        return (
            self.total_joules,
            avg_watts,
            self.duration,
            list(self.per_gpu_joules),
            list(self.per_gpu_peak_watts),
        )
