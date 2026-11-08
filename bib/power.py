"""GPU telemetry sampler (NVML) running in a background process.

Two independent energy measurements are taken for every run:

1. `nvmlDeviceGetTotalEnergyConsumption` (mJ since driver load) read at the
   start and end of the measurement window -> exact board energy for the window.
2. A ~10 Hz time series of power (instantaneous field if the driver exposes it,
   plus NVML's averaged power), SM/memory clocks, temperature, memory used,
   utilization and clock-event (throttle) reasons. The series is used for plots,
   peak memory, and to cross-check (1) by trapezoidal integration.
"""
from __future__ import annotations

import csv
import gzip
import multiprocessing as mp
import time

FIELDS = ["t_mono", "power_inst_w", "power_avg_w", "energy_mj", "sm_mhz", "mem_mhz",
          "temp_c", "mem_used_mib", "util_gpu", "util_mem", "clk_event_reasons"]


def _sampler(path: str, interval: float, stop: mp.Event, ready: mp.Event, index: int):
    import pynvml as nv  # nvidia-ml-py
    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByIndex(index)
    inst_field = getattr(nv, "NVML_FI_DEV_POWER_INSTANT", None)

    def inst_power():
        if inst_field is None:
            return None
        try:
            v = nv.nvmlDeviceGetFieldValues(h, [inst_field])[0]
            if v.nvmlReturn != 0:
                return None
            return v.value.uiVal / 1000.0
        except Exception:
            return None

    def reasons():
        for fn in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            f = getattr(nv, fn, None)
            if f:
                try:
                    return int(f(h))
                except Exception:
                    pass
        return None

    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        ready.set()
        nxt = time.monotonic()
        while not stop.is_set():
            t = time.monotonic()
            try:
                mem = nv.nvmlDeviceGetMemoryInfo(h)
                ut = nv.nvmlDeviceGetUtilizationRates(h)
                row = [
                    round(t, 4),
                    inst_power(),
                    nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
                    nv.nvmlDeviceGetTotalEnergyConsumption(h),
                    nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                    nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                    nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                    mem.used // (1 << 20),
                    ut.gpu,
                    ut.memory,
                    reasons(),
                ]
            except Exception as e:  # never kill the run because of telemetry
                row = [round(t, 4)] + [None] * (len(FIELDS) - 2) + [f"err:{e}"]
            w.writerow(row)
            nxt += interval
            sl = nxt - time.monotonic()
            if sl > 0:
                stop.wait(sl)
            else:
                nxt = time.monotonic()
        # final row after stop, so the series always covers the end of the load window
        try:
            w.writerow([round(time.monotonic(), 4), inst_power(), nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
                        nv.nvmlDeviceGetTotalEnergyConsumption(h), None, None, None, None, None, None, None])
        except Exception:
            pass
    nv.nvmlShutdown()


class PowerSampler:
    def __init__(self, path: str, interval: float = 0.1, index: int = 0):
        self.path = path
        self.interval = interval
        self.index = index
        ctx = mp.get_context("spawn")
        self._stop = ctx.Event()
        self._ready = ctx.Event()
        self._p = ctx.Process(target=_sampler, args=(path, interval, self._stop, self._ready, index), daemon=True)

    def __enter__(self):
        self._p.start()
        self._ready.wait(timeout=30)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._p.join(timeout=30)
        if self._p.is_alive():
            self._p.kill()


def energy_counter_mj(index: int = 0) -> int | None:
    try:
        import pynvml as nv
        nv.nvmlInit()
        v = nv.nvmlDeviceGetTotalEnergyConsumption(nv.nvmlDeviceGetHandleByIndex(index))
        nv.nvmlShutdown()
        return int(v)
    except Exception:
        return None


def gpu_snapshot(index: int = 0) -> dict:
    """Volatile GPU state recorded at the start/end of every run."""
    try:
        import pynvml as nv
        nv.nvmlInit()
        h = nv.nvmlDeviceGetHandleByIndex(index)
        snap = {
            "temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
            "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
            "mem_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
            "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
            "power_limit_w": nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
            "mem_used_mib": nv.nvmlDeviceGetMemoryInfo(h).used // (1 << 20),
        }
        try:
            cur, pend = nv.nvmlDeviceGetEccMode(h)
            snap["ecc_current"], snap["ecc_pending"] = int(cur), int(pend)
        except Exception as e:
            snap["ecc"] = f"n/a: {e}"
        nv.nvmlShutdown()
        return snap
    except Exception as e:
        return {"error": str(e)}


def read_power(path: str) -> list[dict]:
    out = []
    with gzip.open(path, "rt") as f:
        for row in csv.DictReader(f):
            d = {}
            for k, v in row.items():
                if v in ("", "None", None):
                    d[k] = None
                else:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
            out.append(d)
    return out
