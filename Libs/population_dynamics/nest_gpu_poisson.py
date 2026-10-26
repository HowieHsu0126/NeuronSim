"""Bounded native Poisson-driver diagnostics.

Call acquire() in a fresh process owning the native GPU runtime.

Optional configuration:
    finite_window = {"start_ms": ..., "stop_ms": ...}
    expected_arrival_support = {"first_tick": ..., "last_tick": ...}

Support endpoints are inclusive arrival ticks supplied by the caller. No onset
or delivery convention is inferred from finite_window.

analyze() accepts the acquisition directory or its counts.npz file. SciPy is
imported only by analyze(), which belongs on the CPU analysis path.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXPECTED_SOURCE_REVISION = "74fc021cad2b4bd5c011c390b3f95a302b663483"
FAMILY_ALPHA = 0.05
F32_U = 2.0 ** -24
# Conservative additive allowance also covers flushing FP32 subnormals.
F32_TINY = 2.0 ** -126


def native_rng_identity(library_path):
    """Read an optional variant identity without inferring it from a filename."""
    library = ctypes.CDLL(str(library_path))
    try:
        function = library.NeuronSimPoissonEdgeRngName
    except AttributeError:
        return "not_reported_by_native_library"
    function.argtypes = []
    function.restype = ctypes.c_char_p
    value = function()
    if value is None:
        raise RuntimeError("native Poisson RNG identity is null")
    return value.decode("utf-8")


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _config(config):
    result = dict(config)
    for key in ("dt_ms", "tau_syn_ms", "delay_ms", "duration_ms"):
        result[key] = float(config[key])
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("targets_per_rate", "seed"):
        value = config[key]
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{key} must be an integer")
        result[key] = int(value)
    if result["targets_per_rate"] < 1 or result["seed"] < 0:
        raise ValueError("invalid target count or seed")

    result["rates_hz"] = [float(value) for value in config["rates_hz"]]
    if not result["rates_hz"] or any(
        not math.isfinite(value) or value < 0 for value in result["rates_hz"]
    ):
        raise ValueError("rates_hz must contain finite nonnegative rates")

    window = config.get("finite_window")
    if window is not None:
        start, stop = float(window["start_ms"]), float(window["stop_ms"])
        if not (math.isfinite(start) and math.isfinite(stop) and 0 <= start <= stop):
            raise ValueError("invalid finite_window")
        result["finite_window"] = {"start_ms": start, "stop_ms": stop}

    support = config.get("expected_arrival_support")
    if support is not None:
        first, last = support["first_tick"], support["last_tick"]
        if (
            isinstance(first, bool) or isinstance(last, bool)
            or int(first) != first or int(last) != last
            or first < 0 or last < first
        ):
            raise ValueError("support requires ordered nonnegative integer ticks")
        result["expected_arrival_support"] = {
            "first_tick": int(first), "last_tick": int(last)
        }
    return result


def _clock_ticks(times, dt_ms):
    if not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("invalid recording times")
    ticks = np.rint(times / dt_ms).astype(np.int64)
    # Allow rounding of both the FP32 step size and FP32 recorded timestamp.
    radius = (
        0.5 * np.abs(np.spacing(times.astype(np.float32))).astype(np.float64)
        + ticks * 0.5 * float(np.spacing(np.float32(dt_ms)))
        + 4 * np.finfo(np.float64).eps * np.maximum(times, dt_ms)
    )
    if (
        np.any(radius >= dt_ms / 2)
        or np.any(np.abs(times - ticks * dt_ms) > radius)
        or np.any(np.diff(times) <= 0)
        or np.any(np.diff(ticks) != 1)
    ):
        raise ValueError("recordings do not identify consecutive native ticks")
    return ticks


def _decode(currents, decay):
    """Invert FP32 decay and positive unit-weight accumulation conservatively."""
    if (
        not np.all(np.isfinite(currents))
        or np.any(currents < 0)
        or not np.array_equal(
            currents, currents.astype(np.float32).astype(np.float64)
        )
        or not np.all(np.isfinite(decay))
        or np.any(decay < F32_TINY)
        or np.any(decay > 1)
        or not np.array_equal(decay, decay.astype(np.float32).astype(np.float64))
    ):
        raise ValueError("decoder requires finite, nonnegative FP32 states")

    previous, current = currents[:-1], currents[1:]
    down = lambda value: np.nextafter(value, -np.inf)
    up = lambda value: np.nextafter(value, np.inf)

    # current = fl32(decay * accumulated_current).
    accumulated_lo = np.maximum(
        0.0, down(down(current - F32_TINY) / (decay * (1 + F32_U)))
    )
    accumulated_hi = up(
        up(current + F32_TINY) / (decay * (1 - F32_U))
    )

    # Cover either one aggregated addition or separate additions of unit events.
    # Below 2**23, each positive unit addition advances an FP32 accumulator
    # by at least 0.5. This bounds the possible number of additions.
    if np.any(previous >= 2**23) or np.any(accumulated_hi >= 2**23):
        raise ValueError("current too large for unambiguous unit-count decoding")
    operations = np.ceil(2 * accumulated_hi) + 2
    nu = up(operations * F32_U)
    gamma = up(nu / down(1 - nu))
    if np.any(gamma >= 1):
        raise ValueError("accumulation rounding bound is not invertible")
    additive = up(operations * F32_TINY / down(1 - nu))

    exact_sum_lo = down(
        down(accumulated_lo - additive) / up(1 + gamma)
    )
    exact_sum_hi = up(
        up(accumulated_hi + additive) / down(1 - gamma)
    )
    lower = down(exact_sum_lo - previous)
    upper = up(exact_sum_hi - previous)

    first = np.maximum(0.0, np.ceil(lower))
    last = np.floor(upper)
    valid = (first == last) & (first < 2**63)
    if not np.all(valid):
        examples = [
            {
                "row": int(row), "target": int(target),
                "lower": float(lower[row, target]),
                "upper": float(upper[row, target]),
            }
            for row, target in np.argwhere(~valid)[:8]
        ]
        raise ValueError(f"noninteger or ambiguous count intervals: {examples}")

    counts = first.astype(np.int64)
    reconstruction = current / decay - previous
    diagnostics = {
        "method": "outward FP32 multiplication/addition interval",
        "weight_pa": 1.0,
        "unit_roundoff": F32_U,
        "additive_rounding_allowance": F32_TINY,
        "max_interval_width": float(np.max(upper - lower)),
        "max_reconstruction_residual": float(
            np.max(np.abs(reconstruction - counts))
        ),
    }
    return counts, diagnostics


def acquire(config, output):
    """Acquire raw GPU currents and uniquely decoded arrival counts."""
    cfg = _config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("counts.npz", "acquisition.json"):
        if (output / name).exists():
            raise FileExistsError(output / name)

    payload = {}
    metadata = {
        "schema_version": 1,
        "status": "FAILED",
        "config": cfg,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_reference_qualified": False,
        "expected_source_revision": EXPECTED_SOURCE_REVISION,
        "runtime_revision_verified": False,
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "native_library": os.environ.get("NESTGPU_LIB"),
        },
    }
    try:
        import nestgpu as ng

        ng.SetTimeResolution(cfg["dt_ms"])
        ng.SetRandomSeed(cfg["seed"])
        parameters = {
            "tau_m": 10.0, "C_m": 250.0, "E_L": -65.0,
            "V_m_rel": 0.0, "V_reset_rel": 0.0, "Theta_rel": 1.0e6,
            "I_e": 0.0, "t_ref": 0.0, "den_delay": 0.0,
            "tau_ex": cfg["tau_syn_ms"], "tau_in": cfg["tau_syn_ms"],
            "I_syn_ex": 0.0, "I_syn_in": 0.0,
        }
        groups, generators, target_ids, generator_index = [], [], [], []
        for index, rate in enumerate(cfg["rates_hz"]):
            generator = ng.Create("poisson_generator", 1)
            group = ng.Create("iaf_psc_exp", cfg["targets_per_rate"])
            generator_parameters = {"rate": rate}
            if cfg.get("finite_window") is not None:
                generator_parameters.update(
                    start=cfg["finite_window"]["start_ms"],
                    stop=cfg["finite_window"]["stop_ms"],
                )
            ng.SetStatus(generator, generator_parameters)
            ng.SetStatus(group, parameters)
            ng.ActivateRecSpikeTimes(group, 1)
            ng.Connect(
                generator, group, {"rule": "all_to_all"},
                {"weight": 1.0, "delay": cfg["delay_ms"], "receptor": 0},
            )
            groups.append(group)
            generators.append(generator)
            target_ids.extend(int(group[k]) for k in range(cfg["targets_per_rate"]))
            generator_index.extend([index] * cfg["targets_per_rate"])

        record = ng.CreateRecord(
            "", ["I_syn_ex"] * len(target_ids), target_ids, [0] * len(target_ids)
        )
        ng.Simulate(cfg["duration_ms"])
        raw = np.asarray(ng.GetRecordData(record), dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(target_ids) + 1 or len(raw) < 2:
            raise ValueError("unexpected current recorder shape")

        times, currents = raw[:, 0], raw[:, 1:]
        payload.update(times_ms=times, currents_pa=currents)
        ticks = _clock_ticks(times, cfg["dt_ms"])
        if ticks[0] != 0 or ticks[-1] != round(cfg["duration_ms"] / cfg["dt_ms"]):
            raise ValueError("fresh-process recording interval was not obtained")

        target_status = [
            status for group in groups for status in ng.GetStatus(group)
        ]
        decay = np.asarray(
            [status["P11ex"] for status in target_status], dtype=np.float64
        )
        generator_status = [ng.GetStatus(g)[0] for g in generators]
        stored_rates = np.asarray(
            [status["rate"] for status in generator_status], dtype=np.float32
        )
        if decay.shape != (len(target_ids),) or not np.all(np.isfinite(stored_rates)):
            raise ValueError("invalid stored native parameters")
        if any(
            len(spikes)
            for group in groups
            for spikes in ng.GetRecSpikeTimes(group)
        ):
            raise ValueError("an isolation target spiked")

        # Reference mean from the source's FP32 h * rate / 1000 expression.
        with np.errstate(over="raise", invalid="raise"):
            means = (
                np.float32(cfg["dt_ms"]) * stored_rates / np.float32(1000.0)
            ).astype(np.float32)
        mapping = np.asarray(generator_index, dtype=np.int64)
        payload.update(
            arrival_ticks=ticks[:-1],
            arrival_times_ms=times[:-1],
            p11ex=decay,
            rates_hz=stored_rates.astype(np.float64),
            lambda_per_target=means[mapping].astype(np.float64),
            generator_index=mapping,
            target_ids=np.asarray(target_ids, dtype=np.int64),
        )
        counts, decoding = _decode(currents, decay)
        payload["counts"] = counts
        metadata.update(
            status="EXECUTED",
            decoding=decoding,
            isolation_parameters=parameters,
            target_spikes_observed=False,
            count_clock="arrival at the left endpoint of each recorded interval",
            mean_definition="FP32(dt_ms) * stored FP32 rate / FP32(1000)",
            native_generator_parameters=[
                {
                    key: float(status[key]) if math.isfinite(float(status[key]))
                    else str(status[key])
                    for key in ("rate", "origin", "start", "stop")
                    if key in status
                }
                for status in generator_status
            ],
        )
    except Exception as error:
        metadata["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        metadata["ended_utc"] = datetime.now(timezone.utc).isoformat()
        if payload:
            with (output / "counts.npz").open("xb") as stream:
                np.savez_compressed(stream, **payload)
        _write_json(output / "acquisition.json", metadata)
    return output


def analyze(path):
    """Run exact law checks; non-rejection is not a qualification claim."""
    output = Path(path)
    if output.name == "counts.npz":
        output = output.parent
    if (output / "analysis.json").exists():
        raise FileExistsError(output / "analysis.json")
    metadata = json.loads((output / "acquisition.json").read_text())
    if metadata["status"] != "EXECUTED":
        raise ValueError("acquisition did not produce validated integer counts")
    with np.load(output / "counts.npz", allow_pickle=False) as archive:
        counts = archive["counts"].copy()
        ticks = archive["arrival_ticks"].copy()
        means = archive["lambda_per_target"].copy()
        groups = archive["generator_index"].copy()

    if (
        counts.ndim != 2 or counts.dtype.kind not in "iu"
        or np.any(counts < 0) or counts.shape != (len(ticks), len(means))
        or len(groups) != len(means) or np.any(np.diff(ticks) != 1)
        or not np.all(np.isfinite(means)) or np.any(means < 0)
    ):
        raise ValueError("invalid count archive")

    report = {
        "scientific_reference_qualified": False,
        "interpretation": "Non-rejection does not prove a Poisson law or independence.",
        "family_alpha": FAMILY_ALPHA,
        "observed_arrival_ticks": [int(ticks[0]), int(ticks[-1])],
        "expected_arrival_support": metadata["config"].get(
            "expected_arrival_support"
        ),
        "tests": [],
    }
    support = report["expected_arrival_support"]
    if support is None:
        selected = counts
        report.update(
            law_test_outcome="NOT_TESTED",
            reason="Expected arrival support is unspecified; onset is not inferred.",
            outside_support=None,
        )
    else:
        inside = (
            (ticks >= support["first_tick"]) & (ticks <= support["last_tick"])
        )
        selected = counts[inside]
        outside = counts[~inside]
        report["outside_support"] = {
            "bin_count": int((~inside).sum()),
            "nonzero_target_bins": int(np.count_nonzero(outside)),
            "event_count": sum(int(value) for value in outside.flat),
        }
        report["analyzed_arrival_ticks"] = (
            [int(ticks[inside][0]), int(ticks[inside][-1])]
            if np.any(inside) else None
        )

    n, targets = selected.shape
    report["sample_count_per_target"] = n
    report["target_moments"] = [
        {
            "target": target,
            "generator": int(groups[target]),
            "expected_mean": float(means[target]),
            "count_mean": float(np.mean(selected[:, target])) if n else None,
            "count_variance": (
                float(np.var(selected[:, target], ddof=1)) if n > 1 else None
            ),
        }
        for target in range(targets)
    ]
    if support is None or n == 0:
        if support is not None:
            report.update(
                law_test_outcome=(
                    "REJECTED_SUPPORT"
                    if report["outside_support"]["event_count"] else "NOT_TESTED"
                ),
                reason="No decoded bins overlap the explicit expected support.",
            )
        _write_json(output / "analysis.json", report)
        return report

    from scipy.stats import binomtest, poisson

    tests = report["tests"]
    zero = selected == 0
    p_zero = np.exp(-means)
    p_nonzero = -np.expm1(-means)

    def add_binomial(kind, target_indices, successes, trials, probability):
        tests.append({
            "kind": kind, "targets": target_indices,
            "successes": int(successes), "trials": int(trials),
            "null_probability": float(probability),
        })

    for target in range(targets):
        total = sum(int(value) for value in selected[:, target])
        expected_total = n * float(means[target])
        if not math.isfinite(expected_total):
            raise ValueError("nonfinite expected total count")
        # Inclusive, doubled smaller tail: exact conservative two-sided test.
        p_value = min(
            1.0,
            2.0 * min(
                poisson.cdf(total, expected_total),
                poisson.sf(total - 1, expected_total),
            ),
        )
        tests.append({
            "kind": "poisson_total", "targets": [target],
            "observed_total": total, "expected_total": expected_total,
            "p_value": float(p_value),
        })
        add_binomial(
            "zero_count", [target], zero[:, target].sum(), n, p_zero[target]
        )
        add_binomial(
            "multiplicity", [target], (selected[:, target] >= 2).sum(), n,
            poisson.sf(1, float(means[target])),
        )

    for left in range(targets):
        for right in range(left + 1, targets):
            probability = (
                p_zero[left] * p_nonzero[right]
                + p_zero[right] * p_nonzero[left]
            )
            add_binomial(
                "same_generator_zero_disagreement"
                if groups[left] == groups[right]
                else "cross_generator_zero_disagreement",
                [left, right],
                np.count_nonzero(zero[:, left] != zero[:, right]),
                n, probability,
            )

    pairs = n // 2
    if pairs:
        for target in range(targets):
            add_binomial(
                "serial_zero_disagreement", [target],
                np.count_nonzero(
                    zero[:2 * pairs:2, target] != zero[1:2 * pairs:2, target]
                ),
                pairs, 2 * p_zero[target] * p_nonzero[target],
            )

    per_test_alpha = FAMILY_ALPHA / len(tests)
    for test in tests:
        if test["kind"] != "poisson_total":
            result = binomtest(
                test["successes"], test["trials"], test["null_probability"],
                alternative="two-sided",
            )
            interval = result.proportion_ci(
                confidence_level=1 - per_test_alpha, method="exact"
            )
            test.update(
                p_value=float(result.pvalue),
                observed_probability=test["successes"] / test["trials"],
                exact_binomial_ci=[float(interval.low), float(interval.high)],
                ci_confidence_level=1 - per_test_alpha,
                expected_successes=test["trials"] * test["null_probability"],
            )
        test["reject"] = test["p_value"] <= per_test_alpha

    support_failure = report["outside_support"]["event_count"] != 0
    report.update(
        test_count=len(tests),
        per_test_alpha=per_test_alpha,
        rejection_count=sum(test["reject"] for test in tests),
        support_violation=support_failure,
        law_test_outcome=(
            "REJECTED"
            if support_failure or any(test["reject"] for test in tests)
            else "NOT_REJECTED"
        ),
    )
    _write_json(output / "analysis.json", report)
    return report


def main(argv=None):
    """CLI for native acquisition and CPU-only analysis."""
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("acquire", "analyze"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    supplied = yaml.safe_load(args.config.read_text())
    if not isinstance(supplied, dict):
        parser.error("--config must contain a YAML mapping")
    cfg = _config(supplied)
    if args.mode == "acquire":
        destination = acquire(cfg, args.output)
        print(json.dumps({
            "mode": "acquire", "status": "EXECUTED",
            "output": str(destination), "scientific_reference_qualified": False,
        }))
        return 0

    directory = args.output.parent if args.output.name == "counts.npz" else args.output
    receipt = json.loads((directory / "acquisition.json").read_text())
    if cfg != _config(receipt["config"]):
        raise ValueError("analysis configuration differs from acquisition configuration")
    report = analyze(args.output)
    outcome = report["law_test_outcome"]
    print(json.dumps({
        "mode": "analyze", "law_test_outcome": outcome,
        "output": str(directory / "analysis.json"),
        "scientific_reference_qualified": False,
    }))
    if outcome == "NOT_REJECTED":
        return 0
    return 1 if outcome.startswith("REJECTED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
