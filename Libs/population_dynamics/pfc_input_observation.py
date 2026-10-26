"""Observe the pinned PFC source's native NumPy inputs without new RNG draws.

Capture wraps only each native PoissonInput's BinomialFunction implementation.
Audit leaves that implementation untouched. Both save terminal native state and
RNG snapshots for a separate parity comparison; neither mode establishes parity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from types import FunctionType, ModuleType
import weakref

import brian2 as b2
from brian2.input.binomial import BinomialFunction
import numpy as np

from .pfc_factorized_source import SOURCE_ENTRY_SHA256
from .provenance import sha256_file

_STATE_NAMES = ("v", "s_AMPA", "s_GABA", "s_NMDA", "x", "s_NMDA_total",
                "I_stim", "lastspike", "not_refractory")


def _write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _save_npz(path: Path, values):
    with path.open("xb") as stream:
        np.savez(stream, **values)


class _DrawRecorder:
    """One complete native population vector per clock tick, in source order."""

    def __init__(self, path: Path, units: int, ticks: int, dt_seconds: float, clock, on_first_draw=None):
        self.path, self.units, self.ticks = path, units, ticks
        self.dt_seconds, self.clock = dt_seconds, clock
        self.next_tick, self.total_events = 0, 0
        self.on_first_draw = on_first_draw
        self.values = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.uint16, shape=(ticks, units))

    def validate_indices(self, indices):
        if isinstance(indices, (int, np.integer)) and not isinstance(indices, (bool, np.bool_)):
            valid = int(indices) == self.units
        else:
            array = np.asarray(indices)
            valid = (array.shape == (self.units,) and array.dtype.kind in "iu"
                     and np.array_equal(array, np.arange(self.units)))
        if not valid:
            raise ValueError("PFC_INPUT_VECTORIZATION_ORDER_CHANGED")

    def record(self, draws):
        if self.values is None or self.next_tick >= self.ticks:
            raise ValueError("PFC_INPUT_EXTRA_OR_CLOSED_DRAW")
        tick = float(self.clock.t / b2.second) / self.dt_seconds
        dt = float(self.clock.dt / b2.second)
        if (not np.isclose(tick, self.next_tick, rtol=0, atol=1e-8)
                or not np.isclose(dt, self.dt_seconds, rtol=0, atol=1e-15)):
            raise ValueError("PFC_INPUT_CLOCK_OR_DRAW_ORDER_CHANGED")
        values = np.asarray(draws)
        if (values.shape != (self.units,) or values.dtype.kind not in "iu"
                or np.any(values < 0) or np.any(values > 1000)):
            raise ValueError("PFC_INPUT_NATIVE_BINOMIAL_DRAWS_INVALID")
        if self.next_tick == 0 and self.on_first_draw is not None:
            self.on_first_draw()
        self.values[self.next_tick] = values
        self.total_events += int(values.sum(dtype=np.uint64))
        self.next_tick += 1

    def validate_complete(self):
        if self.next_tick != self.ticks or self.values is None:
            raise ValueError("PFC_INPUT_DRAW_CAPTURE_INCOMPLETE")

    def close(self):
        if self.values is not None:
            self.values.flush()
            self.values._mmap.close()
            self.values = None
        self.clock = None
        self.on_first_draw = None


def _observe_sampler(original, recorder: _DrawRecorder):
    """Call the native sampler once with the original argument and return object."""
    def observed(vectorisation_idx):
        recorder.validate_indices(vectorisation_idx)
        result = original(vectorisation_idx)
        recorder.record(result)
        return result
    return observed


def _schedule(obj):
    return dict(name=obj.name, when=obj.when, order=int(obj.order),
                dt_seconds=float(obj.clock.dt / b2.second),
                clock_t_seconds=float(obj.clock.t / b2.second))


def _generated_code(group):
    runners = [("state_updater", group.state_updater)]
    if "spike" not in group.thresholder or "spike" not in group.resetter:
        raise ValueError("PFC_INPUT_NATIVE_SPIKE_RUNNERS_MISSING")
    runners.extend(("threshold:" + key, value) for key, value in group.thresholder.items())
    runners.extend(("reset:" + key, value) for key, value in group.resetter.items())
    output = []
    for kind, runner in runners:
        codeobj = getattr(runner, "codeobj", None)
        if codeobj is None:
            raise ValueError("PFC_INPUT_NATIVE_GENERATED_CODE_MISSING")
        output.append(dict(kind=kind, **_schedule(runner), code_object_name=codeobj.name,
                           code_object_class=type(codeobj).__name__,
                           abstract_code=str(runner.abstract_code),
                           generated_code=str(codeobj.code)))
    return output


def _state_snapshot(group, monitor, population, units, offset):
    indices = np.asarray(group.variables["i"].get_value()).copy()
    counts = np.asarray(monitor.count[:]).copy()
    if (indices.dtype.kind not in "iu" or indices.shape != (units,)
            or not np.array_equal(indices, np.arange(units))
            or counts.dtype.kind not in "iu" or counts.shape != (units,)
            or np.any(counts < 0) or int(counts.sum()) != len(monitor.i)):
        raise ValueError("PFC_INPUT_NATIVE_UNIT_ID_OR_COUNT_MISMATCH")
    arrays = {population + "__source_indices": indices,
              population + "__archive_indices": indices + offset,
              population + "__spike_counts": counts}
    metadata = []
    dimensions = {"v": (b2.volt.dim, "V"), "I_stim": (b2.amp.dim, "A"),
                  "lastspike": (b2.second.dim, "s")}
    for name in _STATE_NAMES:
        if name not in group.variables:
            continue
        variable = group.variables[name]
        dimension, unit = dimensions.get(name, (b2.get_dimensions(1), "1"))
        values = np.asarray(variable.get_value()).copy()
        if variable.dim != dimension or values.shape != (units,) or values.dtype.kind not in "biuf":
            raise ValueError("PFC_INPUT_NATIVE_STATE_SCHEMA_CHANGED:" + population + ":" + name)
        # Brian may use an infinite lastspike sentinel for never-spiking cells.
        if np.isnan(values).any() or (name != "lastspike" and not np.isfinite(values).all()):
            raise ValueError("PFC_INPUT_NATIVE_STATE_NONFINITE:" + population + ":" + name)
        key = population + "__" + name
        arrays[key] = values
        metadata.append(dict(name=name, array_key=key, units=unit,
                             brian_dimension=str(variable.dim), dtype=str(values.dtype),
                             storage="raw_SI_or_dimensionless", nonfinite_count=int((~np.isfinite(values)).sum())))
    if not {"v", "s_AMPA", "s_GABA"}.issubset({row["name"] for row in metadata}):
        raise ValueError("PFC_INPUT_REQUIRED_NATIVE_STATE_MISSING")
    return arrays, metadata


class NativeInputObserver:
    """Optional per-execution observer; a missing receipt means incomplete data.

    Install on a freshly loaded source module. Call begin_case before fresh Brian
    scope/setup and finish_case immediately after simulate_wm returns. close is
    mandatory on failure; it restores the module factory and retains incomplete
    captures. No native PoissonInput or target group is retained across cases.
    """

    def __init__(self, output: Path, mode="capture"):
        if mode not in ("capture", "audit"):
            raise ValueError("PFC_INPUT_OBSERVATION_MODE_INVALID")
        self.output, self.mode = Path(output), mode
        self.output.mkdir(parents=True, exist_ok=False)
        self._module, self._original, self._factory = None, None, None
        self._active, self._providers, self._cases = None, [], []
        self._closed, self._finalized = False, False

    def install(self, module: ModuleType):
        if self._module is not None or self._closed:
            raise ValueError("PFC_INPUT_OBSERVER_ALREADY_INSTALLED_OR_CLOSED")
        if (not isinstance(module, ModuleType) or not isinstance(getattr(module, "__file__", None), str)
                or hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE_ENTRY_SHA256):
            raise ValueError("PFC_INPUT_OBSERVER_REQUIRES_PINNED_SOURCE")
        function = getattr(module, "simulate_wm", None)
        if not isinstance(function, FunctionType) or function.__globals__ is not vars(module):
            raise ValueError("PFC_INPUT_OBSERVER_SOURCE_FUNCTION_UNSUPPORTED")
        if b2.__version__ != "2.5.4" or module.PoissonInput is not b2.PoissonInput:
            raise ValueError("PFC_INPUT_OBSERVER_REQUIRES_NATIVE_BRIAN_2_5_4")
        self._module, self._original = weakref.ref(module), module.PoissonInput
        reference = weakref.ref(self)
        def construct(*args, **kwargs):
            observer = reference()
            if observer is None:
                raise RuntimeError("PFC_INPUT_OBSERVER_NO_LONGER_AVAILABLE")
            return observer._construct(*args, **kwargs)
        self._factory = construct
        module.PoissonInput = construct

    def begin_case(self, case, config):
        if self._module is None or self._closed or self._active is not None:
            raise ValueError("PFC_INPUT_OBSERVER_CASE_STATE_INVALID")
        case_id = case["case_id"]
        if (not isinstance(case_id, str) or Path(case_id).name != case_id
                or case_id in ("", ".", "..") or any(row["case_id"] == case_id for row in self._cases)):
            raise ValueError("PFC_INPUT_OBSERVER_CASE_ID_INVALID")
        sizes = {"E": config["excitatory_neurons"], "I": config["inhibitory_neurons"]}
        dt, duration = config["dt_ms"], config["duration_ms"]
        if (any(type(size) is not int or size <= 0 for size in sizes.values())
                or any(type(value) not in (int, float) or not np.isfinite(value) or value <= 0
                       for value in (dt, duration))):
            raise ValueError("PFC_INPUT_OBSERVER_CLOCK_OR_SIZE_INVALID")
        ticks = int(round(duration / dt))
        if ticks <= 0 or not np.isclose(ticks * dt, duration, rtol=0, atol=1e-9):
            raise ValueError("PFC_INPUT_OBSERVER_DURATION_OFF_CLOCK")
        directory = self.output / case_id
        directory.mkdir(exist_ok=False)
        self._active = dict(case_id=case_id, case=json.loads(json.dumps(case, allow_nan=False)),
                            sizes=sizes, dt_seconds=dt / 1000, ticks=ticks, directory=directory,
                            native_schedule=None)
        self._providers = []

    def _capture_schedule(self):
        if self._active["native_schedule"] is not None:
            return
        objects = b2.magic_network.sorted_objects
        if not objects:
            raise ValueError("PFC_INPUT_ACTIVE_NATIVE_SCHEDULE_UNAVAILABLE")
        rows = []
        for obj in objects:
            function = getattr(obj, "function", None)
            rows.append(dict(**_schedule(obj), component_class=type(obj).__name__,
                             active=bool(obj.active), function_name=getattr(function, "__name__", None)))
        self._active["native_schedule"] = rows

    def _construct(self, target, target_var, N, rate, weight, **kwargs):
        active = self._active
        if active is None or self._closed or len(self._providers) >= 2:
            raise ValueError("PFC_INPUT_NATIVE_PROVIDER_ORDER_CHANGED")
        if b2.__version__ != "2.5.4" or b2.prefs.codegen.target != "numpy":
            raise ValueError("PFC_INPUT_OBSERVER_REQUIRES_BRIAN_2_5_4_NUMPY")
        population = ("I", "E")[len(self._providers)]
        units = active["sizes"][population]
        native_rate = float(rate / b2.Hz)
        dt = float(target.clock.dt / b2.second)
        if (target_var != "s_AMPA" or type(N) is not int or N != 1000
                or native_rate != 1.4 or np.ndim(weight) != 0 or float(weight) != 1.0
                # Brian resets the fresh magic-network clock in run preparation,
                # after PoissonInput construction. Check tick zero at first draw.
                or len(target) != units
                or not np.isclose(dt, active["dt_seconds"], rtol=0, atol=1e-15)):
            raise ValueError("PFC_INPUT_NATIVE_PROVIDER_PARAMETERS_CHANGED")
        probability = native_rate * dt
        if not 0 <= probability <= 1 or (N * probability > 5 and N * (1 - probability) > 5):
            raise ValueError("PFC_INPUT_NORMAL_APPROXIMATION_UNSUPPORTED")
        instance = self._original(target, target_var, N, rate, weight, **kwargs)
        functions = [value for value in instance.variables.values() if isinstance(value, BinomialFunction)]
        if len(functions) != 1:
            raise ValueError("PFC_INPUT_NATIVE_BINOMIAL_PROVIDER_CHANGED")
        function = functions[0]
        implementation = function.implementations["numpy"]
        original = implementation.get_code(instance)
        if not callable(original):
            raise ValueError("PFC_INPUT_NATIVE_NUMPY_SAMPLER_NOT_CALLABLE")
        metadata = dict(population=population, construction_index=len(self._providers), units=units,
                        target_name=target.name, target_variable=target_var, N=N,
                        rate_Hz=native_rate, weight_dimensionless=float(weight), probability=probability,
                        native_sampler="Brian2.5.4 NumPy BinomialFunction (exact binomial branch)",
                        native_function_name=function.name, implementation_name=implementation.name,
                        expected_sample_rows=active["ticks"], source_unit_order="ascending_zero_based_population_index",
                        **_schedule(instance))
        recorder = None
        if self.mode == "capture":
            recorder = _DrawRecorder(active["directory"] / (population + "_external_counts.incomplete.npy"),
                                     units, active["ticks"], dt, target.clock, self._capture_schedule)
            try:
                function.implementations.add_implementation(
                    "numpy", _observe_sampler(original, recorder),
                    namespace=implementation.get_namespace(instance), dependencies=implementation.dependencies,
                    name=implementation.name, availability_check=implementation.availability_check,
                    compiler_kwds=implementation.compiler_kwds)
            except BaseException:
                recorder.close()
                raise
        self._providers.append(dict(metadata=metadata, target=weakref.ref(target), recorder=recorder))
        return instance

    def finish_case(self, case, result):
        active = self._active
        if active is None or case != active["case"] or len(self._providers) != 2:
            raise ValueError("PFC_INPUT_OBSERVER_CASE_OR_PROVIDER_INCOMPLETE")
        rng = np.random.get_state()
        python_rng = random.getstate()
        if (rng[0] != "MT19937" or rng[1].shape != (624,) or rng[1].dtype != np.uint32
                or not 0 <= rng[2] <= 624 or rng[3] not in (0, 1) or not np.isfinite(rng[4])):
            raise ValueError("PFC_INPUT_NATIVE_NUMPY_RNG_SCHEMA_CHANGED")
        arrays, state_metadata, codes, providers = {}, {}, {}, []
        for provider, monitor in zip(self._providers, (result[5], result[1])):
            metadata, recorder = dict(provider["metadata"]), provider["recorder"]
            population, group = metadata["population"], provider["target"]()
            if (group is None or monitor.source.variables is not group.variables
                    or not np.isclose(float(group.clock.t / b2.second),
                                      active["ticks"] * active["dt_seconds"], rtol=0, atol=1e-12)):
                raise ValueError("PFC_INPUT_NATIVE_TARGET_OR_TERMINAL_CLOCK_CHANGED")
            if recorder is not None:
                recorder.validate_complete()
            snapshot, fields = _state_snapshot(group, monitor, population, metadata["units"],
                                               0 if population == "E" else active["sizes"]["E"])
            arrays.update(snapshot)
            state_metadata[population] = fields
            codes[population] = _generated_code(group)
            metadata.update(observed_sampler_calls=recorder.next_tick if recorder else None,
                            captured_external_events=recorder.total_events if recorder else None,
                            terminal_clock_seconds=float(group.clock.t / b2.second))
            providers.append(metadata)
        directory = active["directory"]
        _save_npz(directory / "terminal_state.npz", arrays)
        _save_npz(directory / "numpy_random_state.npz", dict(
            algorithm=np.array(rng[0]), key=rng[1], position=np.array(rng[2], dtype=np.int64),
            has_gauss=np.array(rng[3], dtype=np.int8), cached_gaussian=np.array(rng[4], dtype=np.float64)))
        _write_json(directory / "python_random_state.json", {"state": python_rng})
        _write_json(directory / "generated_code.json", dict(
            populations=codes, native_network_schedule=active["native_schedule"],
            full_network_operation_schedule=("captured_during_first_native_input_draw" if active["native_schedule"]
                                             else "not_observed_in_audit_mode")))
        for provider, metadata in zip(self._providers, providers):
            recorder = provider["recorder"]
            if recorder is not None:
                recorder.close()
                completed = directory / (metadata["population"] + "_external_counts.npy")
                recorder.path.rename(completed)
                metadata.update(counts_file=str(completed.relative_to(self.output)),
                                counts_sha256=sha256_file(completed), counts_dtype="uint16",
                                counts_shape=[active["ticks"], metadata["units"]])
        summary = dict(case_id=active["case_id"], mode=self.mode)
        for key, filename in (("terminal_state", "terminal_state.npz"),
                              ("numpy_rng", "numpy_random_state.npz"),
                              ("python_rng", "python_random_state.json"),
                              ("generated_code", "generated_code.json")):
            path = directory / filename
            summary[key + "_file"] = str(path.relative_to(self.output))
            summary[key + "_sha256"] = sha256_file(path)
        receipt = dict(summary, status="NATIVE_INPUT_OBSERVATION_CASE_COMPLETE", case=active["case"],
                       providers=providers, native_state_fields=state_metadata,
                       source_entry_sha256=SOURCE_ENTRY_SHA256, source_population_archive_order=["E", "I"],
                       source_provider_construction_order=["I", "E"],
                       full_network_operation_schedule=("captured_during_first_native_input_draw" if active["native_schedule"]
                                                        else "not_observed_in_audit_mode"),
                       terminal_snapshot_is_complete_restart_state=False,
                       gpu_parity_qualified=False, scientific_fidelity_qualified=False, production_episodes=0)
        _write_json(directory / "observation.json", receipt)
        summary.update(receipt_file=str((directory / "observation.json").relative_to(self.output)),
                       receipt_sha256=sha256_file(directory / "observation.json"))
        self._cases.append(summary)
        self._providers, self._active = [], None
        return summary

    def close(self):
        try:
            for provider in self._providers:
                if provider["recorder"] is not None:
                    provider["recorder"].close()
        finally:
            # Restore the source even if flushing an incomplete capture fails.
            self._providers = []
            module = self._module() if self._module is not None else None
            if module is not None and module.PoissonInput is self._factory:
                module.PoissonInput = self._original
            self._closed = True

    def finalize(self):
        if self._active is not None or not self._cases or self._finalized:
            raise ValueError("PFC_INPUT_OBSERVATION_NOT_COMPLETE_OR_ALREADY_FINALIZED")
        self.close()
        report = dict(status="NATIVE_SOURCE_INPUT_OBSERVATION_COMPLETE", mode=self.mode,
                      source_entry_sha256=SOURCE_ENTRY_SHA256, brian2_version=b2.__version__,
                      numpy_version=np.__version__, cases=self._cases,
                      draw_capture_performed=self.mode == "capture", source_inputs_replayed=False,
                      gpu_parity_qualified=False, scientific_fidelity_qualified=False, production_episodes=0)
        _write_json(self.output / "report.json", report)
        self._finalized = True
        return report
