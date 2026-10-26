"""Explicit CA3 circuit and stage-noise identities, separate from training seeds.

The opt-in protocol reseeds host noise after setup. This changes the source
coupling between topology/reset draws and external currents. It retains the
source current law, shared uniform stream, common seed of the nine persistent
C++ engines, their distribution caches, and the separate libc cue stream.
Formation changes only the CARLsim constructor seed and retains source host
engines and their coupling to setup draws. Native initial states remain source
resets; a saved graph is not a state restart.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
import struct

from .config import load_config
from .provenance import sha256_file

SCHEMA = "ca3_source_realization_v1"
ENGINE_NAMES = ("pyr_generator", "pyr_generator_recall", "aac_generator",
                "bc_generator", "cck_generator", "bic_generator", "ivy_generator",
                "mfao_generator", "quadD_generator")
# The qualified GCC default engine has modulus 2**31-1. Excluding zero and
# the modulus avoids seeds that collapse onto the engine's default state.
MAX_SEED = 2147483646
FORMATION_MAIN_SHA256 = {'formation': '4a6b04d374279f744fa2ad621ae07cd1c3a748b99d2d82a299f168e378260e0d', 'prefix': 'ac534a4470e1328bc727ad87a357c1a300c7a84446b057da363ffb6fe93165ba'}


@dataclass(frozen=True)
class SourceRealization:
    circuit_id: str
    block_id: str
    phase: str
    split: str
    circuit_seed: int
    uniform_seed: int
    engine_seed: int

    def __post_init__(self):
        for name in ("circuit_id", "block_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value) is None:
                raise ValueError("CA3_REALIZATION_ID_INVALID:" + name)
        if self.phase not in ("formation", "retrieval"):
            raise ValueError("CA3_REALIZATION_PHASE_INVALID")
        if self.split not in ("development", "train", "calibration"):
            raise ValueError("CA3_REALIZATION_QUALIFICATION_SPLIT_REQUIRED")
        for name in ("circuit_seed", "uniform_seed", "engine_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= MAX_SEED:
                raise ValueError("CA3_REALIZATION_SEED_INVALID:" + name)
        if self.phase == "formation" and (self.uniform_seed != self.circuit_seed or self.engine_seed != 1):
            raise ValueError("CA3_FORMATION_REQUIRES_SOURCE_COUPLED_UNIFORM_AND_DEFAULT_ENGINES")

    def record(self) -> dict:
        return dict(schema=SCHEMA, **asdict(self))

    def replay_arguments(self) -> list[str]:
        args = ["--noise-block", str(self.uniform_seed), str(self.engine_seed)]
        if self.circuit_seed != 10:
            args += ["--circuit-seed", str(self.circuit_seed)]
        return args


def from_record(record: dict) -> SourceRealization:
    keys = {"schema", *SourceRealization.__dataclass_fields__}
    if not isinstance(record, dict) or set(record) != keys or record.get("schema") != SCHEMA:
        raise ValueError("CA3_REALIZATION_SCHEMA_INVALID")
    return SourceRealization(**{k: v for k, v in record.items() if k != "schema"})


def load_realization(path: Path) -> SourceRealization:
    # Source RNG limits are checked by SourceRealization, independently of
    # model-training seed metadata in the shared configuration.
    return from_record(load_config(path, validate_training_seeds=False).get("ca3_source_realization"))


def _masked(text: str) -> str:
    return re.sub(r"/\*.*?\*/|//[^\r\n]*",
                  lambda m: re.sub(r"[^\r\n]", " ", m.group()), text, flags=re.S)



RESET_KERNEL_PATH = "carlsim/kernel/src/snn_manager.cpp"
RESET_KERNEL_SHA256 = "d14315a63715c4a0a722a22d6e0acfa542f3946ce40cba64c86c91aca8904a1d"
SOURCE_PARAMETER_CALLS_SHA256 = "6172b95d48dcc4f4d4de74e65dd4e1506af6bb90335d4c91a7b4c85533b7ef49"
# The pinned nine-parameter overload receives C,k,vr,vt,a,b,vpeak,c,d,
# each followed by its SD, then the CA3 refractory declaration.
SOURCE_PARAMETER_ORDER = ("C", "k", "vr", "vt", "a", "b", "vpeak", "c", "d")
SOURCE_RESET_VR = (
    ("CA3_QuadD_LM", "-73.4821116922868"),
    ("CA3_Axo_Axonic", "-57.099782869594"),
    ("CA3_Basket", "-57.506126"),
    ("CA3_BC_CCK", "-58.99667734"),
    ("CA3_Bistratified", "-64.67262808"),
    ("CA3_Ivy", "-70.4345135750261"),
    ("CA3_MFA_ORDEN", "-57.076423571379"),
    ("CA3_Pyramidal", "-63.12"),
)


def _reset_metadata(header_sha256: str) -> dict:
    return dict(
        schema="ca3_source_initial_state_contract_v1",
        scope=["voltage", "recovery", "Izh_ref_c"],
        source_header_sha256=header_sha256,
        source_parameter_calls_sha256=SOURCE_PARAMETER_CALLS_SHA256,
        reset_kernel=dict(path=RESET_KERNEL_PATH, sha256=RESET_KERNEL_SHA256),
        parameter_model="withParamModel_9",
        parameter_standard_deviations_zero=True,
        parameter_draw_rule="param + sd * float(drand48()); nine draws per neuron even when sd is zero",
        populations=[dict(population=name, voltage_parameter="vr", source_vr_literal=value,
                          voltage_mV=struct.unpack("f", struct.pack("f", float(value)))[0],
                          recovery=0.0, Izh_ref_c=0)
                     for name, value in SOURCE_RESET_VR],
        native_reset_deterministic=True,
        noise_block_changes_initial_state=False,
        complete_microscopic_state_qualified=False,
        independent_microscopic_states_qualified=False)


def source_reset_contract(header: bytes, reset_kernel: bytes) -> dict:
    """Authenticate actual copied source; describe only three native reset fields.

    Zero SDs make these fields deterministic, but the original reset still
    consumes its nine uniform draws. No initial-state diversity follows from
    changing the constructor seed or the post-setup external-noise seeds.
    """
    if not isinstance(header, bytes) or not isinstance(reset_kernel, bytes):
        raise ValueError("CA3_RESET_SOURCE_BYTES_REQUIRED")
    if hashlib.sha256(reset_kernel).hexdigest() != RESET_KERNEL_SHA256:
        raise ValueError("CA3_RESET_PINNED_KERNEL_REQUIRED")
    active = _masked(header.decode("utf-8"))
    calls = re.findall(r"\bsim\.setNeuronParameters\s*\(([^;]*)\)\s*;", active)
    if len(calls) != 8 or len(re.findall(r"\bsim\.setNeuronParameters\b", active)) != 8:
        raise ValueError("CA3_RESET_EIGHT_NINE_PARAMETER_CALLS_REQUIRED")
    arguments = [[part.strip() for part in call.split(",")] for call in calls]
    number = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    if ([row[0] for row in arguments] != [name for name, _ in SOURCE_RESET_VR]
            or any(len(row) != 20 or any(re.fullmatch(number, value) is None for value in row[1:])
                   for row in arguments)):
        raise ValueError("CA3_RESET_NINE_PARAMETER_HEADER_INVALID")
    for row in arguments:
        for name, sd in zip(SOURCE_PARAMETER_ORDER, row[2:19:2]):
            if float(sd) != 0.0:
                raise ValueError("CA3_RESET_NONZERO_PARAMETER_SD:" + row[0] + ":" + name)
    canonical = "\n".join(",".join(row) for row in arguments)
    if hashlib.sha256(canonical.encode()).hexdigest() != SOURCE_PARAMETER_CALLS_SHA256:
        raise ValueError("CA3_RESET_PINNED_PARAMETER_CALLS_REQUIRED")
    return _reset_metadata(hashlib.sha256(header).hexdigest())


def validate_source_reset_contract(report: dict) -> dict | None:
    """Validate a declaration against the report's existing source-byte binding.

    Historical reports remain valid without this separate field. The declared
    kernel is pinned by the analyzer; the consumer uses the already qualified
    build/native report provenance rather than a second source manifest.
    """
    if "source_initial_state_contract" not in report:
        return None
    declared = report["source_initial_state_contract"]
    if not isinstance(declared, dict) or "source_realization" not in report:
        raise ValueError("CA3_RESET_EXPLICIT_REALIZATION_REQUIRED")
    rows = [row for row in report.get("source_file_byte_checks", [])
            if row.get("path") == "generateCONFIGStateSTP.h"]
    header_sha = declared.get("source_header_sha256")
    if (not isinstance(header_sha, str) or re.fullmatch(r"[0-9a-f]{64}", header_sha) is None
            or len(rows) != 1 or rows[0].get("staged_sha256") != header_sha
            or declared != _reset_metadata(header_sha)):
        raise ValueError("CA3_RESET_SOURCE_CONTRACT_BINDING_DIFFERS")
    return _reset_metadata(header_sha)


def _injection(realization: SourceRealization, newline: str) -> str:
    lines = ["", "    // Explicit stage-noise realization; source reset has completed.",
             f"    srand48({realization.uniform_seed}L);"]
    lines.extend(f"    {engine}.seed({realization.engine_seed}U);" for engine in ENGINE_NAMES)
    return newline.join(lines)


def program(original: bytes, realization: SourceRealization) -> bytes:
    """Transform an already source-verified copy; never edit the author tree.

    The builder remains responsible for verifying the original source identity,
    circuit extension, integrator, and source phase before this transformation.
    No draws are skipped after the explicit post-setup seed boundary.
    """
    text = original.decode("utf-8")
    masked = _masked(text)
    seed = list(re.finditer(r"\bint\s+randSeed\s*=\s*(10)\s*;", masked))
    setup = list(re.finditer(r"\bsim\.setupNetwork\(\s*\)\s*;", masked))
    declared = re.findall(r"\bstd::default_random_engine\s+(\w+)\s*;", masked)
    if len(seed) != 1 or len(setup) != 1 or tuple(declared) != ENGINE_NAMES:
        raise ValueError("CA3_REALIZATION_SOURCE_RNG_BOUNDARY_CHANGED")
    if not re.search(r"\bint\s+myrandom\s*=\s*3\s*;", masked):
        raise ValueError("CA3_REALIZATION_CUE_SEED_CHANGED")
    boundary = setup[0].end()
    # Each generator is declared before setup and first sampled after it. A
    # changed source that has already used an engine needs a different contract.
    for engine in ENGINE_NAMES:
        if len(re.findall(r"\b" + engine + r"\b", masked[:boundary])) != 1:
            raise ValueError("CA3_REALIZATION_ENGINE_USED_BEFORE_BOUNDARY:" + engine)
    newline = "\r\n" if "\r\n" in text else "\n"
    constructor = re.findall(r'\bCARLsim\s+sim\(\s*"ca3_snn_GPU"\s*,\s*GPU_MODE\s*,\s*SILENT\s*,\s*numGPUs\s*,\s*randSeed\s*\)\s*;', masked[:boundary])
    if len(constructor) != 1:
        raise ValueError("CA3_REALIZATION_CARLSIM_CONSTRUCTOR_SEED_REQUIRED")
    injection = _injection(realization, newline) if realization.phase == "retrieval" else ""
    a, b = seed[0].span(1)
    # Apply in descending position order, retaining all other original bytes.
    changed = text[:boundary] + injection + text[boundary:]
    changed = changed[:a] + str(realization.circuit_seed) + changed[b:]
    return changed.encode("utf-8")


def contract(realization: SourceRealization) -> dict:
    return dict(realization=realization.record(),
                noise_seed_boundary=("after_setup_before_first_host_distribution_draw" if realization.phase == "retrieval"
                                     else "CARLsim_constructor_before_setup_and_source_host_draws"),
                shared_uniform_stream=True, common_seed_for_nine_host_engines=True,
                distribution_caches_persist_within_execution=True,
                source_cue_srand_seed=3, source_topology_noise_coupling_preserved=(realization.phase == "formation"),
                initial_state_policy="fresh_source_reset_not_checkpoint_state",
                independent_microscopic_states_qualified=False,
                independent_noise_qualified=False, scientific_fidelity_qualified=False,
                production_episodes=0)


def _verify_program(effective: bytes, realization: SourceRealization, expected_sha256: str) -> dict:
    """Undo only the declared seed transformation, then verify the pinned source."""
    text = effective.decode("utf-8")
    if realization.phase == "retrieval":
        injection = _injection(realization, "\r\n" if "\r\n" in text else "\n")
        if text.count(injection) != 1:
            raise ValueError("CA3_REALIZATION_EXACT_SEED_BOUNDARY_REQUIRED")
        text = text.replace(injection, "", 1)
    matches = list(re.finditer(r"\bint\s+randSeed\s*=\s*(\d+)\s*;", _masked(text)))
    if len(matches) != 1 or int(matches[0][1]) != realization.circuit_seed:
        raise ValueError("CA3_REALIZATION_COMPILED_CIRCUIT_SEED_DIFFERS")
    a, b = matches[0].span(1)
    base = (text[:a] + "10" + text[b:]).encode("utf-8")
    base_sha = hashlib.sha256(base).hexdigest()
    if base_sha != expected_sha256 or program(base, realization) != effective:
        raise ValueError("CA3_REALIZATION_PINNED_SOURCE_PROGRAM_REQUIRED")
    return dict(base_main_sha256=base_sha,
                effective_main_sha256=hashlib.sha256(effective).hexdigest())


def verify_retrieval_program(effective: bytes, realization: SourceRealization) -> dict:
    """Verify the pinned coupled RK4_5 retrieval after its declared seed overlay."""
    from .ca3_cue_omission import SOURCE_MAIN_SHA256

    if realization.phase != "retrieval":
        raise ValueError("CA3_REALIZATION_PINNED_CIRCUIT_RETRIEVAL_REQUIRED")
    return _verify_program(effective, realization, SOURCE_MAIN_SHA256[2])


def verify_formation_program(effective: bytes, realization: SourceRealization, *, prefix: bool = False) -> dict:
    """Verify source formation or its same-realization independent prefix."""
    if realization.phase != "formation":
        raise ValueError("CA3_REALIZATION_FORMATION_PHASE_REQUIRED")
    return _verify_program(effective, realization, FORMATION_MAIN_SHA256['prefix' if prefix else 'formation'])


def bind_program(report: dict, config: Path | None, source: Path) -> SourceRealization | None:
    """Require explicit intent, immutable compiled source, and the resolved config.

    This is source/configuration qualification only. The native executor retains
    its complete formation/normalization input qualification and resource gates.
    """
    present = "source_realization" in report
    if config is None:
        if present:
            raise ValueError("CA3_REALIZATION_EXPLICIT_CONFIGURATION_REQUIRED")
        return None
    if not present:
        raise ValueError("CA3_REALIZATION_BUILD_REQUIRED")
    realization = load_realization(config)
    declared = report.get("source_realization_config", {})
    scope = report.get("scope")
    expected_phase = "retrieval" if scope == "coupled_extension_retrieval" else "formation"
    if (report["source_realization"] != contract(realization)
            or declared.get("resolved") != realization.record()
            or declared.get("sha256") != sha256_file(config)
            or scope not in ("coupled_extension_prefix", "coupled_extension_formation", "coupled_extension_retrieval")
            or realization.phase != expected_phase
            or report.get("integration") != "RUNGE_KUTTA4_5"
            or report.get("core_neurons") != 168106):
        raise ValueError("CA3_REALIZATION_BUILD_AND_REQUEST_DIFFER")
    proof = (verify_retrieval_program(Path(source).read_bytes(), realization) if expected_phase == "retrieval"
             else verify_formation_program(Path(source).read_bytes(), realization, prefix=scope == "coupled_extension_prefix"))
    rows = [row for row in report.get("source_file_byte_checks", [])
            if row.get("path", "").startswith("src/main_ca3_")]
    if (len(rows) != 1 or report.get("overlay_source_sha256") != proof["effective_main_sha256"]
            or rows[0].get("staged_sha256") != proof["effective_main_sha256"]
            or rows[0].get("pre_realization_staged_sha256") != proof["base_main_sha256"]
            or rows[0].get("source_realization_transformation_verified") is not True
            or report.get("source_rng") != ("explicit_post_setup_noise_block_on_source_reset" if expected_phase == "retrieval"
                                            else "explicit_circuit_seed_with_source_coupled_formation_streams")):
        raise ValueError("CA3_REALIZATION_COMPILED_SOURCE_BINDING_DIFFERS")
    return realization


def bind_runtime(runtime: Path, report: dict, config: Path | None) -> SourceRealization | None:
    return bind_program(report, config, Path(runtime)/"realization_source.cpp")


def bind_circuit_input(formation: dict, replay: dict, realization: SourceRealization | None,
                       normalized_sha256: str) -> dict | None:
    """Reject a retrieval label that does not identify its actual formed graph.

    Legacy seed10 noise blocks retain the original content-addressed identity.
    New circuits require explicit, matching formation and replay provenance.
    Normalization itself remains bound by the source and output binary hashes.
    """
    declared = formation.get('source_realization')
    if realization is None:
        if declared is not None:
            raise ValueError('CA3_CIRCUIT_INPUT_REQUIRES_EXPLICIT_RETRIEVAL_REALIZATION')
        return None
    if realization.phase != 'retrieval':
        raise ValueError('CA3_CIRCUIT_INPUT_RETRIEVAL_PHASE_REQUIRED')
    if declared is None:
        if realization.circuit_seed != 10 or realization.circuit_id != 'normalized-' + normalized_sha256:
            raise ValueError('CA3_NEW_CIRCUIT_REQUIRES_ITS_OWN_FORMATION')
        return dict(circuit_id=realization.circuit_id, circuit_seed=10,
                    formation_realization=None, legacy_content_addressed_graph=True)
    if realization.circuit_id.startswith('normalized-') and realization.circuit_id != 'normalized-' + normalized_sha256:
        raise ValueError('CA3_CONTENT_ADDRESSED_CIRCUIT_CHECKPOINT_DIFFERS')
    source = from_record(declared['realization'])
    if (source.phase != 'formation' or declared != contract(source)
            or source.circuit_id != realization.circuit_id or source.circuit_seed != realization.circuit_seed
            or replay.get('source_realization') != declared):
        raise ValueError('CA3_FORMATION_RETRIEVAL_CIRCUIT_IDENTITY_DIFFERS')
    binding = formation.get('realization_circuit_binding', {})
    if (binding.get('circuit_id') != source.circuit_id or binding.get('circuit_seed') != source.circuit_seed
            or binding.get('fresh_native_graph_constructed') is not True
            or binding.get('formation_checkpoint_sha256') != formation.get('checkpoint_sha256')):
        raise ValueError('CA3_FRESH_FORMATION_CIRCUIT_BINDING_REQUIRED')
    return dict(circuit_id=source.circuit_id, circuit_seed=source.circuit_seed,
                formation_realization=declared, legacy_content_addressed_graph=False)
