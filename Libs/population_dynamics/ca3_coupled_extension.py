"""Prospective twofold CA3 graph extension; never qualifies native source fidelity.

This transforms copied source strings only. The caller must build a fresh native
random graph, form its own learned weights, and bind its own normalized checkpoint.
The preserved 84,053-unit reference and its recordings are not extension inputs.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import re

from .ca3_parameter_comparison import NUMBER, native_parameters
from .carlsim_records import CA3_REFERENCE_POPULATIONS as SOURCE_COUNTS

ASSEMBLY_SIZE = 275
TIME_BINS = 20
ASSIGNMENTS_PER_BIN = 55
PARTIAL_CUE_SIZE = 138
_GROUP = re.compile(
    r'int\s+(\w+)\s*=\s*sim\.createGroup\(\s*"([^\"]+)"\s*,\s*(\d+)\s*,')
_CONNECTION = re.compile(
    r'sim\.connect\(\s*(\w+)\s*,\s*(\w+)\s*,\s*"random"\s*,'
    r'\s*RangeWeight\([^)]*\)\s*,\s*(' + NUMBER + r')f\s*,')


def _active(text: str) -> str:
    """Mask comments without moving character offsets or newline boundaries."""
    return re.sub(r'/\*.*?\*/|//[^\r\n]*',
                  lambda match: re.sub(r'[^\r\n]', ' ', match.group()),
                  text, flags=re.S)


def _replace(text: str, changes: list[tuple[int, int, str]]) -> str:
    for start, stop, value in sorted(changes, reverse=True):
        text = text[:start] + value + text[stop:]
    return text


def _main_changes(main: str, role: str, multiplier: int) -> list[tuple[int, int, str]]:
    active = _active(main)
    expected = {
        "formation": {
            "numPyramidalFire": [("=", 275), ("+=", 275), ("=", 275), ("+=", 275), ("=", 275)],
            "k": [("=", 0), ("+=", 275), ("=", 0), ("+=", 275), ("=", 0)],
        },
        "retrieval": {"numPyramidalFire": [("=", 275), ("+=", 275)],
                      "k": [("=", 0), ("+=", 275)]},
    }[role]
    changes = []
    for name, assignments in expected.items():
        matches = list(re.finditer(r'\b' + name + r'\s*(\+=|=)\s*(\d+)\s*;', active))
        writes = re.findall(r'\b' + name + r'\s*(?:\+\+|--|[+*/%\-]?=)', active)
        if ([(m[1], int(m[2])) for m in matches] != assignments
                or len(writes) != len(matches)):
            raise ValueError("CA3_EXTENSION_MAIN_SIZE_CONTROL_CHANGED:" + name)
        changes.extend((*match.span(2), str(int(match[2]) * multiplier))
                       for match in matches if int(match[2]) == ASSEMBLY_SIZE)
    cue = list(re.finditer(r'\brandPyrCell\s*<\s*k\s*\+\s*(\d+)\b', active))
    if [int(match[1]) for match in cue] != ([PARTIAL_CUE_SIZE] if role == "retrieval" else []):
        raise ValueError("CA3_EXTENSION_PARTIAL_CUE_CONTROL_CHANGED")
    changes.extend((*match.span(1), str(PARTIAL_CUE_SIZE * multiplier)) for match in cue)

    # These are the unchanged native control boundaries, not tunable parameters.
    compact = re.sub(r'\s+', '', active)
    controls = ["intrandSeed=10;", "intmyrandom=3;", "intgammaCycle=20;",
                "intnumIrrG=trainPat.size()/gammaCycle;",
                "sim.setIntegrationMethod(RUNGE_KUTTA4,5);",
                "for(intj=0;j<210;j++)", "if(i%2==0)"]
    controls.append("for(inti=0;i<1200;i++)" if role == "formation"
                    else "for(inti=0;i<526;i++)")
    if role == "formation":
        controls.append("intnumPatterns=3;")
    if any(compact.count(control) != 1 for control in controls):
        raise ValueError("CA3_EXTENSION_NATIVE_CONTROL_CHANGED")
    windows = re.findall(r'if\(i>=(\d+)&&i<(\d+)\)', compact)
    expected_windows = ([('0', '500'), ('500', '1000'), ('1000', '1200')]
                        if role == "formation" else
                        [('0', '260'), ('260', '520'), ('520', '525'), ('525', '526')])
    calls = re.findall(r'sim\.runNetwork\(([^)]*)\)', compact)
    srand_calls = re.findall(r'std::srand\(([^)]*)\)', compact)
    if (windows != expected_windows
            or calls != ["0,1"] * (4 if role == "formation" else 5)
            or srand_calls != ["myrandom"] * (3 if role == "formation" else 2)):
        raise ValueError("CA3_EXTENSION_NATIVE_TIMING_OR_RNG_CHANGED")
    return changes


def extend_source(header: str, main: str, pattern_text: str, *, multiplier: int,
                  role: str) -> tuple[str, str, str, dict]:
    """Return copied source plus consumed provenance; execute nothing.

    Only multiplier 1 (identity) or 2 (unqualified development) is supported.
    Random connections include eligible self-pairs in the pinned CARLsim path,
    hence halving p while doubling N_pre retains expected indegree N_pre*p.
    """
    if type(multiplier) is not int or multiplier not in (1, 2):
        raise ValueError("CA3_EXTENSION_MULTIPLIER_MUST_BE_ONE_OR_TWO")
    if role not in ("formation", "retrieval"):
        raise ValueError("CA3_EXTENSION_ROLE_REQUIRED")
    if not all(isinstance(value, str) for value in (header, main, pattern_text)):
        raise ValueError("CA3_EXTENSION_SOURCE_STRINGS_REQUIRED")
    counts, probabilities = native_parameters(header)
    if counts != SOURCE_COUNTS or sum(counts.values()) != 84053:
        raise ValueError("CA3_EXTENSION_NATIVE_COUNTS_REQUIRED")
    if re.fullmatch(r'\s*\d+(?:\s+\d+)*\s*', pattern_text) is None:
        raise ValueError("CA3_EXTENSION_PATTERN_INTEGER_FORMAT_REQUIRED")
    tokens = list(re.finditer(r'\d+', pattern_text))
    indices = [int(token.group()) for token in tokens]
    if (len(indices) != TIME_BINS * ASSIGNMENTS_PER_BIN
            or set(indices) != set(range(ASSEMBLY_SIZE))):
        raise ValueError("CA3_EXTENSION_PATTERN_SHAPE_OR_MEMBERSHIP_CHANGED")
    main_changes = _main_changes(main, role, multiplier)
    active_header = _active(header)
    header_changes = [(*match.span(3), str(int(match[3]) * multiplier))
                      for match in _GROUP.finditer(active_header)]
    connection_rows = []
    with localcontext() as context:
        context.prec = max(50, max(len(value) for value in probabilities.values()) + 10)
        for match in _CONNECTION.finditer(active_header):
            probability = format(Decimal(match[3]) / multiplier, 'f')
            header_changes.append((*match.span(3), probability))
            connection_rows.append(dict(pre=match[1], post=match[2],
                                        source_probability=match[3], derived_probability=probability))
    cue_size = ASSEMBLY_SIZE if role == "formation" else PARTIAL_CUE_SIZE
    cue_assignments = sum(index < cue_size for index in indices)
    metadata = dict(
        schema="ca3_coupled_extension_v1",
        status="UNCHANGED_SOURCE_REFERENCE" if multiplier == 1 else "PROSPECTIVE_UNQUALIFIED_COUPLED_EXTENSION",
        source_role=role, multiplier=multiplier,
        source_core_neurons=sum(counts.values()), derived_core_neurons=sum(counts.values()) * multiplier,
        source_populations=counts, derived_populations={name: count * multiplier for name, count in counts.items()},
        connections=connection_rows, connection_count=len(connection_rows),
        expected_indegree_rule="N_pre*p; eligible self-pairs retained; p_derived=p_source/multiplier",
        source_assembly_size=ASSEMBLY_SIZE, derived_assembly_size=ASSEMBLY_SIZE * multiplier,
        source_cue_size=cue_size, derived_cue_size=cue_size * multiplier,
        cue_schedule=dict(time_bins=TIME_BINS, bin_duration_ms=1,
                          source_assignments_per_bin=ASSIGNMENTS_PER_BIN,
                          derived_assignments_per_bin=ASSIGNMENTS_PER_BIN * multiplier,
                          source_total_assignments=len(indices), derived_total_assignments=len(indices) * multiplier,
                          source_cued_assignments=cue_assignments, derived_cued_assignments=cue_assignments * multiplier,
                          meaning="scheduled input assignments; not spikes or measured responses",
                          token_mapping="i -> [multiplier*i + j for j in range(multiplier)]"),
        transformation_scope=["core group counts", "random connection probabilities", "assembly offsets and sizes",
                              "retrieval cue cutoff", "input pattern membership"] if multiplier == 2 else [],
        preserved=["weights", "intrinsic parameters", "STP", "ESTDP", "delays", "source timings", "source RNG code",
                   "unrelated source text, comments and line endings"],
        fresh_native_graph_required=multiplier == 2, fresh_native_noise_required=multiplier == 2,
        fresh_formation_and_normalized_checkpoint_required=multiplier == 2,
        graph_or_output_cloning=False, duplicated_input_schedules_only=multiplier == 2,
        source_semantics_preserved=multiplier == 1, scientific_fidelity_qualified=False,
        biological_count_qualified=False, source_scale_qualified=False,
        pattern_completion_qualified=False, normalization_qualified=False, production_eligible=False,
    )
    if multiplier == 1:
        return header, main, pattern_text, metadata
    pattern_changes = [(*token.span(), f'{2 * index} {2 * index + 1}')
                       for token, index in zip(tokens, indices)]
    return (_replace(header, header_changes), _replace(main, main_changes),
            _replace(pattern_text, pattern_changes), metadata)
