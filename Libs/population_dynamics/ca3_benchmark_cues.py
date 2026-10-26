"""Apply declared missing/corrupted cue controls to a copied coupled CA3 program."""
from __future__ import annotations

import hashlib
import re

from .benchmark_tasks import ca3_cue_targets, task_protocol
from .ca3_realization import ENGINE_NAMES
from .ca3_cue_omission import SOURCE_MAIN_SHA256


def cue_program(source: bytes, condition: dict, cue_seed: int) -> tuple[bytes, dict]:
    if hashlib.sha256(source).hexdigest() != SOURCE_MAIN_SHA256[2]:
        raise ValueError("CA3_BENCHMARK_PINNED_COUPLED_LOCAL_INPUT_REQUIRED")
    targets = ca3_cue_targets(condition, cue_seed)
    text = source.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    declaration = "int numPyramidalFire = 550;"
    lookup = "const int benchmark_cue_targets[3][550] = {" + ",".join(
        "{" + ",".join(map(str, row)) + "}" for row in targets) + "};"
    if text.count(declaration) != 1:
        raise ValueError("CA3_BENCHMARK_ASSEMBLY_DECLARATION_CHANGED")
    text = text.replace(declaration, lookup + newline + "    " + declaration, 1)
    expression = (r"if \(randPyrCell < k \+ 276\)\s*\{\s*"
                  r'// std::cout << "PC chosen is: " << randPyrCell << "\\n";\s*'
                  r"pc_current.at\(randPyrCell\) = 10000.0f;\s*\}")
    replacement = newline.join([
        "const int benchmark_slot = randPyrCell - pyr_vec.front() - k;",
        "                        const int benchmark_target = benchmark_cue_targets[k / 550][benchmark_slot];",
        "                        if (benchmark_target >= 0)",
        "                        {",
        "                            pc_current.at(pyr_vec.front() + benchmark_target) = 10000.0f;",
        "                        }"])
    text, matches = re.subn(expression, lambda _: replacement, text)
    restore = "pc_current.at(randPyrCell) = pc_background_current.at(randPyrCell);"
    if matches != 1 or text.count(restore) != 1:
        raise ValueError("CA3_BENCHMARK_CUE_OR_RESTORE_BOUNDARY_CHANGED")
    text = text.replace(restore, newline.join([
        "const int benchmark_slot = randPyrCell - pyr_vec.front() - k;",
        "                        const int benchmark_target = benchmark_cue_targets[k / 550][benchmark_slot];",
        "                        if (benchmark_target >= 0)",
        "                            pc_current.at(pyr_vec.front() + benchmark_target) = pc_background_current.at(pyr_vec.front() + benchmark_target);"]), 1)
    changed = text.encode("utf-8")
    contract = dict(schema="ca3_benchmark_cue_program_v1", condition=condition, cue_seed=cue_seed,
                    base_main_sha256=hashlib.sha256(source).hexdigest(),
                    effective_main_sha256=hashlib.sha256(changed).hexdigest(),
                    targets=[list(row) for row in targets],
                    duration_ms=1001, cue_starts_ms=[520, 750, 980], cue_duration_ms=20,
                    random_draws_changed=False, formation_changed=False,
                    scientific_fidelity_qualified=False, production_episodes=0)
    return changed, contract


def runtime_program(source: bytes, *, duration_ms: int = 1001) -> tuple[bytes, dict]:
    """Compile once; read circuit/block controls before native simulation starts."""
    task_protocol("ca3_pattern_completion", retrieval_duration_ms=duration_ms)
    condition = dict(condition_id="cue_omission", true_cue_units=0, false_cue_units=0)
    changed, _ = cue_program(source, condition, 1)
    text = changed.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    initializer = re.search(r"const int benchmark_cue_targets\[3\]\[550\] = \{[^;]+\};", text)
    if initializer is None or text.count("int main() {") != 1 or text.count("int randSeed = 10;") != 1:
        raise ValueError("CA3_BENCHMARK_RUNTIME_ARGUMENT_BOUNDARY_CHANGED")
    load = newline.join([
        "int benchmark_cue_targets[3][550];",
        "    std::ifstream benchmark_controls(argv[1]);",
        "    for (int memory = 0; memory < 3; ++memory)",
        "        for (int slot = 0; slot < 550; ++slot) {",
        "            int target;",
        "            if (!(benchmark_controls >> target) || target < -1 || target >= 1650) return 2;",
        "            benchmark_cue_targets[memory][slot] = target;",
        "        }",
        "    std::string benchmark_extra;",
        "    if (benchmark_controls >> benchmark_extra) return 2;"])
    text = text[:initializer.start()] + load + text[initializer.end():]
    argument_check = newline.join([
        "int main(int argc, char** argv) {",
        "    if (argc != 5) return 2;",
        "    long benchmark_seeds[3];",
        "    for (int index = 0; index < 3; ++index) {",
        "        char* end = nullptr;",
        "        benchmark_seeds[index] = std::strtol(argv[index + 2], &end, 10);",
        "        if (*end || benchmark_seeds[index] < 1 || benchmark_seeds[index] > 2147483646L) return 2;",
        "    }"])
    text = text.replace("int main() {", argument_check, 1)
    text = text.replace("int randSeed = 10;", "int randSeed = int(benchmark_seeds[2]);", 1)
    setup = "sim.setupNetwork();"
    if text.count(setup) != 1:
        raise ValueError("CA3_BENCHMARK_RUNTIME_SETUP_BOUNDARY_CHANGED")
    seeds = ["srand48(benchmark_seeds[0]);"]
    seeds += [f"{name}.seed(unsigned(benchmark_seeds[1]));" for name in ENGINE_NAMES]
    text = text.replace(setup, setup + newline + "    " + (newline + "    ").join(seeds), 1)
    if duration_ms != 1001:
        # The source outer index advances by one for each initial-background ms,
        # but its middle five iterations execute 480ms of cues and gaps. Extend
        # only the final background branch. Preserve the complete 1001ms prefix.
        outer = "for (int i=0; i<526; i++)"
        terminal = "if (i >= 525 && i < 526)"
        if text.count(outer) != 1 or text.count(terminal) != 1:
            raise ValueError("CA3_BENCHMARK_TERMINAL_BACKGROUND_BOUNDARY_CHANGED")
        limit = 525 + duration_ms - 1000
        text = text.replace(outer, f"for (int i=0; i<{limit}; i++)", 1)
        text = text.replace(terminal, f"if (i >= 525 && i < {limit})", 1)
    result = text.encode("utf-8")
    return result, dict(schema="ca3_benchmark_runtime_program_v1",
                        base_main_sha256=hashlib.sha256(source).hexdigest(),
                        effective_main_sha256=hashlib.sha256(result).hexdigest(),
                        arguments=["cue_targets_file", "uniform_seed", "engine_seed", "circuit_seed"],
                        cue_assignments_and_restore_transformed=True,
                        background_distribution_parameters_changed=False,
                        seed_boundary="after_setup_before_first_host_draw",
                        integration="RUNGE_KUTTA4_5", duration_ms=duration_ms,
                        terminal_background_extension_ms=duration_ms - 1001,
                        source_reset_preserved=True, scientific_fidelity_qualified=False)
