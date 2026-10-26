"""Contract tests against the actual pinned CA3 source, staged without execution."""
from collections import Counter
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import unittest

from .ca3_coupled_extension import extend_source
from .ca3_parameter_comparison import native_parameters, NUMBER

TARGETS = {
    "formation": "ca3_full_net_lognormalI_09_05_23_assem_275",
    "retrieval": "ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275",
}


def source_fixture(role):
    configured = os.environ.get("NEURONSIM_CA3_SOURCE_ROOT")
    root = Path(configured) if configured else Path(__file__).resolve().parents[1] / "external/ca3_formation_retrieval"
    target = TARGETS[role]
    project = root / role / "project" / target
    paths = [project / "generateCONFIGStateSTP.h", project / "src" / ("main_" + target + ".cpp"),
             root / role / "build" / target / "trainPat.csv"]
    if any(not path.is_file() for path in paths):
        raise RuntimeError("Stage the known CA3 fixtures under NEURONSIM_CA3_SOURCE_ROOT: " + str(root))
    return tuple(path.read_bytes().decode("utf-8") for path in paths)


def masked(text):
    return re.sub(r'/\*.*?\*/|//[^\r\n]*',
                  lambda match: re.sub(r'[^\r\n]', ' ', match.group()), text, flags=re.S)


def header_without_sizes_and_probabilities(text):
    active = masked(text)
    spans = [m.span(1) for m in re.finditer(r'createGroup\("[^\"]+",\s*(\d+)', active)]
    spans += [m.span(1) for m in re.finditer(
        r'"random"\s*,\s*RangeWeight\([^)]*\)\s*,\s*(' + NUMBER + r')f', active)]
    for start, stop in sorted(spans, reverse=True):
        text = text[:start] + "<changed>" + text[stop:]
    return text


class CoupledExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {role: source_fixture(role) for role in TARGETS}

    def test_reference_is_exact_identity_including_crlf_and_comments(self):
        for role, inputs in self.sources.items():
            crlf = tuple(value.replace('\r\n', '\n').replace('\n', '\r\n') for value in inputs)
            header, main, pattern = crlf
            header += '\r\n// sim.connect(fake, fake, "random", RangeWeight(0,1,2), 1f);\r\n'
            main += '\r\n/* numPyramidalFire = 999; k += 999; randPyrCell < k + 999 */\r\n'
            expected = (header, main, pattern)
            result = extend_source(*expected, multiplier=1, role=role)
            with self.subTest(role=role):
                self.assertEqual(result[:3], expected)
                self.assertEqual(result[3]['status'], 'UNCHANGED_SOURCE_REFERENCE')
                self.assertEqual(result[3]['transformation_scope'], [])

    def test_all_groups_and_expected_indegrees_with_all_other_header_bytes_preserved(self):
        for role, inputs in self.sources.items():
            header, _, _, metadata = extend_source(*inputs, multiplier=2, role=role)
            old_counts, old_p = native_parameters(inputs[0])
            new_counts, new_p = native_parameters(header)
            with self.subTest(role=role):
                self.assertEqual(sum(new_counts.values()), 168106)
                self.assertEqual(new_counts, {name: 2 * value for name, value in old_counts.items()})
                self.assertEqual(set(new_p), set(old_p))
                self.assertEqual(len(new_p), 51)
                for pair, value in old_p.items():
                    self.assertEqual(Decimal(value) * old_counts[pair[0]],
                                     Decimal(new_p[pair]) * new_counts[pair[0]])
                self.assertEqual(header_without_sizes_and_probabilities(header),
                                 header_without_sizes_and_probabilities(inputs[0]))
                self.assertEqual(metadata['connection_count'], 51)

    def test_only_role_specific_active_main_literals_change(self):
        for role, inputs in self.sources.items():
            header, main, pattern = inputs
            main += '\n// numPyramidalFire += 275; k += 275; randPyrCell < k + 138\n'
            main += '/* an unrelated 275 and 138 remain 275 and 138 */\n'
            result = extend_source(header, main, pattern, multiplier=2, role=role)[1]
            old_active, new_active = masked(main), masked(result)
            for name, count in [('numPyramidalFire', 5 if role == 'formation' else 2),
                                ('k', 2 if role == 'formation' else 1)]:
                expression = r'\b' + name + r'\s*(?:\+=|=)\s*550\s*;'
                self.assertEqual(len(re.findall(expression, new_active)), count)
            cue = re.findall(r'randPyrCell\s*<\s*k\s*\+\s*(\d+)', new_active)
            self.assertEqual(cue, ['276'] if role == 'retrieval' else [])
            reverse = result
            for match in reversed(list(re.finditer(r'\b(numPyramidalFire|k)\s*(?:\+=|=)\s*(550)\s*;', new_active))):
                start, stop = match.span(2)
                reverse = reverse[:start] + '275' + reverse[stop:]
            if role == 'retrieval':
                match = re.search(r'randPyrCell\s*<\s*k\s*\+\s*(276)', masked(reverse))
                reverse = reverse[:match.start(1)] + '138' + reverse[match.end(1):]
            self.assertEqual(reverse, main)
            self.assertEqual(re.findall(r'sim\.runNetwork\([^)]*\)', old_active),
                             re.findall(r'sim\.runNetwork\([^)]*\)', new_active))

    def test_complete_temporal_membership_and_partial_cue_fraction(self):
        for role, inputs in self.sources.items():
            result = extend_source(*inputs, multiplier=2, role=role)
            old, new = list(map(int, inputs[2].split())), list(map(int, result[2].split()))
            self.assertEqual(len(old), 1100)
            self.assertEqual(len(new), 2200)
            self.assertEqual(set(new), set(range(550)))
            for time_bin in range(20):
                source = old[55 * time_bin:55 * (time_bin + 1)]
                derived = new[110 * time_bin:110 * (time_bin + 1)]
                self.assertEqual(derived, [child for index in source for child in (2 * index, 2 * index + 1)])
                self.assertEqual(Counter(value // 2 for value in derived),
                                 Counter({key: 2 * value for key, value in Counter(source).items()}))
                self.assertEqual([value for value in derived if value < 276],
                                 [child for index in source if index < 138 for child in (2 * index, 2 * index + 1)])
            self.assertEqual(Decimal(276) / 550, Decimal(138) / 275)
            self.assertEqual(result[3]['cue_schedule']['time_bins'], 20)
            self.assertEqual(result[3]['derived_cue_size'], 550 if role == 'formation' else 276)

    def test_development_metadata_never_claims_generated_or_qualified_results(self):
        metadata = extend_source(*self.sources['retrieval'], multiplier=2, role='retrieval')[3]
        self.assertEqual(metadata['status'], 'PROSPECTIVE_UNQUALIFIED_COUPLED_EXTENSION')
        self.assertEqual(metadata['source_core_neurons'], 84053)
        self.assertEqual(metadata['derived_core_neurons'], 168106)
        self.assertTrue(metadata['fresh_native_graph_required'])
        self.assertTrue(metadata['fresh_formation_and_normalized_checkpoint_required'])
        self.assertFalse(metadata['graph_or_output_cloning'])
        self.assertFalse(metadata['source_semantics_preserved'])
        for key, value in metadata.items():
            if key.endswith('_qualified') or key == 'production_eligible':
                self.assertIs(value, False)
        json.dumps(metadata, allow_nan=False)

    def test_invalid_multiplier_and_role(self):
        for multiplier in (True, False, 0, 3, -1, 1.0, 2.0, '2', None):
            with self.subTest(multiplier=multiplier), self.assertRaises(ValueError):
                extend_source(*self.sources['formation'], multiplier=multiplier, role='formation')
        with self.assertRaises(ValueError):
            extend_source(*self.sources['formation'], multiplier=2, role='prefix')

    def test_changed_counts_and_nonrandom_missing_duplicate_connections_rejected(self):
        header, main, pattern = self.sources['formation']
        start = header.index('sim.connect(')
        stop = header.index(';', start) + 1
        connection = header[start:stop]
        variants = [header.replace('2609', '2610', 1), header.replace('"random"', '"full"', 1),
                    header[:start] + header[stop:], header + '\n' + connection]
        for changed in variants:
            with self.subTest(header=changed[:80]), self.assertRaises(ValueError):
                extend_source(changed, main, pattern, multiplier=2, role='formation')

    def test_changed_size_timing_and_rng_controls_rejected(self):
        for role, (header, main, pattern) in self.sources.items():
            variants = [main.replace('numPyramidalFire = 275', 'numPyramidalFire = 276', 1),
                        main.replace('k += 275', 'k += 274', 1),
                        main.replace('gammaCycle = 20', 'gammaCycle = 21', 1),
                        main.replace('randSeed = 10', 'randSeed = 11', 1),
                        main + '\nnumPyramidalFire *= 2;\n',
                        main.replace('i < 500' if role == 'formation' else 'i < 260',
                                     'i < 501' if role == 'formation' else 'i < 261', 1)]
            if role == 'retrieval':
                variants.append(main.replace('randPyrCell < k + 138', 'randPyrCell < k + 137', 1))
            for changed in variants:
                self.assertNotEqual(changed, main)
                with self.subTest(role=role), self.assertRaises(ValueError):
                    extend_source(header, changed, pattern, multiplier=2, role=role)

    def test_invalid_pattern_shape_range_and_incomplete_membership(self):
        header, main, pattern = self.sources['formation']
        tokens = pattern.split()
        variants = [' '.join(tokens[:-1]), pattern + ' 0', '-1 ' + ' '.join(tokens[1:]),
                    '275 ' + ' '.join(tokens[1:]), 'x ' + ' '.join(tokens[1:]),
                    ' '.join('0' if int(value) == 274 else value for value in tokens)]
        for changed in variants:
            with self.assertRaises(ValueError):
                extend_source(header, main, changed, multiplier=2, role='formation')


if __name__ == '__main__':
    unittest.main()
