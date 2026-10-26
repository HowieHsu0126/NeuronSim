import unittest

from .lfads_evaluator_compatibility import numerical_program

SOURCE = '''from .lfads_trajectory_training import TrajectoryCohort, _binding
MATCHED_PROTOCOL = dict(joint_samples=16)
def _verified_training(x):
    return x
def _verify_inference_binding(x):
    return x
def evaluate(x):
    record = dict(status='RUNNING', model_type='lfads')
    return x + 1, record
'''


class EvaluatorCompatibility(unittest.TestCase):
    def test_only_continuation_metadata_surface_can_differ(self):
        changed=SOURCE.replace('TrajectoryCohort, _binding','TrajectoryCohort, _binding, _verified_run')
        changed=changed.replace('MATCHED_PROTOCOL =',"BUDGET_TRAINER_SHA256 = 'metadata'\nMATCHED_PROTOCOL =")
        changed=changed.replace("model_type='lfads')","model_type='lfads', effective_training_epoch_budget=80, training_continuation=None)")
        changed=changed.replace('return x\n','return checked(x)\n')
        self.assertEqual(numerical_program(SOURCE),numerical_program(changed))

    def test_sampler_protocol_import_or_nonmetadata_body_changes_are_rejected(self):
        for old,new in [('samples=16','samples=17'),('x + 1','x + 2'),
                        ('from .lfads_trajectory_training','from .different_training'),
                        ("model_type='lfads'","model_type='other'")]:
            with self.subTest(change=new):
                self.assertNotEqual(numerical_program(SOURCE),numerical_program(SOURCE.replace(old,new)))

    def test_optional_prior_default_path_preserves_original_program(self):
        changed = SOURCE.replace('def evaluate(x):', 'def evaluate(x, *, prior_refit=None):')
        changed = changed.replace('    record =', '''    prior_parameters = None
    if prior_refit is not None:
        prior_parameters = verify_and_load(prior_refit)
    record =''')
        self.assertEqual(numerical_program(SOURCE), numerical_program(changed))
        self.assertNotEqual(numerical_program(SOURCE),
                            numerical_program(changed.replace('prior_refit=None', 'prior_refit=1')))
        self.assertNotEqual(numerical_program(SOURCE),
                            numerical_program(changed.replace('prior_parameters = None', 'prior_parameters = 1')))

    def test_conditional_mean_none_default_retains_the_original_numerical_branch(self):
        changed = SOURCE.replace('def evaluate(x):', 'def evaluate(x, *, conditional_prior=None):')
        changed = changed.replace('    record =', "    mean_prior = None\n    if conditional_prior is not None:\n        mean_prior = load(conditional_prior)\n    record =")
        changed = changed.replace('    return x + 1, record', "    if mean_prior is None:\n        return x + 1, record\n    else:\n        return variant(x), record")
        self.assertEqual(numerical_program(SOURCE), numerical_program(changed))
        for old, new in [('conditional_prior=None', 'conditional_prior=1'),
                         ('mean_prior = None', 'mean_prior = 1'), ('return x + 1', 'return x + 2')]:
            with self.subTest(change=new):
                self.assertNotEqual(numerical_program(SOURCE), numerical_program(changed.replace(old, new)))


if __name__ == '__main__':
    unittest.main()
