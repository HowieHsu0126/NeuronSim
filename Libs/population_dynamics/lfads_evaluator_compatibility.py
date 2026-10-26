"""Verify the historical evaluator's numerical program before metric rescoring."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

LEGACY_EVALUATOR_SHA256 = 'd9f1aa59ee1a23945449fe83f69bc0493a9d6835f984967ad2f3b97acd19c95a'
PRE_REFIT_EVALUATOR_SHA256 = '250eda11996efc9b38161d9d9ee7e4c5c741b6a4b913bfa6203ae838a4051d8d'
TRAINER_METADATA_FUNCTIONS = {'_verify_inference_binding', '_verified_training'}
TRAINER_METADATA_IMPORTS = {'CONTINUATION_SCHEMA', 'LEGACY_TRAINER_SHA256', '_verified_run'}
RUN_METADATA_FIELDS = {'effective_training_epoch_budget', 'training_continuation'}


class _RemoveRunMetadata(ast.NodeTransformer):
    def visit_Assign(self, node):
        if (any(isinstance(t, ast.Name) and t.id == 'record' for t in node.targets)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == 'dict'):
            node.value.keywords = [k for k in node.value.keywords if k.arg not in RUN_METADATA_FIELDS]
        return self.generic_visit(node)


class _DefaultPriorPath(ast.NodeTransformer):
    """Specialize the optional prior extension to its original None default.

    Refitted forecasts use exact current source hashes. Historical forecasts
    may take only this unchanged default path, with no fitted prior supplied.
    """
    names = {'prior_refit', 'prior_parameters', 'conditional_prior', 'mean_prior', 'continue_from', 'completed_keys'}

    def visit_FunctionDef(self, node):
        for index in reversed(range(len(node.args.kwonlyargs))):
            if (node.args.kwonlyargs[index].arg in self.names
                    and isinstance(node.args.kw_defaults[index], ast.Constant)
                    and node.args.kw_defaults[index].value is None):
                del node.args.kwonlyargs[index]
                del node.args.kw_defaults[index]
        return self.generic_visit(node)

    def visit_If(self, node):
        test = node.test
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                and test.left.id in self.names and len(test.ops) == len(test.comparators) == 1
                and isinstance(test.ops[0], ast.IsNot) and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None and not node.orelse):
            return None
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                and test.left.id in self.names and len(test.ops) == len(test.comparators) == 1
                and isinstance(test.ops[0], ast.Is) and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None):
            return [self.visit(statement) for statement in node.body]
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {'prior_parameters', 'mean_prior', 'completed_keys'} and isinstance(node.value, ast.Constant)
                and node.value.value is None):
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == 'parser'
                and call.func.attr == 'add_argument' and call.args
                and isinstance(call.args[0], ast.Constant) and call.args[0].value in {'--prior-refit', '--conditional-prior', '--continue-from'}):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [key for key in node.keywords if not (
            key.arg == 'prior_parameters' and isinstance(key.value, ast.Name)
            and key.value.id == 'prior_parameters') and not (
            key.arg == 'prior_refit' and isinstance(key.value, ast.Attribute)
            and isinstance(key.value.value, ast.Name) and key.value.value.id == 'args'
            and key.value.attr == 'prior_refit') and not (
            key.arg == 'conditional_prior' and isinstance(key.value, ast.Attribute)
            and isinstance(key.value.value, ast.Name) and key.value.value.id == 'args'
            and key.value.attr == 'conditional_prior') and not (
            key.arg == 'continue_from' and isinstance(key.value, ast.Attribute)
            and isinstance(key.value.value, ast.Name) and key.value.value.id == 'args'
            and key.value.attr == 'continue_from')]
        return self.generic_visit(node)


def numerical_program(source: str) -> str:
    """Remove only the separately verified continuation-metadata surface."""
    tree = _DefaultPriorPath().visit(_RemoveRunMetadata().visit(ast.parse(source)))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in TRAINER_METADATA_FUNCTIONS:
            continue
        if (isinstance(node, ast.ImportFrom) and node.level == 1
                and node.module == 'lfads_trajectory_training'):
            node.names = [n for n in node.names if n.name not in TRAINER_METADATA_IMPORTS]
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'BUDGET_TRAINER_SHA256'):
            continue
        body.append(node)
    tree.body = body
    return ast.dump(tree, include_attributes=False)


def verify_legacy_evaluator(recorded_sha256: str, legacy: Path, current: Path) -> dict:
    legacy, current = Path(legacy), Path(current)
    original, updated = legacy.read_bytes(), current.read_bytes()
    if (recorded_sha256 not in {LEGACY_EVALUATOR_SHA256, PRE_REFIT_EVALUATOR_SHA256}
            or hashlib.sha256(original).hexdigest() != recorded_sha256):
        raise ValueError('pinned original evaluator source does not match its recorded identity')
    if numerical_program(original.decode()) != numerical_program(updated.decode()):
        raise ValueError('legacy evaluator numerical program changed outside continuation metadata')
    return dict(original_evaluator_sha256=recorded_sha256,
                execution_evaluator_sha256=hashlib.sha256(updated).hexdigest(),
                numerical_program_ast_equal=True,
                optional_prior_refit_disabled=True, optional_conditional_prior_disabled=True,
                separately_verified_training_functions=sorted(TRAINER_METADATA_FUNCTIONS),
                added_run_metadata_fields=sorted(RUN_METADATA_FIELDS),
                scope='numerical bodies and module setup unchanged; actual checkpoints and trainer bridge independently verified')
