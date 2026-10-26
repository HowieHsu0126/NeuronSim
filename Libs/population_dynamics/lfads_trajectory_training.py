"""Full-episode LFADS development fitting with separate validation randomness.

Posterior loss selects checkpoints within LFADS. It is not a free-forecast score,
convergence certificate, or a likelihood comparison against other model families.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import tensorflow as tf
import yaml

from .lfads import LFADSConditionalSimulator, SOURCE_REVISION, TRAINING_RANDOM_SCHEMA, VALIDATION_RANDOM_SCHEMA
from .native_count_windows import NativeEpisodeCounts
from .native_trajectory_cohort import (
    TrajectoryCohort as NativeTrajectoryCohort, load_dataset_index, partition,
)
from .provenance import freeze_json, sha256_file


QUALIFIED_MODEL_SHA256 = '55cc455d574b879700f97c7617fa16c787943d6d16baf1cf20638d52b585e405'
FIXED_PROTOCOL = dict(schema='neural_population_lfads_trajectory_training_v1',
    training_seeds=[41,42,43], fit_blocks=list(range(14)), validation_blocks=list(range(14,18)),
    bin_ms=5., dt_ms=.1, duration_ms=6000., expected_units=100000, history_bins=200, horizon_bins=1000,
    expected_parameters=40130752, maximum_epochs=20, early_stopping_patience=4, validation_draws=4,
    expected_poisson_rng_family='per_edge_philox4_32_10', cpu_threads=2, formal_training_completed=False,
    selection_metric='full_posterior_source_cost')
LOSS_TERMS = ('source_cost','reconstruction_nll','kl_cost','l2_cost','kl_weight','l2_weight')


def validate_config(config, seed):
    if any(config.get(key) != value for key, value in FIXED_PROTOCOL.items()):
        raise ValueError('qualified full-episode LFADS development protocol required')
    digest = hashlib.sha256(json.dumps(config['model'], sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()
    if digest != QUALIFIED_MODEL_SHA256:
        raise ValueError('LFADS model differs from the qualified full-capacity source configuration')
    if type(seed) is not int or seed not in config['training_seeds']:
        raise ValueError('registered training seed41/42/43 required')
    if type(config['validation_seed_base']) is not int or not 0 <= config['validation_seed_base'] < 2**63:
        raise ValueError('nonnegative int63 validation seed base required')
    if config['device'] not in ('cpu','cuda'):
        raise ValueError('explicit cpu or cuda placement required')
    if config['device'] == 'cuda' and os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
        raise ValueError('qualified LFADS CUDA execution requires NVIDIA_TF32_OVERRIDE=0')


def validation_seed(base, seed, case_id, draw):
    identity = json.dumps(['lfads_full_posterior_validation_v1',base,seed,case_id,draw], separators=(',',':'))
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little') % (2**63)


class TrajectoryCohort(NativeTrajectoryCohort):
    """Shared complete count cohort with the LFADS capacity check."""

    def verify_model(self, model):
        parameters = sum(int(np.prod(v.shape.as_list())) for v in model._full.model.train_vars)
        if parameters != self.config['expected_parameters']:
            raise ValueError('full-capacity LFADS parameter count differs')
        return parameters


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _alias(output, name, target):
    alias = output / name
    if alias.exists() and not alias.is_symlink():
        raise ValueError('checkpoint alias would overwrite a non-symlink path')
    temporary = output / ('.' + name + '_' + Path(target).name + '.writing')
    if temporary.is_symlink():
        temporary.unlink()
    os.symlink(target, temporary)
    os.replace(temporary, alias)


def _checkpoint_directory(output, relative):
    path = Path(relative)
    if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != 'checkpoints' or path.parts[1] in ('.','..'):
        raise ValueError('checkpoint reference must remain inside this run')
    directory = output / path
    if directory.is_symlink() or directory.resolve().parent != (output / 'checkpoints').resolve():
        raise ValueError('checkpoint directory escaped this run')
    return directory


def _verify_checkpoint(output, relative):
    directory = _checkpoint_directory(output, relative)
    state = json.loads((directory / 'training.json').read_text())
    for name, digest in state['model_artifacts'].items():
        if name not in ('model.json','weights.npz') or sha256_file(directory / name) != digest:
            raise ValueError('committed LFADS checkpoint changed')
    if set(state['model_artifacts']) != {'model.json','weights.npz'}:
        raise ValueError('complete model/optimizer checkpoint artifacts required')
    return directory, state


def _commit(output, model, binding, state, attempt, improved):
    relative = f"checkpoints/epoch_{state['next_epoch']:04d}_attempt_{attempt:04d}"
    directory = output / relative
    model.save(directory)
    committed = dict(state, binding=binding, checkpoint_directory=relative,
        training_updates=model.training_updates, execution_device=model.execution_device,
        model_artifacts={name:sha256_file(directory / name) for name in ('model.json','weights.npz')})
    if improved:
        committed['best_checkpoint'] = relative
    freeze_json(directory / 'training.json', committed)
    # The latest alias is the commit point. A stale best alias can be repaired
    # from this committed state after interruption.
    _alias(output, 'latest', relative)
    if committed['best_checkpoint'] is not None:
        _alias(output, 'best', committed['best_checkpoint'])
    # One full checkpoint is about half a GB at 100k units. Preserve the
    # authoritative latest and selected best; epoch curves and logs are kept.
    keep = {committed['checkpoint_directory'], committed['best_checkpoint']}
    for old in (output / 'checkpoints').iterdir():
        relative_old = str(old.relative_to(output))
        if old.is_symlink() or not old.is_dir() or relative_old in keep:
            continue
        metadata = old / 'training.json'
        if not metadata.exists():
            continue  # Preserve interrupted writes for failure diagnosis.
        old_state = json.loads(metadata.read_text())
        if old_state['binding'] == binding and old_state['next_epoch'] <= committed['next_epoch']:
            shutil.rmtree(old)
    return committed


def _binding(cohort, config, seed):
    package = Path(__file__).parent
    sources = {name:sha256_file(package / name) for name in
        ('lfads_trajectory_training.py','lfads.py','native_count_windows.py','native_renewal_data.py','poisson_trajectory_training.py','native_trajectory_cohort.py')}
    sources.update({'external/'+name:sha256_file(package.parent / 'external/lfads' / name)
                    for name in ('lfads.py','distributions.py','utils.py')})
    return dict(config=config, seed=seed, **cohort.binding, source_revision=SOURCE_REVISION,
        sources=sources, training_random_schema=TRAINING_RANDOM_SCHEMA,
        validation_random_schema=VALIDATION_RANDOM_SCHEMA,
        backend=dict(tensorflow=tf.__version__, numpy=np.__version__,
                     nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE')))


# The existing 20-epoch trainer is the only pre-continuation revision admitted.
LEGACY_TRAINER_SHA256 = '4fa7d6176079958e0627776f3abb3ff03a8860d90bd1e194cec7789d9294f1b1'
CONTINUATION_SCHEMA = 'neural_population_lfads_budget_continuation_v1'


def _execution_options(output, resume, resume_from, total_epochs, parent_checkpoint_sha256, stop_after_epochs):
    if resume and resume_from is not None:
        raise ValueError('ordinary resume and parent continuation are mutually exclusive')
    if resume_from is None and (total_epochs is not None or parent_checkpoint_sha256 is not None):
        raise ValueError('total epochs and parent checksum require explicit resume-from')
    if resume_from is not None:
        parent = Path(resume_from).resolve()
        if type(total_epochs) is not int or total_epochs < 1:
            raise ValueError('explicit positive prospective total-epochs required')
        if output.resolve() == parent or parent in output.resolve().parents:
            raise ValueError('continuation output must not modify the parent run')
        if output.exists():
            raise FileExistsError('continuation requires a new output path')
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError('positive invocation epoch boundary required')


def _source_transition(parent, current):
    """Budget-only continuation changes the trainer, never numerical/model code."""
    if parent == current:
        return {}
    old, new = parent.get('sources', {}), current.get('sources', {})
    expected = dict(current, sources=dict(new, **{'lfads_trajectory_training.py': LEGACY_TRAINER_SHA256}))
    if parent != expected:
        raise ValueError('parent config/data/adapter/numerical source/backend binding changed')
    return {'lfads_trajectory_training.py': dict(parent=old['lfads_trajectory_training.py'],
                                                current=new['lfads_trajectory_training.py'])}


def _history_digest(history):
    return hashlib.sha256((json.dumps(history, indent=2, allow_nan=False) + '\n').encode()).hexdigest()


def _checkpoint_budget(state, config):
    base = config['maximum_epochs']
    effective = state.get('effective_maximum_epochs', base)
    continuation = state.get('continuation')
    if (type(effective) is not int or effective < base
            or (continuation is None and effective != base)
            or (continuation is not None and (
                not isinstance(continuation, dict)
                or continuation.get('schema') != CONTINUATION_SCHEMA
                or continuation.get('total_epochs') != effective
                or continuation.get('base_maximum_epochs') != base
                or effective <= continuation.get('parent_effective_maximum_epochs', effective)))):
        raise ValueError('LFADS checkpoint continuation budget provenance invalid')
    return effective, continuation


def _verify_continuation(output, state):
    effective, continuation = _checkpoint_budget(state, state['binding']['config'])
    if continuation is None:
        if state.get('continuation_sha256') is not None:
            raise ValueError('LFADS continuation checksum lacks a declared lineage')
        return effective, None
    path = output / 'continuation.json'
    if (sha256_file(path) != state.get('continuation_sha256')
            or json.loads(path.read_text()) != continuation
            or continuation['current_binding'] != state['binding']
            or continuation['current_source_hashes'] != state['binding']['sources']
            or continuation['parent_source_hashes'] != continuation['parent_binding']['sources']
            or continuation['source_migration'] != _source_transition(continuation['parent_binding'], state['binding'])
            or continuation['inherited_early_stopping_patience'] != state['binding']['config']['early_stopping_patience']):
        raise ValueError('LFADS frozen continuation lineage or source binding changed')
    prefix = continuation['parent_completed_epochs']
    if (type(prefix) is not int or prefix < 1 or prefix > continuation['parent_effective_maximum_epochs']
            or continuation['parent_best_epoch'] not in range(prefix)
            or continuation['inherited_stale_epochs'] != prefix - continuation['parent_best_epoch'] - 1):
        raise ValueError('LFADS inherited epoch, best or patience boundary changed')
    # A selected checkpoint inherited from the parent can precede the boundary.
    if state['next_epoch'] >= prefix:
        if _history_digest(state['history'][:prefix]) != continuation['parent_history_sha256']:
            raise ValueError('LFADS inherited history prefix changed')
    else:
        inherited = state.get('inherited_parent_checkpoint')
        if (not inherited or inherited['relative_path'] != continuation['parent_best_checkpoint']
                or inherited['training_sha256'] != continuation['parent_best_training_sha256']
                or state['next_epoch'] != continuation['parent_best_epoch'] + 1
                or state['model_artifacts'] != continuation['parent_best_model_artifacts']):
            raise ValueError('LFADS inherited selected checkpoint lineage changed')
    return effective, continuation


def _verified_run(training, fitting_episodes, *, require_fitted=True):
    """Verify committed source-bound history and effective budget without a graph."""
    run = json.loads((training / 'run.json').read_text())
    allowed = ('FITTED_DEVELOPMENT',) if require_fitted else ('FITTED_DEVELOPMENT', 'CHECKPOINTED_DEVELOPMENT')
    if run.get('status') not in allowed:
        raise ValueError('terminal FITTED_DEVELOPMENT LFADS training required' if require_fitted
                         else 'terminal committed LFADS parent required')
    saved = run['binding']
    config, seed = saved['config'], saved['seed']
    for alias in ('latest', 'best'):
        path = training / alias
        if not path.is_symlink() or str(path.readlink()) != run.get(alias + '_checkpoint'):
            raise ValueError('selected/latest LFADS checkpoint alias or lineage changed')
    latest_directory, latest = _verify_checkpoint(training, run['latest_checkpoint'])
    selected_directory, selected = _verify_checkpoint(training, run['best_checkpoint'])
    effective, continuation = _verify_continuation(training, latest)
    selected_effective, selected_continuation = _verify_continuation(training, selected)
    history, epochs = latest['history'], run['completed_epochs']
    if (latest['binding'] != saved or selected['binding'] != saved
            or sha256_file(latest_directory / 'training.json') != run['latest_training_sha256']
            or json.loads((training / 'epochs.json').read_text()) != history
            or type(epochs) is not int or not 1 <= epochs <= effective
            or latest['next_epoch'] != epochs or len(history) != epochs
            or [row['epoch'] for row in history] != list(range(epochs))
            or run['training_seed'] != seed or run['training_updates'] != epochs * fitting_episodes
            or latest['training_updates'] != run['training_updates']
            or run.get('effective_maximum_epochs', config['maximum_epochs']) != effective
            or run.get('continuation') != continuation or selected_effective != effective
            or selected_continuation != continuation
            or run.get('continuation_sha256') != latest.get('continuation_sha256')):
        raise ValueError('committed LFADS epoch history, source binding or training receipt changed')
    values = np.asarray([row['validation']['source_cost'] for row in history], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('finite LFADS checkpoint-selection objective required')
    best_epoch = int(values.argmin())
    if (run['best_epoch'] != best_epoch or latest['best_epoch'] != best_epoch
            or selected['next_epoch'] != best_epoch + 1 or selected['history'] != history[:best_epoch + 1]
            or selected['training_updates'] != (best_epoch + 1) * fitting_episodes
            or latest['best_checkpoint'] != run['best_checkpoint']
            or selected['best_checkpoint'] != run['best_checkpoint']
            or latest['best'] != float(values[best_epoch]) or selected['best'] != float(values[best_epoch])
            or latest['stale'] != epochs - best_epoch - 1 or run['stale_epochs'] != latest['stale']
            or run['best_validation_source_cost'] != float(values[best_epoch])):
        raise ValueError('selected LFADS checkpoint does not match the registered validation minimum')
    patience_stop = latest['stale'] >= config['early_stopping_patience']
    completed = patience_stop or epochs == effective
    expected_stop = 'validation_patience' if patience_stop else 'epoch_budget' if completed else 'invocation_epoch_boundary'
    if (run['stop_reason'] != expected_stop
            or (run['status'] == 'FITTED_DEVELOPMENT') != completed):
        raise ValueError('LFADS fit did not terminate at its declared budget, patience or invocation boundary')
    return run, latest_directory, latest, selected_directory, selected


def _prepare_continuation(parent, total_epochs, binding, fitting_episodes, declared_digest=None):
    run, latest_dir, latest, selected_dir, selected = _verified_run(parent, fitting_episodes, require_fitted=False)
    migration = _source_transition(run['binding'], binding)
    parent_budget, _ = _checkpoint_budget(latest, binding['config'])
    if total_epochs <= max(parent_budget, latest['next_epoch']):
        raise ValueError('prospective total-epochs must exceed parent completed and declared ceilings')
    if latest['stale'] >= binding['config']['early_stopping_patience']:
        raise ValueError('parent already exhausted inherited validation patience; budget extension cannot reset it')
    digest = sha256_file(latest_dir / 'training.json')
    if declared_digest is not None and declared_digest != digest:
        raise ValueError('parent latest checkpoint checksum differs')
    artifacts = {name: sha256_file(parent / name) for name in
                 ('run.json', 'epochs.json', 'optimization.csv', 'validation.csv') if (parent / name).is_file()}
    continuation = dict(schema=CONTINUATION_SCHEMA, total_epochs=total_epochs,
        base_maximum_epochs=binding['config']['maximum_epochs'], parent_effective_maximum_epochs=parent_budget,
        parent_completed_epochs=latest['next_epoch'], parent_best_epoch=latest['best_epoch'],
        parent_run=str(parent.resolve()), parent_checkpoint_sha256=digest,
        parent_latest_checkpoint=run['latest_checkpoint'], parent_best_checkpoint=run['best_checkpoint'],
        parent_best_training_sha256=sha256_file(selected_dir / 'training.json'),
        parent_latest_model_artifacts=latest['model_artifacts'], parent_best_model_artifacts=selected['model_artifacts'],
        parent_binding=run['binding'], current_binding=binding,
        parent_source_hashes=run['binding']['sources'], current_source_hashes=binding['sources'],
        source_migration=migration, parent_artifact_sha256=artifacts,
        parent_history_sha256=_history_digest(latest['history']), history_prefix_epochs=latest['next_epoch'],
        inherited_early_stopping_patience=binding['config']['early_stopping_patience'],
        inherited_stale_epochs=latest['stale'],
        training_rng='unchanged source stateless training namespace at saved training_updates; absolute epoch shuffle',
        checkpoint_state='full source variables, Adam moments, learning rate and training counter retained unchanged',
        treatment='optimization_budget_only', scientific_claim_qualified=False)
    return latest, selected, continuation


def _seed_continuation(output, parent, latest, selected, binding, continuation):
    """Copy immutable model/optimizer bytes; only child metadata names its lineage."""
    output.mkdir(parents=True, exist_ok=False)
    freeze_json(output / 'continuation.json', continuation)
    lineage_digest = sha256_file(output / 'continuation.json')
    states = {}
    for old in (selected, latest):
        relative = old['checkpoint_directory']
        if relative in states:
            continue
        source, checked = _verify_checkpoint(parent, relative)
        original_digest = sha256_file(source / 'training.json')
        expected_digest = (continuation['parent_checkpoint_sha256'] if relative == latest['checkpoint_directory']
                           else continuation['parent_best_training_sha256'])
        if checked != old or original_digest != expected_digest:
            raise ValueError('parent checkpoint changed before continuation import')
        target = _checkpoint_directory(output, relative)
        target.mkdir(parents=True, exist_ok=False)
        for name, digest in old['model_artifacts'].items():
            shutil.copyfile(source / name, target / name)
            if sha256_file(target / name) != digest:
                raise ValueError('imported parent model/optimizer checkpoint changed')
        state = dict(old, binding=binding, effective_maximum_epochs=continuation['total_epochs'],
            continuation=continuation, continuation_sha256=lineage_digest,
            inherited_parent_checkpoint=dict(relative_path=relative, training_sha256=original_digest,
                                              binding=old['binding'], model_artifacts=old['model_artifacts']))
        freeze_json(target / 'training.json', state)
        states[relative] = state
    for name in ('optimization.csv', 'validation.csv'):
        if name in continuation['parent_artifact_sha256']:
            if sha256_file(parent / name) != continuation['parent_artifact_sha256'][name]:
                raise ValueError('parent diagnostic log changed before continuation import')
            shutil.copyfile(parent / name, output / name)
    _write_json(output / 'epochs.json', latest['history'])
    _alias(output, 'latest', latest['checkpoint_directory'])
    _alias(output, 'best', latest['best_checkpoint'])
    return output / latest['checkpoint_directory'], states[latest['checkpoint_directory']]



def train(acquisition, fields, config_path, output, seed, resume=False, *, stop_after_epochs=None,
          resume_from=None, total_epochs=None, parent_checkpoint_sha256=None):
    acquisition, fields, config_path, output = map(Path, (acquisition, fields, config_path, output))
    resume_from = None if resume_from is None else Path(resume_from)
    _execution_options(output, resume, resume_from, total_epochs, parent_checkpoint_sha256, stop_after_epochs)
    config = yaml.safe_load(config_path.read_text())
    validate_config(config, seed)
    if output.exists() and not resume:
        raise FileExistsError('existing LFADS output requires explicit resume')
    cohort = TrajectoryCohort(acquisition, fields, config)
    binding = _binding(cohort, config, seed)
    previous = json.loads((output / 'run.json').read_text()) if resume else None
    state = dict(next_epoch=0, history=[], best=None, best_epoch=None, stale=0, best_checkpoint=None)
    directory = None
    continuation = None
    effective_epochs = config['maximum_epochs']
    parent_state = parent_selected = None
    if resume_from is not None:
        parent_state, parent_selected, continuation = _prepare_continuation(
            resume_from, total_epochs, binding, len(cohort.fit), parent_checkpoint_sha256)
        effective_epochs = total_epochs
    if resume:
        if not (output / 'latest').is_symlink():
            raise ValueError('no committed LFADS checkpoint is available for resume')
        directory, state = _verify_checkpoint(output, str((output / 'latest').readlink()))
        if state['binding'] != binding or previous['binding'] != binding:
            raise ValueError('resume LFADS config/data/source/backend binding changed')
        effective_epochs, continuation = _verify_continuation(output, state)
        if (previous.get('effective_maximum_epochs', config['maximum_epochs']) != effective_epochs
                or previous.get('continuation') != continuation
                or previous.get('continuation_sha256') != state.get('continuation_sha256')):
            raise ValueError('resume LFADS effective budget or continuation lineage changed')
        if (state['training_updates'] != state['next_epoch'] * len(cohort.fit)
                or len(state['history']) != state['next_epoch']
                or [row['epoch'] for row in state['history']] != list(range(state['next_epoch']))):
            raise ValueError('committed epoch/history/training-update boundary differs')
        if (previous.get('completed_epochs') == state['next_epoch'] and previous.get('latest_training_sha256') is not None
                and previous['latest_training_sha256'] != sha256_file(directory / 'training.json')):
            raise ValueError('committed training-state receipt changed')
        if state['best_checkpoint'] is not None:
            _, selected = _verify_checkpoint(output, state['best_checkpoint'])
            if (selected['binding'] != binding or selected['next_epoch'] != state['best_epoch'] + 1
                    or _verify_continuation(output, selected) != (effective_epochs, continuation)):
                raise ValueError('selected LFADS checkpoint identity differs')
    if resume_from is not None:
        directory, state = _seed_continuation(output, resume_from, parent_state, parent_selected, binding, continuation)
    attempt = previous['attempt'] + 1 if previous else 0
    output.mkdir(parents=True, exist_ok=True)
    (output / 'attempts').mkdir(exist_ok=True)
    started = time.perf_counter()
    report = dict(status='RUNNING', phase='graph_construction', binding=binding, training_seed=seed,
        attempt=attempt, resumed=resume or resume_from is not None, completed_epochs=state['next_epoch'],
        resume_from=str(resume_from.resolve()) if resume_from is not None else None,
        effective_maximum_epochs=effective_epochs, continuation=continuation,
        continuation_sha256=state.get('continuation_sha256'),
        fitting_episodes=len(cohort.fit), selection_episodes=len(cohort.validation), units=len(cohort.unit_ids),
        history_bins=cohort.history_bins, horizon_bins=cohort.horizon_bins,
        maximum_epochs=config['maximum_epochs'], selection_metric=config['selection_metric'],
        checkpoint_retention='latest_and_best',
        selection_interpretation='full-response posterior objective within LFADS; not free forecasting or cross-model likelihood',
        validation_namespace='counter=-(validation_seed+1); training counters are nonnegative',
        formal_training_completed=False, generative_fidelity_qualified=False, convergence_established=False,
        calibration_outcomes_accessed=False, test_outcomes_accessed=False,
        started_utc=datetime.now(timezone.utc).isoformat())
    def receipt():
        report['elapsed_seconds'] = time.perf_counter() - started
        _write_json(output / 'attempts' / f'attempt_{attempt:04d}.json', report)
        _write_json(output / 'run.json', report)
    receipt()
    model = None
    try:
        if directory is None:
            model = LFADSConditionalSimulator(list(map(int,cohort.unit_ids)), 1, cohort.history_bins,
                cohort.horizon_bins, cohort.bin_seconds, config['model'], initialization_seed=seed, device=config['device'])
        else:
            model = LFADSConditionalSimulator.load(directory, device=config['device'])
        if (model.loaded_for_inference or tuple(model.unit_ids) != tuple(map(int,cohort.unit_ids))
                or (model.history_bins,model.horizon_bins) != (cohort.history_bins,cohort.horizon_bins)
                or model.model_config != config['model'] or model.initialization_seed != seed
                or model.training_updates != state['next_epoch'] * len(cohort.fit)):
            raise ValueError('full LFADS checkpoint configuration or training RNG boundary differs')
        report.update(parameters=cohort.verify_model(model), execution_device=model.execution_device)
        if directory is None:
            report['phase'] = 'initial_checkpoint'; receipt()
            state = _commit(output, model, binding, state, attempt, False)
        elif state['best_checkpoint'] is not None:
            _alias(output, 'best', state['best_checkpoint'])
        report.update(latest_checkpoint=state['checkpoint_directory'],
            latest_training_sha256=sha256_file(output / state['checkpoint_directory'] / 'training.json'))
        receipt()
        end = effective_epochs
        if stop_after_epochs is not None:
            end = min(end, state['next_epoch'] + stop_after_epochs)
        with (output / 'optimization.csv').open('a',newline='') as fit_log, (output / 'validation.csv').open('a',newline='') as val_log:
            fit_writer = csv.DictWriter(fit_log, fieldnames=['attempt','epoch','case_id',*LOSS_TERMS,'gradient_norm','seconds'])
            val_writer = csv.DictWriter(val_log, fieldnames=['attempt','epoch','case_id','draw','validation_seed','validation_counter',*LOSS_TERMS,'seconds'])
            if fit_log.tell() == 0: fit_writer.writeheader()
            if val_log.tell() == 0: val_writer.writeheader()
            for epoch in range(state['next_epoch'], end):
                if state['stale'] >= config['early_stopping_patience']: break
                order = cohort.fit.copy(); random.Random(seed * 100000 + epoch).shuffle(order)
                trained, evaluated = [], []
                for case in order:
                    report.update(phase='optimization',active_epoch=epoch,active_case_id=case['case_id']); receipt()
                    counts, stimulus = cohort.window(case)
                    call_start = time.perf_counter()
                    values = model.fit_step(counts[None],stimulus[None]); trained.append(values)
                    fit_writer.writerow(dict(attempt=attempt,epoch=epoch,case_id=case['case_id'],**values,seconds=time.perf_counter()-call_start)); fit_log.flush()
                    del counts, stimulus
                for case in cohort.validation:
                    report.update(phase='posterior_validation',active_epoch=epoch,active_case_id=case['case_id']); receipt()
                    counts, stimulus = cohort.window(case)
                    for draw in range(config['validation_draws']):
                        key = validation_seed(config['validation_seed_base'],seed,case['case_id'],draw)
                        call_start = time.perf_counter()
                        values = model.validation_loss(counts[None],stimulus[None],validation_seed=key); evaluated.append(values)
                        val_writer.writerow(dict(attempt=attempt,epoch=epoch,case_id=case['case_id'],draw=draw,
                            validation_seed=key,validation_counter=-(key+1),**values,seconds=time.perf_counter()-call_start)); val_log.flush()
                    del counts, stimulus
                row = dict(epoch=epoch,training={key:float(np.mean([r[key] for r in trained])) for key in (*LOSS_TERMS,'gradient_norm')},
                           validation={key:float(np.mean([r[key] for r in evaluated])) for key in LOSS_TERMS})
                value = row['validation']['source_cost']
                improved = state['best'] is None or value < state['best']
                candidate = dict(state,next_epoch=epoch+1,history=state['history']+[row],
                    best=value if improved else state['best'], best_epoch=epoch if improved else state['best_epoch'],
                    stale=0 if improved else state['stale']+1)
                report.update(phase='epoch_checkpoint',active_case_id=None); receipt()
                state = _commit(output,model,binding,candidate,attempt,improved)
                report.update(completed_epochs=state['next_epoch'],best_epoch=state['best_epoch'],
                    best_validation_source_cost=state['best'],stale_epochs=state['stale'],
                    latest_checkpoint=state['checkpoint_directory'],best_checkpoint=state['best_checkpoint'],
                    latest_training_sha256=sha256_file(output / state['checkpoint_directory'] / 'training.json'))
                _write_json(output / 'epochs.json',state['history']); receipt()
                print(json.dumps(row,allow_nan=False),flush=True)
        completed = state['next_epoch'] >= effective_epochs or state['stale'] >= config['early_stopping_patience']
        report.update(status='FITTED_DEVELOPMENT' if completed else 'CHECKPOINTED_DEVELOPMENT',phase='stopped',
            stop_reason='validation_patience' if state['stale'] >= config['early_stopping_patience'] else 'epoch_budget' if completed else 'invocation_epoch_boundary',
            completed_epochs=state['next_epoch'],best_epoch=state['best_epoch'],best_validation_source_cost=state['best'],
            stale_epochs=state['stale'],training_updates=model.training_updates,best_checkpoint=state['best_checkpoint'])
        _write_json(output / 'epochs.json',state['history'])
    except BaseException as error:
        report.update(status='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED',
            error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc(),
            recovery='resume from latest committed epoch; partial-attempt rows remain diagnostic')
        receipt(); raise
    finally:
        if model is not None: model.close()
    report['finished_utc'] = datetime.now(timezone.utc).isoformat(); receipt()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('acquisition','fields','config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=[41,42,43],required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--resume-from',type=Path,help='immutable parent run for a new budget-only descendant')
    parser.add_argument('--total-epochs',type=int,help='prospective total ceiling; frozen base config remains unchanged')
    parser.add_argument('--parent-checkpoint-sha256',help='optional declared hash of the parent latest training.json')
    parser.add_argument('--stop-after-epochs',type=int,help='committed invocation boundary; does not extend the registered budget')
    args = parser.parse_args()
    print(json.dumps(train(args.acquisition,args.fields,args.config,args.output,args.seed,args.resume,stop_after_epochs=args.stop_after_epochs,
        resume_from=args.resume_from,total_epochs=args.total_epochs,parent_checkpoint_sha256=args.parent_checkpoint_sha256),allow_nan=False))


if __name__ == '__main__':
    main()
