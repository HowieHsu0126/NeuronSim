"""Bounded indexed SONATA lookup for three pre-existing arithmetic witnesses."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import io
import ipaddress
import json
from pathlib import Path
import signal
import sys
import time
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import h5py
import numpy as np
import requests

from .provenance import freeze_json, runtime_manifest, sha256_file
from .v1_source_assets import _range_response, _strong_etag

MAX_BYTES = 64 * 1024 * 1024
MAX_SECONDS = 210.0
BUFFER_BYTES = 65536
ID_CHUNK = 65536
POPULATION = '/edges/v1_to_v1'
EDGE_FIELDS = ('source_node_id', 'target_node_id', 'edge_type_id', 'edge_group_id', 'edge_group_index')
PROPERTIES = ('nsyns', 'syn_weight', 'weight_sigma', 'delay')


def validate_proxy(value):
    parsed = urlparse(value)
    try:
        loopback = ipaddress.ip_address(parsed.hostname or '').is_loopback
        port = parsed.port
    except ValueError as error:
        raise ValueError('SONATA_LOOPBACK_HTTP_PROXY_REQUIRED') from error
    if (parsed.scheme != 'http' or not loopback or port is None or not 1 <= port <= 65535
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
        raise ValueError('SONATA_LOOPBACK_HTTP_PROXY_REQUIRED')
    return value


def validate_limits(max_bytes, max_seconds):
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
        raise ValueError('SONATA_BYTE_LIMIT_MUST_BE_WITHIN_64_MIB')
    if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
            or not np.isfinite(max_seconds) or not 0 < max_seconds <= MAX_SECONDS):
        raise ValueError('SONATA_TIME_LIMIT_MUST_BE_WITHIN_210_SECONDS')


class RangeFile(io.RawIOBase):
    """Sequential HTTP requests; BufferedReader owns the sole small byte buffer."""
    def __init__(self, source, proxy_url, *, max_bytes=MAX_BYTES, max_seconds=MAX_SECONDS, session=None):
        super().__init__()
        validate_limits(max_bytes, max_seconds)
        validate_proxy(proxy_url)
        parsed = urlparse(source.get('resolved_url', ''))
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.fragment
                or type(source.get('expected_bytes')) is not int or source['expected_bytes'] < 1
                or not _strong_etag(source)):
            raise ValueError('SONATA_FIXED_HTTPS_SOURCE_IDENTITY_REQUIRED')
        self.source = source
        self.position = 0
        self.max_bytes, self.max_seconds = max_bytes, float(max_seconds)
        self.started = time.monotonic()
        self.body_bytes = 0
        self.requests_made = 0
        self.requested_bytes = 0
        self.failed = None
        self.redirects = 0
        self.redirect_body_bytes = 0
        self.body_accounting_is_upper_bound = False
        self.published_url = source.get('published_url')
        if self.published_url:
            published = urlparse(self.published_url)
            if (published.scheme != 'https' or not published.hostname
                    or published.username is not None or published.password is not None
                    or published.fragment):
                raise ValueError('SONATA_PUBLISHED_HTTPS_SOURCE_REQUIRED')
        self.request_nonce = str(time.time_ns())
        self.session = requests.Session() if session is None else session
        self.session.trust_env = False
        self.session.max_redirects = 3
        self.session.proxies = {'http': proxy_url, 'https': proxy_url}

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def _check(self):
        if self.closed:
            raise ValueError('I/O operation on closed SONATA range file')
        if self.failed is not None:
            raise OSError('SONATA_PREVIOUS_HTTP_OR_BUDGET_FAILURE:' + self.failed)
        remaining = self.max_seconds-(time.monotonic()-self.started)
        if remaining <= 0:
            raise TimeoutError('SONATA_ELAPSED_TIME_LIMIT')
        return remaining

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self._check()
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position+offset
        elif whence == io.SEEK_END:
            position = self.source['expected_bytes']+offset
        else:
            raise ValueError('SONATA_INVALID_SEEK_WHENCE')
        if not isinstance(position, int) or position < 0:
            raise ValueError('SONATA_INVALID_SEEK_POSITION')
        self.position = position
        return position

    def _range_url(self, start, stop):
        if not self.published_url:
            return self.source['resolved_url']
        parsed = urlparse(self.published_url)
        query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                 if key not in ('dl', 'nrnsim_range')]
        # Published-entry requests work where detached signed URLs return 403.
        # A unique range URL also avoids reusing an expired cached redirect.
        query.extend([('dl', '1'), ('nrnsim_range', f'{self.request_nonce}-{start}-{stop}')])
        return urlunparse(parsed._replace(query=urlencode(query)))

    def _redirect_response(self, response, **kwargs):
        if not response.is_redirect:
            return response
        try:
            self._check()
            destination = urlparse(urljoin(response.url, response.headers['Location']))
            declared = response.headers.get('Content-Length', '')
            if (destination.scheme != 'https' or not destination.hostname
                    or destination.username is not None or destination.password is not None
                    or response.headers.get('Transfer-Encoding')
                    or response.headers.get('Content-Encoding', 'identity') != 'identity'
                    or not declared.isdecimal() or int(declared) > 4096
                    or int(declared) > self.max_bytes-self.body_bytes):
                raise ValueError('SONATA_REDIRECT_HTTPS_OR_BODY_LIMIT_INVALID')
            # Requests consumes redirect bodies even in streaming mode. Check
            # their framed lengths before consumption and charge the same cap.
            expected = int(declared)
            self.body_bytes += expected
            self.redirect_body_bytes += expected
            self.redirects += 1
            try:
                body = response.content
            except BaseException:
                self.body_accounting_is_upper_bound = True
                raise
            if len(body) != expected:
                self.body_accounting_is_upper_bound = True
                raise ValueError('SONATA_REDIRECT_BODY_LENGTH_MISMATCH')
            return response
        except BaseException:
            response.close()
            raise

    def read(self, size=-1):
        try:
            remaining_time = self._check()
            available = max(0, self.source['expected_bytes']-self.position)
            size = available if size is None or size < 0 else min(size, available)
            if not size:
                return b''
            if size > self.max_bytes-self.body_bytes:
                raise ValueError('SONATA_HTTP_BODY_BUDGET_EXCEEDED_BEFORE_FETCH')
            start, stop = self.position, self.position+size-1
            self.requests_made += 1
            self.requested_bytes += size
            response = self.session.get(self._range_url(start, stop),
                headers={'Range': f'bytes={start}-{stop}', 'Accept-Encoding': 'identity',
                         'If-Match': self.source['etag'], 'Cache-Control': 'no-cache'},
                proxies=self.session.proxies, verify=True, stream=True, allow_redirects=bool(self.published_url),
                hooks={'response': self._redirect_response},
                timeout=(min(10.0, remaining_time), min(20.0, remaining_time)))
            with response:
                self._check()
                _range_response(response, self.source, start, stop, ranged=True, metadata=True)
                # The receipt may have an unquoted ETag; require equality for every request.
                if response.headers.get('ETag') != self.source['etag']:
                    raise ValueError('SONATA_SOURCE_ETAG_CHANGED')
                declared = response.headers.get('Content-Length')
                if declared is None or not declared.isdecimal() or int(declared) != size:
                    raise ValueError('SONATA_HTTP_CONTENT_LENGTH_MISMATCH')
                if response.headers.get('Transfer-Encoding'):
                    raise ValueError('SONATA_HTTP_TRANSFER_ENCODING_UNEXPECTED')
                if size > self.max_bytes-self.body_bytes:
                    raise ValueError('SONATA_HTTP_BODY_BUDGET_EXCEEDED_AFTER_REDIRECT')
                response.raw.decode_content = False
                body = bytearray()
                while len(body) < size:
                    self._check()
                    block = response.raw.read(min(BUFFER_BYTES, size-len(body)))
                    if not block:
                        raise ValueError('SONATA_HTTP_BODY_TRUNCATED')
                    self.body_bytes += len(block)
                    if self.body_bytes > self.max_bytes or len(body)+len(block) > size:
                        raise ValueError('SONATA_HTTP_BODY_LENGTH_OR_BUDGET_EXCEEDED')
                    body.extend(block)
                self._check()
            self.position += size
            return bytes(body)
        except BaseException as error:
            # Retaining a traceback here can keep failed HDF5 construction
            # frames alive after their underlying Python file has closed.
            self.failed = type(error).__name__
            raise

    def readinto(self, buffer):
        target = memoryview(buffer).cast('B')
        data = self.read(len(target))
        target[:len(data)] = data
        return len(data)

    def counters(self):
        return dict(response_body_bytes=self.body_bytes, requests=self.requests_made,
                    redirects=self.redirects, redirect_body_bytes=self.redirect_body_bytes,
                    body_accounting_is_upper_bound=self.body_accounting_is_upper_bound,
                    url_source='published_url' if self.published_url else 'resolved_url',
                    requested_bytes=self.requested_bytes, elapsed_seconds=time.monotonic()-self.started,
                    max_bytes=self.max_bytes, max_seconds=self.max_seconds)

    def close(self):
        if not self.closed:
            try:
                self.session.close()
            finally:
                super().close()


@contextmanager
def wall_deadline(seconds):
    """Interrupt stalled network/HDF5 work so an inactivity timeout cannot extend the query."""
    old_handler = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError('SONATA_EXISTING_PROCESS_TIMER_NOT_SUPPORTED')
    def expired(signum, frame):
        raise TimeoutError('SONATA_WALL_TIME_LIMIT')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _integer_dataset(group, name, columns=None):
    if name not in group or not isinstance(group[name], h5py.Dataset):
        raise ValueError('SONATA_REQUIRED_DATASET_MISSING:' + group.name + '/' + name)
    dataset = group[name]
    expected_dim = 1 if columns is None else 2
    if (dataset.ndim != expected_dim or dataset.dtype.kind not in 'iu'
            or (columns is not None and dataset.shape[1] != columns)):
        raise ValueError('SONATA_INTEGER_DATASET_SCHEMA_INVALID:' + dataset.name)
    return dataset


def _property(group, name, index):
    in_dataset, in_attribute = name in group, name in group.attrs
    if in_dataset and in_attribute:
        return dict(status='INVALID', reason='Both dataset and attribute declare the property.')
    if not in_dataset and not in_attribute:
        return dict(status='MISSING')
    if in_dataset:
        dataset = group[name]
        if not isinstance(dataset, h5py.Dataset) or dataset.dtype.kind not in 'iuf':
            return dict(status='INVALID', reason='Numeric scalar property required.')
        if dataset.ndim == 0:
            value, location = dataset[()], 'group_scalar_dataset'
        elif dataset.ndim == 1 and index < dataset.shape[0]:
            value, location = dataset[index], 'edge_group_index'
        else:
            return dict(status='INVALID', reason='Property shape or group index invalid.',
                        shape=list(dataset.shape), dtype=str(dataset.dtype))
    else:
        value, location = group.attrs[name], 'group_attribute'
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in 'iuf':
        return dict(status='INVALID', reason='Numeric scalar property required.', dtype=str(array.dtype))
    scalar = array.item()
    valid = bool(np.isfinite(scalar))
    if name == 'nsyns':
        valid = valid and array.dtype.kind in 'iu' and scalar > 0
    elif name == 'weight_sigma':
        valid = valid and scalar > 0
    elif name == 'delay':
        valid = valid and scalar >= 0
    return dict(status='RECORDED' if valid else 'INVALID',
        value=scalar if np.isfinite(scalar) else None, raw_repr=repr(scalar),
        dtype=str(array.dtype), location=location)


def query_indexed(handle, selectors, report):
    if POPULATION not in handle or not isinstance(handle[POPULATION], h5py.Group):
        raise ValueError('SONATA_V1_TO_V1_POPULATION_MISSING')
    group = handle[POPULATION]
    report['population_fields'] = {name: dict(kind='dataset', shape=list(value.shape), dtype=str(value.dtype))
        if isinstance(value, h5py.Dataset) else dict(kind='group') for name, value in group.items()}
    fields = {name: _integer_dataset(group, name) for name in EDGE_FIELDS}
    denominator = len(fields['source_node_id'])
    if any(len(dataset) != denominator for dataset in fields.values()):
        raise ValueError('SONATA_EDGE_FIELD_DENOMINATORS_DIFFER')
    report['source_edge_denominator'] = denominator
    index_paths = [name+'/target_to_source' for name in ('indices', 'indicies')
                   if name+'/target_to_source' in group]
    if not index_paths:
        raise ValueError('SONATA_TARGET_INDEX_MISSING_NO_FULL_SCAN_ALLOWED')
    if len(index_paths) != 1 or not isinstance(group[index_paths[0]], h5py.Group):
        raise ValueError('SONATA_TARGET_INDEX_AMBIGUOUS_OR_INVALID')
    index_path = index_paths[0]
    report['target_index_path'] = index_path
    index = group[index_path]
    nodes_to_ranges = _integer_dataset(index, 'node_id_to_range', columns=2)
    ranges_to_edges = _integer_dataset(index, 'range_to_edge_id', columns=2)
    report['target_index'] = {name: dict(shape=list(index[name].shape), dtype=str(index[name].dtype))
                              for name in ('node_id_to_range', 'range_to_edge_id')}
    report['property_groups'] = {}
    report['cases'] = []
    for selector in selectors:
        case = dict(selector, status='INCOMPLETE', matches_complete=False, matching_rows=[], scanned_source_rows=0)
        report['cases'].append(case)
        if report.get('include_target_context', False):
            case['target_context'] = dict(complete=False, incoming_rows=[], edge_type_id_counts={})
        target, source = selector['target_id'], selector['source_id']
        if target >= len(nodes_to_ranges):
            raise ValueError('SONATA_TARGET_OUTSIDE_INDEX:' + str(target))
        start, stop = map(int, nodes_to_ranges[target])
        if not 0 <= start <= stop <= len(ranges_to_edges):
            raise ValueError('SONATA_TARGET_RANGE_INTERVAL_INVALID')
        case['index_range_interval'] = [start, stop]
        intervals = []
        for offset in range(start, stop, ID_CHUNK):
            for left, right in ranges_to_edges[offset:min(stop, offset+ID_CHUNK)]:
                left, right = int(left), int(right)
                if not 0 <= left <= right <= denominator:
                    raise ValueError('SONATA_EDGE_RANGE_OUTSIDE_DENOMINATOR')
                intervals.append((left, right))
        intervals.sort()
        if any(intervals[i][0] < intervals[i-1][1] for i in range(1, len(intervals))):
            raise ValueError('SONATA_OVERLAPPING_INDEX_RANGES')
        case['edge_interval_count'] = len(intervals)
        for left, right in intervals:
            for offset in range(left, right, ID_CHUNK):
                stop_row = min(right, offset+ID_CHUNK)
                candidates = fields['source_node_id'][offset:stop_row]
                case['scanned_source_rows'] += len(candidates)
                if 'target_context' in case:
                    targets = fields['target_node_id'][offset:stop_row]
                    kinds = fields['edge_type_id'][offset:stop_row]
                    if np.any(targets != target) or np.any(candidates < 0) or np.any(kinds < 0):
                        raise ValueError('SONATA_INCOMING_CONTEXT_TARGET_OR_IDENTIFIER_INVALID')
                    context = case['target_context']
                    context['incoming_rows'].extend(dict(global_row_id=offset+i,
                        source_node_id=int(src), target_node_id=int(dst), edge_type_id=int(kind))
                        for i, (src, dst, kind) in enumerate(zip(candidates, targets, kinds)))
                    for kind, count in zip(*np.unique(kinds, return_counts=True)):
                        key = str(int(kind))
                        context['edge_type_id_counts'][key] = context['edge_type_id_counts'].get(key, 0) + int(count)
                for local in np.flatnonzero(candidates == source):
                    row_id = offset+int(local)
                    row = dict(global_row_id=row_id, **{name: int(dataset[row_id]) for name, dataset in fields.items()})
                    case['matching_rows'].append(row)
                    if row['source_node_id'] != source or row['target_node_id'] != target:
                        raise ValueError('SONATA_INDEXED_ROW_TARGET_OR_SOURCE_MISMATCH')
                    if any(row[name] < 0 for name in EDGE_FIELDS):
                        raise ValueError('SONATA_NEGATIVE_ROW_IDENTIFIER')
                    group_id = str(row['edge_group_id'])
                    if group_id not in group or not isinstance(group[group_id], h5py.Group):
                        raise ValueError('SONATA_EDGE_PROPERTY_GROUP_MISSING:' + group_id)
                    properties = group[group_id]
                    if group_id not in report['property_groups']:
                        report['property_groups'][group_id] = dict(dataset_names=sorted(properties.keys()),
                                                                   attribute_names=sorted(properties.attrs.keys()))
                    row['properties'] = {name: _property(properties, name, row['edge_group_index']) for name in PROPERTIES}
        if 'target_context' in case:
            context = case['target_context']
            if (len(context['incoming_rows']) != case['scanned_source_rows']
                    or sum(context['edge_type_id_counts'].values()) != case['scanned_source_rows']):
                raise ValueError('SONATA_INCOMING_CONTEXT_COUNT_MISMATCH')
            context.update(complete=True, indexed_incoming_rows=case['scanned_source_rows'],
                coverage='All nonoverlapping indexed incoming rows; independent index completeness is not established.')
        case.update(status='COMPLETE', matches_complete=True, matching_row_count=len(case['matching_rows']),
                    source_pair_present=bool(case['matching_rows']))
    report['completed_selectors'] = sum(case['matches_complete'] for case in report['cases'])


def _resume_cases(path, report, selectors):
    if path is None:
        return {}
    previous = json.loads(path.read_text())
    if bool(previous.get('include_target_context', False)) != bool(report.get('include_target_context', False)):
        raise ValueError('SONATA_RESUME_TARGET_CONTEXT_SCOPE_MISMATCH')
    for key in ('comparison_report_sha256', 'witness_report_sha256',
                'source_expected_bytes', 'source_etag', 'population', 'selectors'):
        if previous.get(key) != report.get(key):
            raise ValueError('SONATA_RESUME_SOURCE_OR_WITNESS_MISMATCH:' + key)
    selected = {case['name']: case for case in selectors}
    complete = {}
    seen = set()
    for case in previous.get('cases', []):
        name = case.get('name')
        if name not in selected or name in seen:
            raise ValueError('SONATA_RESUME_CASE_IDENTITY_INVALID')
        seen.add(name)
        if any(case.get(key) != value for key, value in selected[name].items()):
            raise ValueError('SONATA_RESUME_CASE_IDENTITY_INVALID')
        if case.get('matches_complete') is True:
            if (case.get('status') != 'COMPLETE'
                    or case.get('matching_row_count') != len(case.get('matching_rows', []))):
                raise ValueError('SONATA_RESUME_COMPLETE_CASE_INVALID')
            if report.get('include_target_context', False):
                context = case.get('target_context', {})
                if (context.get('complete') is not True
                        or context.get('indexed_incoming_rows') != case.get('scanned_source_rows')
                        or len(context.get('incoming_rows', [])) != case.get('scanned_source_rows')
                        or sum(context.get('edge_type_id_counts', {}).values()) != case.get('scanned_source_rows')):
                    raise ValueError('SONATA_RESUME_TARGET_CONTEXT_INCOMPLETE')
            complete[name] = case
    if len(complete) != previous.get('completed_selectors'):
        raise ValueError('SONATA_RESUME_COMPLETION_COUNT_MISMATCH')
    report['resume_report_sha256'] = sha256_file(path)
    report['reused_complete_selectors'] = [case['name'] for case in selectors if case['name'] in complete]
    report['previous_source_edge_denominator'] = previous.get('source_edge_denominator')
    return complete


def run_query(source_receipt: Path, witness_report: Path, comparison_report: Path,
              proxy_url: str, output: Path, max_bytes=MAX_BYTES, max_seconds=MAX_SECONDS, resume_report: Path | None = None,
              include_target_context: bool = False):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='FAILED_SOURCE_SONATA_WITNESS_QUERY', exit_code=1,
        started_utc=datetime.now(timezone.utc).isoformat(), source_qualification=False,
        scientific_qualification=False, full_asset_acquired=False, production_episodes=0,
        historical_nsyns_inferred=False, population=POPULATION, completed_selectors=0,
        source_edge_denominator=None, include_target_context=include_target_context,
        scope=('All indexed incoming rows of the same three witness targets, plus original pair rows.'
               if include_target_context else 'Indexed source rows for three existing ID-pair witnesses only.'))
    reader = None
    reused = {}
    selectors = []
    try:
        validate_limits(max_bytes, max_seconds)
        receipt = json.loads(source_receipt.read_text())
        witnesses = json.loads(witness_report.read_text())
        comparison = json.loads(comparison_report.read_text())
        comparison_hash = sha256_file(comparison_report)
        if comparison_hash != witnesses.get('comparison_report_sha256'):
            raise ValueError('SONATA_ORIGINAL_COMPARISON_REPORT_HASH_MISMATCH')
        denominator = comparison.get('comparison', {}).get('tuples')
        if type(denominator) is not int or denominator < 1:
            raise ValueError('SONATA_ORIGINAL_COMPARISON_DENOMINATOR_MISSING')
        if witnesses.get('status') != 'SOURCE_WEIGHT_ARITHMETIC_DIAGNOSTIC_COMPLETE' or witnesses.get('exit_code') != 0:
            raise ValueError('SONATA_COMPLETE_ARITHMETIC_WITNESSES_REQUIRED')
        cases = witnesses.get('cases')
        if not isinstance(cases, list) or len(cases) != 3:
            raise ValueError('SONATA_EXACTLY_THREE_WITNESSES_REQUIRED')
        selectors = []
        for case in cases:
            if (not isinstance(case.get('name'), str) or not case['name']
                    or any(type(case.get(key)) is not int or case[key] < 0 for key in ('source_id', 'target_id'))):
                raise ValueError('SONATA_WITNESS_IDENTITY_INVALID')
            selectors.append({key: case[key] for key in ('name', 'source_id', 'target_id', 'raw_group', 'raw_offset')})
        if len({(row['source_id'], row['target_id']) for row in selectors}) != 3:
            raise ValueError('SONATA_THREE_DISTINCT_ID_PAIRS_REQUIRED')
        if type(receipt.get('expected_bytes')) is not int or receipt['expected_bytes'] <= max_bytes:
            raise ValueError('SONATA_BODY_BUDGET_MUST_BE_SMALLER_THAN_FULL_ASSET')
        report.update(source_receipt_sha256=sha256_file(source_receipt), witness_report_sha256=sha256_file(witness_report),
            comparison_report_sha256=comparison_hash, original_comparison_denominator=denominator,
            source_expected_bytes=receipt['expected_bytes'], source_etag=receipt.get('etag'), selectors=selectors,
            limits={'max_bytes': max_bytes, 'max_seconds': max_seconds},
            original_comparison_status=comparison.get('status'))
        reused = _resume_cases(resume_report, report, selectors)
        remaining = [case for case in selectors if case['name'] not in reused]
        if not remaining:
            raise ValueError('SONATA_RESUME_ALREADY_COMPLETE_NO_QUERY_REQUIRED')
        reader = RangeFile(receipt, proxy_url, max_bytes=max_bytes, max_seconds=max_seconds)
        with wall_deadline(max_seconds):
            with reader:
                with io.BufferedReader(reader, buffer_size=BUFFER_BYTES) as buffered:
                    # Fetch the first buffer before entering HDF5 callbacks, so
                    # expired URLs and other initial HTTP errors close normally.
                    if not buffered.peek(1):
                        raise ValueError('SONATA_EMPTY_INITIAL_RESPONSE')
                    with h5py.File(buffered, 'r') as handle:
                        query_indexed(handle, remaining, report)
        fresh = {case['name']: case for case in report['cases']}
        report['cases'] = [fresh.get(case['name'], reused.get(case['name'])) for case in selectors]
        report['completed_selectors'] = sum(case['matches_complete'] for case in report['cases'])
        if (reused and report['source_edge_denominator'] != report['previous_source_edge_denominator']):
            raise ValueError('SONATA_RESUME_SOURCE_DENOMINATOR_CHANGED')
        if report['completed_selectors'] != 3:
            raise ValueError('SONATA_INCOMPLETE_SELECTOR_QUERIES')
        report.update(status='SOURCE_SONATA_WITNESS_QUERY_COMPLETE', exit_code=0,
                      zero_match_selectors=[case['name'] for case in report['cases'] if not case['source_pair_present']])
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        current = {case['name']: case for case in report.get('cases', [])}
        current = {**reused, **current}
        report['cases'] = [current[case['name']] for case in selectors if case['name'] in current]
        report['completed_selectors'] = sum(case.get('matches_complete', False)
                                            for case in report.get('cases', []))
        if reader is not None:
            reader.close()
            report['http'] = reader.counters()
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['runtime'] = runtime_manifest({'max_bytes': max_bytes, 'max_seconds': max_seconds}, list(sys.argv), report['exit_code'])
        freeze_json(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-receipt', 'witness-report', 'comparison-report', 'proxy-url', 'output'):
        parser.add_argument('--'+name, required=True, type=str if name == 'proxy-url' else Path)
    parser.add_argument('--include-target-context', action='store_true')
    parser.add_argument('--resume-report', type=Path)
    parser.add_argument('--max-bytes', type=int, default=MAX_BYTES)
    parser.add_argument('--max-seconds', type=float, default=MAX_SECONDS)
    return run_query(**vars(parser.parse_args()))['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
