"""Qualify tokenless HF Jobs results without inventing a RunPod/local producer.

The caller authenticates provider readback and retrieves the bounded bucket tree.
This module binds that receipt to immutable worker evidence and runs the existing
full-tensor/two-cold-capture qualification gates; it never executes model code.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from . import common, dsformat as F, dsmanifest, dsvalidate, jobcontract, resultsink


class HFQualificationError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise HFQualificationError(message)


def _digest(value):
    return common.sha256_hex(common.canonical_json(value))


def _sealed(document, field, schema):
    _require(isinstance(document, dict) and document.get('schema') == schema,
             'unsupported ' + schema)
    _require(re.fullmatch(r'[0-9a-f]{64}', str(document.get(field, ''))) is not None,
             'missing ' + field)
    blank = dict(document, **{field: ''})
    _require(_digest(blank) == document[field], field + ' does not recompute')


def _read(path):
    from fidelity_dataset import _read_json_file
    return _read_json_file(str(path), str(path))


def _inside(root, relative):
    jobcontract.canonical_relative_path(relative, 'HF result path')
    return Path(F.resolve_inside(str(root), relative, owner='HF Jobs result'))


def _plan_receipt(plan, receipt):
    _sealed(plan, 'plan_sha256', 'qfs.hf-workflow-plan.v1')
    _require(plan.get('mode') in ('root', 'candidate'), 'only capture modes qualify')
    _require(re.fullmatch(r'[0-9a-f]{32}', str(plan.get('workflow_id', ''))) is not None,
             'invalid workflow identity')
    owner = plan.get('owner')
    _require(isinstance(owner, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]*', owner),
             'invalid owner namespace')
    source = plan.get('source') or {}
    _require(source.get('repository') == 'https://github.com/malaiwah/quant-fidelity-suite'
             and re.fullmatch(r'[0-9a-f]{40}', str(source.get('revision', '')))
             and re.fullmatch(r'[0-9a-f]{64}', str(source.get('worker_sha256', ''))),
             'source must identify an immutable QFS worker')
    _require(isinstance(plan.get('image'), str)
             and re.fullmatch(r'.+@sha256:[0-9a-f]{64}', plan['image']),
             'HF Jobs requires immutable image digest')
    _require(isinstance(receipt, dict) and receipt.get('schema') == 'qfs.hf-jobs-execution.v1',
             'unsupported provider receipt')
    hardware = plan.get('hardware') or {}
    _require(type(hardware.get('timeout_seconds')) is int and hardware['timeout_seconds'] > 0
             and hardware.get('device') in ('cpu', 'cuda') and bool(hardware.get('flavor')),
             'invalid requested hardware/timeout')
    for key, expected in {'namespace': owner, 'flavor': hardware['flavor'],
                          'docker_image': plan['image'], 'plan_sha256': plan['plan_sha256'],
                          'source_revision': source['revision'], 'status': 'COMPLETED',
                          'requested_timeout_seconds': hardware['timeout_seconds']}.items():
        _require(receipt.get(key) == expected, 'provider/plan mismatch: ' + key)
    _require(isinstance(receipt.get('job_id'), str) and bool(receipt['job_id'])
             and bool(receipt.get('created_at')) and bool(receipt.get('provider_identity_note')),
             'provider receipt lacks authenticated job identity/provenance')
    output = plan.get('output') or {}
    _require(re.fullmatch(re.escape(owner) + r'/[^\s/]+', str(output.get('dataset_repository', '')))
             and re.fullmatch(re.escape(owner) + r'/[^\s/]+', str(output.get('bucket', '')))
             and output.get('prefix') == 'runs/' + plan['workflow_id'],
             'output namespace differs from verified owner/workflow')
    runtime = plan.get('runtime') or {}
    _require(runtime.get('dtype') == 'bfloat16' and runtime.get('schedule') == 'layer-outer',
             'unsupported scientific runtime')


def _scope(plan):
    raw = plan.get('scope')
    _require(isinstance(raw, dict), 'candidate requires an actual scope object')
    return dsmanifest.scope_block(raw['assignments'], raw['head_policy'],
                                  raw.get('kv_cache_dtype', 'bf16'), raw['policy'])


def validate_execution(job):
    """Called by jobcontract: execution identity is included in the job self-seal."""
    execution = job.get('hf_execution')
    _require(isinstance(execution, dict), 'HF Jobs job lacks execution evidence')
    plan, receipt = execution.get('plan'), execution.get('provider_receipt')
    _plan_receipt(plan, receipt)
    _require(job.get('execution_attempt') == {
        'number': 1, 'kind': 'hf-jobs', 'attempt_id': plan['workflow_id'][:24],
        'job_id': receipt['job_id'], 'namespace': plan['owner']},
        'HF Jobs attempt differs from provider receipt')
    _require(job.get('recipe') == 'hf-jobs'
             and (job.get('produced_by') or {}).get('dependencies', {}).get('provider') == 'hf-jobs'
             and (job.get('environment') or {}).get('image') == plan['image'],
             'HF Jobs execution identity differs from job')
    model = plan['inputs']['model']
    _require(job['target']['repo_id'] == model['repository']
             and job['target']['revision'] == model['revision']
             and job['capture']['dataset_repository'] == plan['output']['dataset_repository']
             and job['capture']['author'] == plan['owner']
             and job['capture']['device'] == plan['hardware']['device'],
             'HF Jobs model/owner/device differs from plan')
    metadata_census = {row['path']: {'bytes': row['bytes'], 'sha256': row['sha256']}
                       for row in model['files']}
    target = _target(model, metadata_census, job['capture'].get('weights_license'))
    _require(all(job['target'].get(key) == value for key, value in target.items()),
             'HF Jobs target census differs from authenticated model metadata')
    bundle = job['bundle']
    sources = {row['path']: row['sha256'] for row in bundle['files']}
    _require(bundle['source'] == plan['source']['revision']
             and sources.get('explorer/job_worker.py') == plan['source']['worker_sha256']
             and job['produced_by'].get('source_files') == sources,
             'HF Jobs measured source identities differ from plan/bundle')
    candidate = job['capture'].get('candidate')
    _require((candidate is not None) == (plan['mode'] == 'candidate'),
             'HF Jobs candidate/root role differs from plan')
    if candidate is not None:
        reference = plan['inputs']['reference']
        _require(candidate['codec'] == plan.get('codec')
                 and candidate['declared_bits'] == plan.get('declared_bits')
                 and candidate['scope']['scope_digest'] == _scope(plan)['scope_digest']
                 and all(candidate['reference'][key] == reference[key]
                         for key in ('repository', 'revision', 'dataset_sha256')),
                 'HF Jobs candidate scope/codec/reference differs from plan')
    for name in ('worker_result_sha256', 'worker_manifest_sha256'):
        _require(re.fullmatch(r'[0-9a-f]{64}', str(execution.get(name, ''))),
                 'missing worker evidence digest: ' + name)
    _require(execution.get('provenance') == _provenance()
             and execution.get('publication_staging') == 'private-recoverable',
             'HF Jobs provenance must distinguish observed/requested/worker evidence')


def _provenance():
    return {
        'provider_observed_fields': ['job_id', 'namespace', 'flavor', 'docker_image',
                                     'status', 'created_at', 'started_at', 'finished_at'],
        'requested_fields': ['plan_sha256', 'source_revision', 'requested_timeout_seconds'],
        'worker_reported_fields': ['source_files', 'checkpoint_files', 'stack_fingerprint',
                                   'container', 'cold_run'],
        'independent_reproduction': False,
    }


def _result(root, plan):
    result = _read(root / 'result.json')
    _sealed(result, 'result_sha256', 'qfs.hf-workflow-result.v1')
    for key in ('workflow_id', 'owner', 'mode', 'plan_sha256'):
        _require(result.get(key) == plan[key], 'worker/plan mismatch: ' + key)
    _require(result.get('status') == 'complete', 'worker result did not complete')
    _require(_read(root / 'plan.json') == plan, 'persisted plan differs from submitted plan')
    rows = result.get('files')
    _require(isinstance(rows, list) and rows, 'worker result has no file manifest')
    seen = set()
    total = 0
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'},
                 'noncanonical result file record')
        path = _inside(root, row['path'])
        _require(row['path'] not in seen and row['path'] != 'result.json'
                 and path.is_file() and not path.is_symlink(), 'duplicate/missing result file')
        seen.add(row['path'])
        _require(type(row['bytes']) is int and row['bytes'] >= 0
                 and path.stat().st_size == row['bytes']
                 and common.sha256_file(str(path)) == row['sha256'],
                 'result file differs from worker manifest: ' + row['path'])
        total += row['bytes']
    _require(total <= plan['limits']['max_output_bytes'], 'result exceeds output bound')
    for required in ('plan.json', 'source-manifest.json'):
        _require(required in seen, 'worker manifest omits ' + required)
    source = _read(root / 'source-manifest.json')
    _require(source.get('schema') == 'qfs.hf-workflow-source.v1', 'invalid worker source manifest')
    for key in ('repository', 'revision'):
        _require(source.get(key) == plan['source'][key]
                 and (result.get('source') or {}).get(key) == source[key],
                 'worker source pin differs from plan')
    source_rows = source.get('source_files')
    bundle = jobcontract.finalize_bundle_manifest(source_rows, plan['source']['revision'])
    _require(bundle['files'] == source_rows, 'noncanonical worker source manifest')
    source_hashes = {row['path']: row['sha256'] for row in source_rows}
    _require(source_hashes.get('explorer/job_worker.py') == plan['source']['worker_sha256'],
             'worker source hash differs from plan')
    result_sources = (result.get('source') or {}).get('source_files')
    _require(result_sources in (source_rows, source_hashes), 'result source inventory differs')
    return result, bundle, seen


def _target(model, census, license_identity):
    _require(isinstance(model, dict) and re.fullmatch(r'[0-9a-f]{40}', str(model.get('revision', ''))),
             'model input needs immutable revision')
    rows = model.get('files')
    _require(isinstance(rows, list) and rows, 'model metadata has no exact file census')
    by_path = {}
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'},
                 'invalid model census record')
        jobcontract.canonical_relative_path(row['path'], 'model census path')
        _require(row['path'] not in by_path and type(row['bytes']) is int and row['bytes'] >= 0
                 and re.fullmatch(r'[0-9a-f]{64}', str(row['sha256'])), 'invalid model census identity')
        by_path[row['path']] = {'bytes': row['bytes'], 'sha256': row['sha256']}
    for name, row in census.items():
        _require(by_path.get(name) == row, 'capture/metadata checkpoint mismatch: ' + name)
    shards = [{'path': name, 'bytes': row['bytes']} for name, row in sorted(by_path.items())
              if name.endswith(('.safetensors', '.gguf'))]
    _require(shards and all(row['path'] in census for row in shards),
             'capture lacks complete metadata-bound weight census')
    config = by_path.get('config.json')
    _require(config is not None and census.get('config.json') == config
             and config['sha256'] == model.get('config_sha256'), 'config identity differs')
    index = by_path.get('model.safetensors.index.json')
    _require((index or {}).get('sha256') == model.get('index_sha256')
             and (index or {}).get('bytes') == model.get('index_bytes'),
             'index identity differs')
    _require(model.get('weight_bytes') == sum(row['bytes'] for row in shards), 'weight byte census differs')
    if license_identity is not None:
        _require(license_identity['source_path'] == model.get('license_file')
                 and by_path.get(license_identity['source_path']) == {
                     'bytes': license_identity['bytes'], 'sha256': license_identity['sha256']},
                 'original model license differs from capture license')
    downloads = [{'path': name, 'bytes': row['bytes']} for name, row in sorted(by_path.items())]
    return {'config_sha256': config['sha256'], 'config_bytes': config['bytes'],
            'index_sha256': (index or {}).get('sha256'), 'index_bytes': (index or {}).get('bytes'),
            'index_source': 'model.safetensors.index.json' if index else (
                'gguf-files' if all(row['path'].endswith('.gguf') for row in shards) else 'single-safetensors'),
            'shards': shards, 'shard_manifest_sha256': _digest(shards),
            'model_bytes': model['weight_bytes'], 'download_manifest': downloads,
            'download_bytes_total': sum(row['bytes'] for row in downloads),
            'download_manifest_sha256': _digest(downloads)}


def qualify_result(result_dir, plan, execution_receipt, *, suite_root):
    """Return job_path, qualification_path, dataset_path after full qualification.

    suite_root selects trusted installed qualification code, not measured source
    identity. All measured code hashes come exclusively from the worker manifest.
    The returned files are private staging; public upload still uses publish gates.
    """
    import fidelity_dataset as fd
    _require(Path(suite_root).resolve() == Path(fd.REPO).resolve(),
             'suite_root differs from trusted qualification implementation')
    _plan_receipt(plan, execution_receipt)
    root = Path(result_dir).resolve()
    result, bundle, manifest_names = _result(root, plan)
    outputs = result.get('outputs') or {}
    _require(outputs.get('first') == 'first' and outputs.get('repeat') == 'repeat',
             'capture outputs must identify distinct canonical first/repeat directories')
    first, repeat = root / 'first', root / 'repeat'
    manifests, runtimes = [], []
    for label, path in (('canonical', first), ('repeat', repeat)):
        names = {str(Path(path.name) / rel) for rel in F.iter_dataset_files(str(path), exclude=())}
        _require(names <= manifest_names, 'dataset has files absent from worker manifest')
        worker_verify_name = path.name + '.verify.json'
        _require(worker_verify_name in manifest_names, 'worker omitted original full verification')
        worker_verify = _read(root / worker_verify_name)
        _require(common.verify_seal(worker_verify)
                 and worker_verify.get('schema') == F.VALIDATION_SCHEMA
                 and worker_verify.get('structural_status') == 'sealed'
                 and worker_verify.get('error_count') == 0 and worker_verify.get('errors') == []
                 and worker_verify.get('subject') == '/outputs/result/' + path.name,
                 'worker verification identity/status differs from capture output')
        manifest, runtime = fd._local_runtime_receipt(str(path), label)
        panel_input = plan['inputs']['panel']
        _require(panel_input.get('role') == 'final'
                 and all((manifest.get('panel') or {}).get(key) == (
                     None if panel_input.get('kind') == 'bundled' else panel_input.get(key))
                         for key in ('repository', 'revision')),
                 'capture panel differs from immutable requested final panel')
        _require((runtime.get('runtime_environment') or {}).get('cold_run')
                 == plan['workflow_id'] + '-' + path.name,
                 'capture cold-process label differs from workflow attempt')
        _require(not any(row.get('severity') == 'blocking' for row in manifest.get('disclosures', [])),
                 'blocking capture disclosure cannot be qualified')
        _require((manifest.get('determinism') or {}).get('cold_start_per_run') is True,
                 'capture lacks cold-start evidence')
        trusted = plan['runtime'].get('trusted_code')
        verified_code = (runtime.get('capture_tool') or {}).get('verified_code')
        _require((trusted is None and verified_code is None)
                 or (isinstance(trusted, dict) and isinstance(verified_code, dict)
                     and all(verified_code.get(key) == trusted.get(key)
                             for key in ('repository', 'revision'))),
                 'capture custom-code pin differs from vetted requested code')
        sources = runtime.get('source_files')
        hashes = {row['path']: row['sha256'] for row in bundle['files']}
        _require(isinstance(sources, dict) and sources
                 and all(hashes.get(name) == sha for name, sha in sources.items()),
                 'capture source hashes differ from measured worker source')
        manifests.append(manifest)
        runtimes.append(runtime)
    census = fd._local_checkpoint_census(runtimes[0], 'canonical')
    _require(census == fd._local_checkpoint_census(runtimes[1], 'repeat'),
             'cold captures have different checkpoint censuses')
    _require(runtimes[0].get('source_files') == runtimes[1].get('source_files'),
             'cold captures have different source identities')
    dataset, capture, runtime_meta = (manifests[0][key] for key in ('dataset', 'capture', 'runtime'))
    tool, fp = runtimes[0]['capture_tool'], runtimes[0]['stack_fingerprint']
    observed_license = tool.get('weights_license')
    license_identity = ({'source_path': plan['inputs']['model'].get('license_file'), 'dataset_path': observed_license['dataset_path'],
                         'bytes': observed_license['bytes'], 'sha256': observed_license['sha256']}
                        if isinstance(observed_license, dict) else None)
    target = _target(plan['inputs']['model'], census, license_identity)
    evidence = tool.get('resolved_panel_binding') or {}
    _require(isinstance(evidence.get('binding'), dict), 'capture lacks resolved panel binding')
    candidate = None
    surface, codec, bits = 'native-bf16', 'bf16', 16
    comparison_path = _inside(root, outputs.get('reproduction'))
    _require(outputs['reproduction'] in manifest_names, 'reproduction receipt absent from manifest')
    comparison = _read(comparison_path)
    if plan['mode'] == 'candidate':
        scope = _scope(plan)
        _require(scope == manifests[0].get('scope') and scope == manifests[1].get('scope'),
                 'candidate scope is not the actual canonical scope')
        scope_path = _inside(root, 'scope.json')
        _require('scope.json' in manifest_names and _read(scope_path) == plan['scope'],
                 'candidate scope file differs from plan')
        measurement = _read(_inside(root, outputs.get('comparison')))
        _require(outputs['comparison'] in manifest_names, 'comparison absent from worker manifest')
        report = dsvalidate.validate_receipt(measurement)
        _require(not report.errors, 'candidate comparison does not validate')
        _require((measurement.get('estimator') or {}).get('head_policy') == 'native_head'
                 and all(gate.get('passed') is True and gate.get('overridden_by') is None
                         for gate in measurement.get('gates', {}).values())
                 and (measurement.get('comparator') or {}).get('replay_backend') == 'numpy:cpu:float32'
                 and (measurement.get('comparator') or {}).get('vocab_chunk') == 8192,
                 'candidate comparison requires own heads and unoverridden scientific gates')
        ref = plan['inputs']['reference']
        side = measurement.get('reference') or {}
        _require(side.get('dataset_sha256') == ref.get('dataset_sha256'),
                 'comparison reference differs from immutable planned dataset')
        decode = tool.get('weights_decode') or {}
        method = decode.get('method')
        surfaces = {'fp8-block-dequant-to-bf16': 'fp8-block',
                    'exl3-trellis-decode-to-bf16': 'exl3hf',
                    'exl3-trellis-tp-compose-to-bf16': 'exl3hf',
                    'nvfp4-modelopt-dequant-to-bf16': 'nvfp4',
                    'gguf-dequant-to-bf16': 'gguf'}
        surfaces.update({'affine-weight-reconstruction': 'affine-reconstructed',
                         'microfloat-weight-reconstruction': 'microfloat-reconstructed'})
        if method in ('affine-weight-reconstruction', 'microfloat-weight-reconstruction'):
            _require(decode.get('scope') == 'weights_reconstructed'
                     and decode.get('comparison_class') == 'advisory'
                     and decode.get('modules_not_decoded') == []
                     and decode.get('components_not_consumed') == [],
                     'packed candidate must retain complete reconstruction and advisory evidence')
        _require(method in surfaces, 'unsupported candidate weight decode')
        surface, codec, bits = surfaces[method], plan.get('codec'), plan.get('declared_bits')
        candidate = {'scope': {'path': 'scope.json', 'sha256': common.sha256_file(str(scope_path)),
                               'scope_digest': scope['scope_digest']},
                     'codec': codec, 'declared_bits': bits,
                     'weights_decode': {'method': method, 'quantization_config': decode.get('quantization_config')},
                     'reference': {'repository': ref['repository'], 'revision': ref['revision'],
                                   'dataset_sha256': ref['dataset_sha256'],
                                   'capture_content_digest': side.get('capture_content_digest'),
                                   'dataset_id': side.get('dataset_id'),
                                   'panel_id': manifests[0]['panel']['panel_id'],
                                   'suite_token_hash_sha256': manifests[0]['panel']['suite_token_hash_sha256']}}
        _require(jobcontract.valid_candidate(candidate), 'candidate contract is incomplete')
    else:
        _require(plan.get('scope') is None and plan.get('codec') is None
                 and plan['inputs'].get('reference') is None,
                 'root plan must not claim candidate scope/reference')
    device, lane, form = fp['device'], runtime_meta['lane'], capture['form']
    registry = next((row for row in bundle['files'] if row['path'] == 'bin/BUNDLE.txt'), None)
    _require(registry is not None, 'measured source manifest omits bundle registry')
    control = dict(bundle, schema='fidelity-suite/control-plane-manifest.v1')
    profile = {'profile_id': 'root-hf-transformers-bf16', 'lane': 'root', 'source': 'native',
               'surface': surface, 'form': form, 'engine': 'hf-transformers',
               'compute_dtype': 'bfloat16', 'device': device, 'schedule': 'two-fresh-process-qualification'}
    replay = {'device': 'numpy', 'dtype': 'float32', 'vocab_chunk': 8192}
    repository = plan['output']['dataset_repository']
    hf_execution = {'plan': plan, 'provider_receipt': execution_receipt,
                    'worker_result_sha256': result['result_sha256'],
                    'worker_manifest_sha256': common.sha256_file(str(root / 'source-manifest.json')),
                    'worker_reported': {label: run['stack_fingerprint']
                                        for label, run in zip(('canonical', 'repeat'), runtimes)},
                    'provenance': _provenance(), 'publication_staging': 'private-recoverable'}
    doc = {'schema': 'fidelity-suite/job.v2', 'role': 'root', 'recipe': 'hf-jobs',
           'execution_attempt': {'number': 1, 'kind': 'hf-jobs', 'attempt_id': plan['workflow_id'][:24],
                                 'job_id': execution_receipt['job_id'], 'namespace': plan['owner']},
           'hf_execution': hf_execution, 'bundle': bundle, 'control_plane': control,
           'bundle_registry': registry, 'bundle_contract_sha256': _digest({'bundle': bundle, 'registry': registry}),
           'lane': lane, 'cold_runs': 2, 'reduce_order': 'fp32', 'profile': profile,
           'timing': {'kind': 'hf-jobs', 'requested_timeout_seconds': plan['hardware']['timeout_seconds'],
                      'provider_running_seconds': execution_receipt.get('running_seconds')},
           'target': dict(target, repo_id=plan['inputs']['model']['repository'],
                          revision=plan['inputs']['model']['revision'], path=None,
                          surface=surface, codec=codec, bits=bits, weights_license=license_identity),
           'panel': {'resolved_binding': evidence['binding'], 'binding_path': evidence['binding_file'],
                     'binding_file_sha256': evidence['binding_file_sha256']},
           'reference': {'reference_ref': None, 'teacher_receipt_sha256': None, 'teacher_backend_identity_sha256': None},
           'measurer': {'name': plan['owner'], 'handle': plan['owner'],
                        'url': 'https://huggingface.co/' + plan['owner'], 'is_artifact_author': False},
           'environment': {'gpu': fp.get('device_name') if device == 'cuda' else None,
                           'gpu_count': 1 if device == 'cuda' else 0,
                           'tensor_parallel': 1 if device == 'cuda' else None,
                           'host': None, 'execution_mode': 'hf-jobs', 'image': plan['image']},
           'runtime': {'device': device, 'reduce_order': 'fp32'}, 'keep_student_logits': False,
           'resource_requirements': {'workspace_available_bytes_minimum': None,
                                     'container_available_bytes_minimum': None,
                                     'min_vcpu_count': None, 'min_memory_gb': None,
                                     'expected_vram_bytes': 0 if device == 'cpu' else None},
           'disclosures': [], 'scope': plan.get('scope') or {'kind': 'root-capture', 'engine': 'hf-transformers',
                                                          'dtype': 'bfloat16', 'form': form},
           'produced_by': {'dependencies': {'profile': profile['profile_id'], 'lane': lane, 'provider': 'hf-jobs'},
                           'source_files': {row['path']: row['sha256'] for row in bundle['files']},
                           'capture_source_files': runtimes[0]['source_files']},
           'capture': {'role': 'root', 'form': form, 'replay': replay,
                       'root_protocol': {'schedule': 'two-fresh-process-qualification', 'fresh_processes': 2,
                                         'run_count_per_process': 1, 'exact_self_comparison': True,
                                         'qualification_required': True, 'canonical_publication_required': True,
                                         'publication_mode': 'canonical-public'},
                       'schedule': tool.get('schedule'), 'panel_id': evidence['binding']['panel']['id'],
                       'designated_reference': None, 'dataset_id': dataset.get('id'),
                       'dataset_repository': repository, 'dataset_name': dataset.get('name'),
                       'author': dataset.get('author', {}).get('name'), 'race': False, 'preview_of': None,
                       'publish_root_to': repository, 'dataset_license': dataset.get('license'),
                       'weights_license': license_identity, 'engine': 'hf-transformers', 'dtype': 'bfloat16',
                       'device': device, 'replay_device': 'numpy', 'replay_dtype': 'float32', 'vocab_chunk': 8192,
                       'own_heads': True, 'unexpected_tensor_allowlist': plan['runtime'].get('unexpected_allowlist'),
                       'resume_capture': None, 'candidate': candidate}}
    job = jobcontract.finalize_job(doc)
    if candidate is not None:
        reference_verify_path = _inside(root, 'reference.verify.json')
        _require('reference.verify.json' in manifest_names, 'candidate lacks original reference verification')
        reference_verify = _read(reference_verify_path)
        _require(common.verify_seal(reference_verify)
                 and reference_verify.get('schema') == F.VALIDATION_SCHEMA
                 and reference_verify.get('subject') == plan['inputs']['reference']['mount_path'],
                 'reference verification seal/subject differs')
        canonical = fd._capture_identity(str(first), runtimes[0]['runtime_environment']['cold_run'],
                                         'canonical', candidate=candidate)
        resultsink._validate_candidate_comparison(
            job, {'captures': {'canonical': canonical}}, measurement, reference_verify)
    job_path, qualification_path = root / 'job.json', root / 'qualification.json'
    _require(not job_path.exists() and not qualification_path.exists(), 'qualification is written once')
    # Fresh controller verification records the fetched paths, without rewriting
    # the original worker verification receipts or their measured identities.
    verify_paths = []
    for name, path in (('first', first), ('repeat', repeat)):
        report = dsvalidate.validate_dataset(str(path), verify_tensors=True)
        _require(not report.errors, name + ' fails full lossless tensor verification')
        verify_path = root / (name + '.controller-verify.json')
        common.write_json(str(verify_path), report.to_dict())
        verify_paths.append(str(verify_path))
    common.write_json(str(job_path), job)
    labels = [run['runtime_environment']['cold_run'] for run in runtimes]
    args = SimpleNamespace(first=str(first), repeat=str(repeat), first_label=labels[0], repeat_label=labels[1],
                           job=str(job_path), local=False, first_verify=verify_paths[0], repeat_verify=verify_paths[1],
                           comparison=str(comparison_path), imported_canonical=None, out=str(qualification_path))
    _require(fd.cmd_qualify_root(args) == fd.OK, 'HF Jobs scientific qualification refused')
    return {'job_path': str(job_path), 'qualification_path': str(qualification_path), 'dataset_path': str(first)}


def publication_source(dataset_path, qualification_path, job_path):
    """Bind public upload to actual bucket manifest and provider-bound worker result."""
    import fidelity_dataset as fd
    try:
        job = _read(job_path)
        validate_execution(job)
        root = Path(job_path).resolve().parent
        execution = job['hf_execution']
        result, _, names = _result(root, execution['plan'])
        _require(result['result_sha256'] == execution['worker_result_sha256']
                 and common.sha256_file(str(root / 'source-manifest.json')) == execution['worker_manifest_sha256'],
                 'publication worker result identity differs from qualification')
        _require(Path(dataset_path).resolve() == root / 'first', 'publication selects another dataset')
        _require(all('first/' + rel in names for rel in F.iter_dataset_files(dataset_path, exclude=())),
                 'publication dataset includes unmanifested bytes')
        source = fd._local_publish_source(dataset_path, qualification_path)
        source['source'] = 'hf-jobs-bucket-result'
        source['worker_result_sha256'] = result['result_sha256']
        source['provider_job_id'] = execution['provider_receipt']['job_id']
        return source
    except (HFQualificationError, jobcontract.JobContractError, KeyError, TypeError, OSError) as exc:
        raise fd.RootQualificationError('HF Jobs publication source invalid: ' + str(exc)) from exc
