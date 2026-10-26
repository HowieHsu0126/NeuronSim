"""Admission checks against a restored, pinned official source on Merlin CPU."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path

import pytest

from population_dynamics import appworld_benchmark_development as development
from population_dynamics.appworld_benchmark_runner import _AGENT_CLASSES, _load_agent_class


@pytest.fixture
def pinned_source():
    source = os.environ.get("APPWORLD_PINNED_SOURCE_ROOT")
    manifest_path = os.environ.get("APPWORLD_PINNED_SOURCE_MANIFEST")
    if source is None or manifest_path is None:
        pytest.skip("requires the restored pinned source and its original bound manifest")
    path = Path(manifest_path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == os.environ["APPWORLD_PINNED_SOURCE_MANIFEST_SHA256"]
    config = deepcopy(development.load_benchmark_config(development._DEFAULT_CONFIG))
    config["source"]["archive_manifest"] = {"path": str(path), "sha256": digest}
    return config, Path(source), json.loads(payload)


@pytest.mark.parametrize("method", tuple(_AGENT_CLASSES))
def test_pinned_official_prompt_and_constructor_admission(pinned_source, method, record_property):
    config, source, manifest = pinned_source
    prepared = development.prepare_official_agent(config, method, source)
    provenance = prepared.source_provenance
    assert provenance["admission_method"] == "source_archive_manifest"
    assert provenance["revision"] == manifest["revision"]
    assert provenance["prompts"]
    assert all(row["matches_admitted_archive_bytes"] for row in provenance["prompts"])
    native_class = _load_agent_class(method)
    module = _AGENT_CLASSES[method][0]
    relative = f"experiments/code/simplified/{module}.py"
    actual = Path(inspect.getsourcefile(native_class)).read_bytes()
    expected = manifest["files"][relative]
    assert len(actual) == expected["bytes"]
    assert hashlib.sha256(actual).hexdigest() == expected["sha256"]
    signature = inspect.signature(native_class.__init__)
    signature.bind_partial(None, **prepared.agent_config)
    record_property("source_revision", provenance["revision"])
    record_property("prompts", json.dumps(provenance["prompts"], sort_keys=True))
    record_property("constructor_signature", str(signature))
    if method == "official_function_calling":
        assert prepared.agent_config["prompt_file_path"].endswith(
            "/function_calling_agent/instructions.txt"
        )
        assert prepared.agent_config["demo_messages_file_path"] is None
        record_property("constructor_source", inspect.getsource(native_class.__init__))


def test_absent_official_prompt_rejected_by_pinned_manifest(pinned_source):
    config, source, _ = pinned_source
    config["development_execution"]["methods"]["official_function_calling"]["prompt_file_path"] = (
        "experiments/prompts/function_calling_agent/zero_shot_instructions.txt"
    )
    with pytest.raises(ValueError, match="official prompt is not listed in the bound source archive"):
        development.prepare_official_agent(config, "official_function_calling", source)
