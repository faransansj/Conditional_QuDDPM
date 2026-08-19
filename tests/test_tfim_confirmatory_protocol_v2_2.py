import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conditional_quddpm.experiments.tfim_confirmatory_protocol_v2 import named_seed
from conditional_quddpm.experiments.tfim_confirmatory_protocol_v2_2 import (
    CALIBRATION_POOL_HASH, GENERATION_DOMAINS, GENERATION_ROOT, V2_1_CHECKSUM_MANIFEST_HASH,
    V2_1_SPLIT_IMPLEMENTATION_HASH, freeze_protocol, generation_rng, generation_seed,
    generation_seed_manifest, validate_confirmatory_source, validate_generation_contract,
)

V21 = Path("results/tfim_manifold_augmentation/confirmatory_protocol_v2_1")
V22 = Path("results/tfim_manifold_augmentation/confirmatory_protocol_v2_2")


def test_generation_seed_manifest_is_deterministic_and_streams_are_independent():
    first = generation_seed_manifest(); second = generation_seed_manifest()
    assert first == second and first["root_seed"] == GENERATION_ROOT
    seeds = first["domains"]
    assert set(seeds) == set(GENERATION_DOMAINS) and len(set(seeds.values())) == 4
    assert seeds["confirmatory.random.parameter_generation"] != seeds["confirmatory.blocked_g.parameter_generation"]
    assert seeds["confirmatory.random.parameter_generation"] != seeds["confirmatory.random.replacement_sampling"]


def test_new_named_domains_do_not_alter_existing_frozen_split_seed():
    before = named_seed(92001, "dataset.split")
    _ = [generation_seed(GENERATION_ROOT, domain) for domain in reversed(GENERATION_DOMAINS)]
    assert before == named_seed(92001, "dataset.split") == 15007963261698017722


def test_replacement_streams_and_pcg64dxsm_are_deterministic():
    manifest = generation_seed_manifest()["domains"]
    for domain in ("confirmatory.random.replacement_sampling", "confirmatory.blocked_g.replacement_sampling"):
        a = generation_rng(manifest[domain]); b = generation_rng(manifest[domain])
        assert isinstance(a.bit_generator, np.random.PCG64DXSM)
        assert np.array_equal(a.random(8), b.random(8))
    with pytest.raises(ValueError):
        generation_rng(1, bit_generator="PCG64")


def test_calibration_pool_cannot_be_confirmatory_corpus():
    with pytest.raises(ValueError, match="calibration-only"):
        validate_confirmatory_source(CALIBRATION_POOL_HASH)
    validate_confirmatory_source("fresh-artifact-hash")


def test_protocol_v21_immutable_and_v22_hash_reproducible(tmp_path):
    entries = [line.split("  ", 1) for line in (V21 / "checksums.sha256").read_text().splitlines()]
    assert all(hashlib.sha256((V21 / name).read_bytes()).hexdigest() == digest for digest, name in entries)
    assert hashlib.sha256((V21 / "checksums.sha256").read_bytes()).hexdigest() == V2_1_CHECKSUM_MANIFEST_HASH
    assert hashlib.sha256(Path("src/conditional_quddpm/experiments/tfim_confirmatory_protocol_v2_1.py").read_bytes()).hexdigest() == V2_1_SPLIT_IMPLEMENTATION_HASH
    protocol = json.loads((V22 / "protocol_manifest.json").read_text())
    digest = freeze_protocol(protocol, tmp_path / "protocol.json")
    gate = json.loads((V22 / "gate.json").read_text())
    assert digest == gate["protocol_hash"] == hashlib.sha256((V22 / "protocol_manifest.json").read_bytes()).hexdigest()
    assert protocol["generation_determinism"]["unspecified_randomness_remaining"] == []
    assert not gate["dataset_generation_performed"] and gate["qcnn_run_count"] == 0
    broken = json.loads(json.dumps(protocol)); broken["generation_determinism"].pop("stream_initialization")
    with pytest.raises(ValueError, match="determinism/provenance"):
        validate_generation_contract(broken)
