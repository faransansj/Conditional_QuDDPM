"""Execution-free Protocol v2.2 generation-seed contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from conditional_quddpm.experiments.tfim_confirmatory_protocol_v2 import canonical_json

PARENT_PROTOCOL_HASH = "3a1c242f43b7b8366fdcf4cfc37de51edecee4e4f3979f3dc751b21beec2e28d"
GENERATION_ROOT = 92001
GENERATION_DOMAINS = (
    "confirmatory.random.parameter_generation",
    "confirmatory.random.replacement_sampling",
    "confirmatory.blocked_g.parameter_generation",
    "confirmatory.blocked_g.replacement_sampling",
)
CALIBRATION_POOL_HASH = "02a4d4414437b310151f2e9f9cbabaf6161289715ee72a2d9e7f70062ec9c4c1"
V2_1_SPLIT_IMPLEMENTATION_HASH = "defeb2a57497025244175159539170dfb1ef6f986e473d95e4a264e7a4a3249b"
V2_1_CHECKSUM_MANIFEST_HASH = "18edb299e4f1532bc26f39cc219d09c46062507601e3f71cb9614c78f9fae647"
GENERATION_DETERMINISM_HASH = "997486d6567cb0aa199c25a7a418d49f2019eaca20089fabe75ae19b945a6305"
DATASET_CONTRACT_HASH = "05fe49a074512cd7dab8faf5484a70b18f583731bdd958d6b06b47363ccd9f47"
FRESH_CORPUS_CONTRACT_HASH = "610b6be087e17d3a4a68f51966ec6bae81de09a3eac1f8db81f8220aa71ff499"
PROVENANCE_CONTRACT_HASH = "60a95a4f9f3912a7e982027c47b4e58ea2059628868e084e951bd22d581738c3"
REPO_ROOT = Path(__file__).resolve().parents[3]
V2_1_ARTIFACT_HASHES = {
    "fs_calibration.json": "96d73a81e55e393833e9d26115f2f4c415bf4e797d5432131ad157b87fb505cb",
    "gate.json": "6df8e37db348521b5d39ca3de04852c2c08dc51a1b541f191e99434622460c6d",
    "protocol_manifest.json": PARENT_PROTOCOL_HASH,
    "split_audit.json": "d05794be6d824ae7a3d7844a416e228848e822a0924b672f4e7ea7b6c11fb5af",
    "split_manifest.json": "44dea1b9eea7383aa5ac457dcc9fb10d67a9f94e72934099cfde203b657fce2e",
}


def generation_seed(root_seed: int, domain: str) -> int:
    if domain not in GENERATION_DOMAINS:
        raise ValueError(f"unknown generation RNG domain: {domain}")
    digest = hashlib.sha256(f"tfim-confirmatory-v2|{int(root_seed)}|{domain}".encode()).digest()
    words = np.frombuffer(digest, dtype="<u4")
    return int(np.random.SeedSequence([int(root_seed), *map(int, words)]).generate_state(1, dtype=np.uint64)[0])


def generation_seed_manifest(root_seed: int = GENERATION_ROOT) -> dict:
    return {
        "schema_version": 1,
        "root_seed": int(root_seed),
        "derivation": "SeedSequence([root_seed, uint32_le(SHA256('tfim-confirmatory-v2|root_seed|domain'))])",
        "bit_generator": "PCG64DXSM",
        "domains": {domain: generation_seed(root_seed, domain) for domain in GENERATION_DOMAINS},
        "frozen_random_split_seed": 15007963261698017722,
    }


def generation_rng(seed: int, *, bit_generator: str = "PCG64DXSM") -> np.random.Generator:
    if bit_generator != "PCG64DXSM":
        raise ValueError("confirmatory generation RNG must use PCG64DXSM")
    return np.random.Generator(np.random.PCG64DXSM(int(seed)))


def validate_confirmatory_source(source_hash: str) -> None:
    if source_hash == CALIBRATION_POOL_HASH:
        raise ValueError("calibration pool is calibration-only and cannot be a confirmatory corpus")


def _json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_generation_contract(protocol: dict) -> None:
    if protocol.get("protocol_version") != "2.2.0" or protocol.get("parent_protocol_hash") != PARENT_PROTOCOL_HASH:
        raise ValueError("invalid Protocol v2.2 lineage")
    required = set(GENERATION_DOMAINS)
    contract = protocol.get("generation_seed_contract", {})
    seeds = contract.get("materialized_seeds", {})
    if contract.get("root_seed") != GENERATION_ROOT or set(seeds) != required or seeds != generation_seed_manifest()["domains"]:
        raise ValueError("generation seed contract is incomplete")
    if contract.get("bit_generator") != "PCG64DXSM" or contract.get("random_split_seed") != 15007963261698017722:
        raise ValueError("generation RNG contract is not frozen")
    if (_json_hash(protocol.get("generation_determinism")) != GENERATION_DETERMINISM_HASH
            or _json_hash(protocol.get("dataset_contract")) != DATASET_CONTRACT_HASH
            or _json_hash(protocol.get("fresh_corpus_contract")) != FRESH_CORPUS_CONTRACT_HASH
            or _json_hash(protocol.get("sample_provenance_required")) != PROVENANCE_CONTRACT_HASH):
        raise ValueError("generation determinism/provenance contract is incomplete")
    anchors = protocol.get("parent_artifact_anchors", {})
    expected_anchor_map = {name: digest for name, digest in V2_1_ARTIFACT_HASHES.items() if name != "protocol_manifest.json"}
    if (anchors.get("split_implementation_path") != "src/conditional_quddpm/experiments/tfim_confirmatory_protocol_v2_1.py"
            or anchors.get("split_implementation_sha256") != V2_1_SPLIT_IMPLEMENTATION_HASH
            or anchors.get("v2_1_checksums_sha256") != V2_1_CHECKSUM_MANIFEST_HASH
            or anchors.get("protocol_manifest_sha256") != PARENT_PROTOCOL_HASH
            or anchors.get("artifact_sha256") != expected_anchor_map):
        raise ValueError("Protocol v2.1 implementation/artifact anchors are incomplete")
    v21 = REPO_ROOT / "results/tfim_manifold_augmentation/confirmatory_protocol_v2_1"
    if (_file_hash(REPO_ROOT / anchors["split_implementation_path"]) != V2_1_SPLIT_IMPLEMENTATION_HASH
            or _file_hash(v21 / "checksums.sha256") != V2_1_CHECKSUM_MANIFEST_HASH
            or any(_file_hash(v21 / name) != digest for name, digest in V2_1_ARTIFACT_HASHES.items())):
        raise ValueError("Protocol v2.1 implementation/artifact integrity failure")
    if protocol["fresh_corpus_contract"].get("calibration_pool_usage") != "forbidden":
        raise ValueError("calibration pool reuse is not forbidden")


def freeze_protocol(protocol: dict, output: str | Path) -> str:
    validate_generation_contract(protocol)
    payload = canonical_json(protocol)
    Path(output).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
