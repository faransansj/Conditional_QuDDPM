# TFIM Simulation 실행 및 데이터 가이드

## 1. 실행 환경: `uv`를 사용해도 되는가?

가능하다. 이 프로젝트는 표준 `pyproject.toml`을 사용하므로 `venv + pip` 대신 `uv`를 사용해도 시뮬레이션 결과는 바뀌지 않는다. `uv`는 가상환경 생성, 의존성 설치, lockfile 관리를 한 명령으로 처리하므로 이 저장소의 기본 실행 방법으로 사용한다.

필요 조건:

- Python 3.11 이상
- [`uv`](https://docs.astral.sh/uv/) 설치

```bash
uv --version
uv sync --extra dev
```

`uv sync`는 프로젝트의 `.venv/`를 생성하거나 갱신하고 `uv.lock`에 기록된 버전을 설치한다. 별도로 `source .venv/bin/activate`를 실행할 필요는 없다.

## 2. TFIM 시뮬레이션 실행

저장소 루트에서 다음을 실행한다.

Random-split benchmark:

```bash
uv run generate-tfim \
  --config configs/dataset/tfim_4q.yaml \
  --output data/tfim_4q_random
```

Blocked-g benchmark:

```bash
uv run generate-tfim \
  --config configs/dataset/tfim_4q_blocked.yaml \
  --output data/tfim_4q_blocked
```

Blocked 방식은 class별 `g` 영역을 train/val/test 연속 구간으로 나누고 인접 구간 사이에 `blocked_g_gap`을 비워 near-duplicate state를 방지한다. Random 방식은 기존 class-stratified random assignment를 유지하며 두 결과를 별도 benchmark로 보고한다.

동일한 명령을 source-tree script로 실행할 수도 있다.

```bash
uv run python scripts/generate_tfim.py \
  --config configs/dataset/tfim_4q.yaml \
  --output data/tfim_4q
```

실행이 끝나면 validation 결과가 출력된다. 현재 기본 config의 정상 출력 예시는 다음과 같다.

```json
{
  "valid": true,
  "errors": [],
  "samples": 400,
  "split_strategy": "random",
  "split_counts": {
    "train": 280,
    "val": 60,
    "test": 60
  },
  "max_norm_error": 8.881784197001252e-16,
  "max_eigenpair_residual": 4.839349969133126e-15,
  "minimum_cross_split_g_gap": 1.8443025010306258e-05,
  "checksums_valid": {
    "states.npz": true,
    "split_manifest.json": true,
    "validation.json": true
  }
}
```

`valid: false`이면 CLI가 non-zero exit code로 종료한다. 실패한 데이터를 후속 QCNN/QuDDPM 입력으로 사용하지 않는다.

## 3. 무엇을 시뮬레이션하는가?

Hamiltonian convention은 다음과 같다.

```text
H = -J Σ_i Z_i Z_(i+1) - g Σ_i X_i
```

기본 설정:

| 설정 | 값 | 의미 |
|---|---:|---|
| `n_qubits` | 4 | Hilbert-space dimension은 `2^4 = 16` |
| `J` | 1.0 | nearest-neighbor Ising coupling |
| `boundary` | open | 마지막 qubit과 첫 qubit을 연결하지 않음 |
| ferromagnetic `g/J` | 0.2–0.8 | class 0 sampling 영역 |
| paramagnetic `g/J` | 1.2–1.8 | class 1 sampling 영역 |
| samples/class | 200 | 총 400 parameter points, train 140/class |
| split | 70/15/15% | class-stratified random 또는 blocked-g |

각 `g`에서 dense Hermitian exact diagonalization을 수행하여 최소 eigenvalue `E0`와 정규화된 ground-state vector `|ψ0⟩`를 계산한다.

`g/J = 1`은 thermodynamic-limit critical point이지만 4-qubit finite chain에는 sharp transition이 없다. 따라서 class label은 물리적 상의 엄밀한 판정이 아니라 사전에 고정한 benchmark 영역이다. 초기 데이터에서는 모호한 `0.8 < g/J < 1.2` 구간을 제외한다.

## 4. 생성되는 파일

```text
data/tfim_4q/
├── states.npz
├── split_manifest.json
├── checksums.json
└── validation.json
```

`data/`는 재생성 가능한 결과물이므로 Git에 commit하지 않는다.

### `states.npz`

NumPy compressed archive이며 기본 실행 결과는 다음 배열을 포함한다.

| Key | Shape | dtype | 설명 |
|---|---:|---|---|
| `states` | `(400, 16)` | `complex128` | computational basis의 ground-state amplitudes |
| `energies` | `(400,)` | `float64` | 각 Hamiltonian의 minimum eigenvalue |
| `labels` | `(400,)` | `int8` | `0`: ferromagnetic, `1`: paramagnetic |
| `parameter_ids` | `(400,)` | string | leakage 검사용 고유 parameter ID |
| `splits` | `(400,)` | string | `train`, `val`, `test` |
| `g` | `(400,)` | `float64` | transverse-field strength |
| `magnetization_x` | `(400,)` | `float64` | transverse magnetization `⟨Mx⟩` |
| `magnetization_z2` | `(400,)` | `float64` | longitudinal order diagnostic `⟨Mz²⟩` |

`states[i]`, `energies[i]`, `labels[i]`, `parameter_ids[i]`, `splits[i]`, `g[i]`는 모두 같은 sample을 나타낸다.

### `split_manifest.json`

실행에 사용한 resolved config와 sample별 metadata를 사람이 읽을 수 있는 JSON으로 저장한다. split은 diagonalization 전에 parameter ID 수준에서 지정된다. 향후 QuDDPM/MSQuDDPM은 `split == "train"`인 ID만 읽어야 한다.

### `checksums.json`

`states.npz`, `split_manifest.json`, `validation.json` 각각의 SHA-256 checksum을 기록한다. CLI가 생성 직후 세 checksum을 모두 검증한다.

### `validation.json`

다음을 검사한 machine-readable report다.

- train/val/test parameter ID가 서로 겹치지 않는가;
- 모든 state가 정규화되어 있는가;
- `||H|ψ⟩ - E|ψ⟩||` residual이 tolerance 이내인가;
- ferromagnetic class에서 평균 `⟨Mz²⟩`가 더 크고 paramagnetic class에서 평균 `⟨Mx⟩`가 더 큰가;
- blocked split의 실제 minimum cross-split `g` gap이 config 기준 이상인가.

## 5. 데이터 읽기

```bash
uv run python - <<'PY'
import numpy as np

data = np.load("data/tfim_4q/states.npz")

train = data["splits"] == "train"
train_states = data["states"][train]
train_labels = data["labels"][train]

print(train_states.shape)  # (280, 16)
print(train_labels.shape)  # (280,)
print(np.unique(train_labels, return_counts=True))
PY
```

한 sample 확인:

```python
i = 0
psi = data["states"][i]
print("id:", data["parameter_ids"][i])
print("split:", data["splits"][i])
print("g:", data["g"][i])
print("label:", data["labels"][i])
print("ground energy:", data["energies"][i])
print("norm:", np.vdot(psi, psi).real)
```

`states`는 density matrix가 아니라 pure-state vector다. density matrix가 필요한 단계에서는 명시적으로 `rho = np.outer(psi, psi.conj())`를 사용한다. 4-qubit 기준 `psi.shape == (16,)`, `rho.shape == (16, 16)`이다.

## 6. 설정 변경

`configs/dataset/tfim_4q.yaml`을 복사하여 별도 config를 만든다.

```bash
cp configs/dataset/tfim_4q.yaml configs/dataset/tfim_6q.yaml
```

예를 들어 6-qubit smoke run은 `n_qubits: 6`과 작은 `samples_per_class`로 먼저 검증한다. 기존 benchmark config를 덮어쓰지 않는다.

중요 설정:

- `dataset_seed`: class별 `g` sampling을 결정한다.
- `split_seed`: random 방식에서 동일한 sampled parameters의 split 배치를 결정한다.
- `split_strategy`: `random` 또는 `blocked`.
- `blocked_g_gap`: blocked 방식에서 인접 split 구간 사이에 제외할 최소 `g` 폭.
- `phase_regions`: 두 구간은 순서대로 배치되고 겹치지 않아야 한다.
- `split_ratios`: 음수가 아니며 합이 1이어야 한다.
- `numerical_tolerance`: validation 기준이며 결과를 보고 사후 조정하지 않는다.

동일 config와 코드에서는 manifest와 state가 재현되어야 한다.

## 7. 테스트

```bash
uv run pytest
```

현재 테스트는 다음을 확인한다.

1. 2-qubit Hamiltonian이 Pauli-term 정의와 일치함;
2. Hamiltonian이 Hermitian임;
3. ground state가 정규화된 최소 eigenpair임;
4. 잘못된 phase-region overlap이 거부됨;
5. dataset이 seed에 대해 재현되고 split leakage가 없음;
6. blocked split이 configured `g` gap을 지킴;
7. `⟨Mx⟩`, `⟨Mz²⟩`가 class를 기대 방향으로 분리함;
8. 세 artifact checksum이 모두 검증되고 변조가 탐지됨.

## 8. 결과 해석 시 주의사항

- `label`은 sampled `g/J` 영역에서 부여된 supervised target이며 ground state에서 추론한 label이 아니다.
- test split은 QCNN 최종 평가 전까지 generator training, early stopping, hyperparameter 선택에 사용하면 안 된다.
- 서로 가까운 연속 `g` 값의 state는 강하게 유사할 수 있다. 향후 split 전략의 민감도 분석이 필요하다.
- dense exact diagonalization과 mixed-state 변환은 qubit 수에 따라 지수적으로 증가한다. 6/8-qubit은 작은 smoke run과 profiling 후 확장한다.
- 현재 출력은 noiseless pure states다. depolarizing/dephasing/amplitude-damping 데이터는 아직 구현되지 않았다.
