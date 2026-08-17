"""Global and reduced-density-matrix kernels for four-qubit state ensembles."""
from __future__ import annotations
from itertools import combinations
import numpy as np


def _normalized(state: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    state=np.asarray(state,dtype=np.complex128); norm=float(np.vdot(state,state).real)
    if state.ndim != 1 or state.size < 2 or state.size & (state.size-1): raise ValueError("state must be a power-of-two vector")
    if abs(norm-1.0)>tolerance: raise ValueError("state must be normalized")
    return state


def reduced_density_matrix(state: np.ndarray, kept_qubits: tuple[int,...]) -> np.ndarray:
    state=_normalized(state); n=int(np.log2(state.size)); kept=tuple(kept_qubits)
    if not kept or len(set(kept))!=len(kept) or any(q<0 or q>=n for q in kept): raise ValueError("invalid kept qubits")
    traced=tuple(q for q in range(n) if q not in kept); matrix=np.transpose(state.reshape((2,)*n),kept+traced).reshape(2**len(kept),-1); rho=matrix@matrix.conj().T
    return (rho+rho.conj().T)/2


def density_fidelity(left: np.ndarray, right: np.ndarray) -> float:
    left=np.asarray(left,dtype=np.complex128); right=np.asarray(right,dtype=np.complex128)
    values,vectors=np.linalg.eigh((left+left.conj().T)/2); sqrt_left=(vectors*np.sqrt(np.clip(values,0,None)))@vectors.conj().T
    middle=sqrt_left@right@sqrt_left; fidelity=float(np.square(np.sqrt(np.clip(np.linalg.eigvalsh((middle+middle.conj().T)/2),0,None)).sum()).real)
    return float(np.clip(fidelity,0.0,1.0))


def global_fidelity_kernel(left: np.ndarray, right: np.ndarray) -> float:
    left,right=_normalized(left),_normalized(right)
    return float(abs(np.vdot(left,right))**2)


def one_rdm_kernel(left: np.ndarray, right: np.ndarray) -> float:
    n=int(np.log2(_normalized(left).size)); _normalized(right)
    return float(np.mean([density_fidelity(reduced_density_matrix(left,(q,)),reduced_density_matrix(right,(q,))) for q in range(n)]))


def two_rdm_kernel(left: np.ndarray, right: np.ndarray) -> float:
    n=int(np.log2(_normalized(left).size)); _normalized(right); pairs=list(combinations(range(n),2))
    return float(np.mean([density_fidelity(reduced_density_matrix(left,pair),reduced_density_matrix(right,pair)) for pair in pairs]))


KERNELS={"global":global_fidelity_kernel,"1-rdm":one_rdm_kernel,"2-rdm":two_rdm_kernel}


def kernel_matrix(left: np.ndarray, right: np.ndarray, kernel: str) -> np.ndarray:
    function=KERNELS[kernel]
    return np.asarray([[function(a,b) for b in right] for a in left],dtype=float)


def kernel_mmd(left: np.ndarray, right: np.ndarray, kernel: str) -> float:
    value=kernel_matrix(left,left,kernel).mean()+kernel_matrix(right,right,kernel).mean()-2*kernel_matrix(left,right,kernel).mean()
    return float(max(value,0.0))


def class_weighted_kernel_mmd(generated: dict[int,np.ndarray],target: dict[int,np.ndarray],kernel: str) -> float:
    if set(generated)!=set(target): raise ValueError("generated and target classes must match")
    return float(np.mean([kernel_mmd(generated[label],target[label],kernel) for label in sorted(target)]))
