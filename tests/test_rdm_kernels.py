import numpy as np
import pytest
from conditional_quddpm.models.quddpm import fidelity_matrix,fidelity_mmd,haar_states
from conditional_quddpm.models.rdm_kernels import (class_weighted_kernel_mmd,density_fidelity,global_fidelity_kernel,kernel_matrix,kernel_mmd,kernel_mmd_raw,one_rdm_kernel,reduced_density_matrix,two_rdm_kernel)


def test_global_kernel_matches_existing_fidelity_implementation():
    states=haar_states(5,90,4)
    assert np.allclose(kernel_matrix(states,states,"global"),fidelity_matrix(states,states))


def test_global_kernel_mmd_matches_existing_fidelity_mmd_implementation():
    for seed in (96,97,98):
        left=haar_states(4,seed,4); right=haar_states(5,seed+100,4)
        assert np.isclose(kernel_mmd(left,right,"global"),fidelity_mmd(left,right))
        raw=fidelity_matrix(left,left).mean()+fidelity_matrix(right,right).mean()-2*fidelity_matrix(left,right).mean()
        assert np.isclose(kernel_mmd_raw(left,right,"global"),raw)
        assert np.isclose(kernel_mmd(left,right,"global"),max(raw,0.0))


def test_rdm_kernels_are_symmetric_deterministic_and_maximal_for_identical_states():
    left,right=haar_states(2,91,4)
    for kernel in (global_fidelity_kernel,one_rdm_kernel,two_rdm_kernel):
        assert np.isclose(kernel(left,left),1.0)
        assert np.isclose(kernel(left,right),kernel(right,left))
        assert kernel(left,right)==kernel(left,right)


def test_partial_trace_of_bell_state_is_correct_and_physical():
    bell=np.asarray([1,0,0,1],dtype=np.complex128)/np.sqrt(2)
    single=reduced_density_matrix(bell,(0,)); pair=reduced_density_matrix(bell,(0,1))
    assert np.allclose(single,np.eye(2)/2)
    assert np.allclose(pair,np.outer(bell,bell.conj()))
    for rho in (single,pair):
        assert np.allclose(rho,rho.conj().T)
        assert np.isclose(np.trace(rho),1)
        assert np.linalg.eigvalsh(rho).min()>=-1e-12
        assert np.isclose(density_fidelity(rho,rho),1)


def test_rdm_kernels_equal_manual_equal_weighting():
    left,right=haar_states(2,92,4)
    singles=[density_fidelity(reduced_density_matrix(left,(q,)),reduced_density_matrix(right,(q,))) for q in range(4)]
    pairs=[density_fidelity(reduced_density_matrix(left,(i,j)),reduced_density_matrix(right,(i,j))) for i in range(4) for j in range(i+1,4)]
    assert np.isclose(one_rdm_kernel(left,right),np.mean(singles))
    assert np.isclose(two_rdm_kernel(left,right),np.mean(pairs))


def test_mmd_is_order_invariant_and_classes_are_not_mixed():
    class0=haar_states(4,93,4); class1=haar_states(4,94,4); generated={0:class0,1:class1}; target={0:class1,1:class0}
    for kernel in ("global","1-rdm","2-rdm"):
        assert np.isclose(kernel_mmd(class0,class1,kernel),kernel_mmd(class0[[2,0,3,1]],class1[[1,3,0,2]],kernel))
        expected=np.mean([kernel_mmd(generated[c],target[c],kernel) for c in (0,1)])
        assert np.isclose(class_weighted_kernel_mmd(generated,target,kernel),expected)


def test_unnormalized_pure_states_are_rejected():
    with pytest.raises(ValueError,match="normalized"):
        one_rdm_kernel(np.ones(16,dtype=complex),haar_states(1,95,4)[0])
