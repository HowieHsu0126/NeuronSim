from copy import deepcopy
import numpy as np
import pytest

from .v1_glif_component_reference import SIGNALS, build_cases, recording_matrices


def reference(identity):
    return dict(neuronal_model_id=identity, converted_template="nest:glif_psc",
        converted_parameters=dict(after_spike_currents=True, adapting_threshold=False,
            spike_dependent_threshold=False, asc_init=[0., 0.], asc_amps=[-2., 1.],
            asc_decay=[.1, .3], tau_syn=[2., 5.], C_m=100., g=5., V_th=-50., E_L=-70.))


def test_cases_keep_every_receptor_sign_and_source_model():
    bank=[reference(101), reference(102)];before=deepcopy(bank)
    cases=build_cases(bank)
    assert bank==before and len(cases)==12 and len({x["case_id"] for x in cases})==12
    for index in range(2):
        signed=[x for x in cases if x["source_index"]==index and x["input_kind"]=="sparse"]
        assert {(x["receptor_type"],np.sign(x["weight_pA"])) for x in signed}=={(1,1),(1,-1),(2,1),(2,-1)}


def test_lif_substitution_is_rejected():
    bank=[reference(101)];bank[0]["converted_parameters"]["after_spike_currents"]=False
    with pytest.raises(ValueError,match="SOURCE_GLIF3_REQUIRED"):build_cases(bank)


def events():
    return dict(senders=np.array([3,10,3,10]), times=np.array([.5,.25,.25,.5]),
                **{key:np.array([-42.,-70.,-40.,-72.]) for key in SIGNALS})


def test_recording_retains_declared_case_order():
    result=recording_matrices(events(),np.array([10,3]),np.array([.25,.5]))
    assert np.array_equal(result["V_m"],np.array([[-70.,-72.],[-40.,-42.]]))


def test_duplicate_time_pair_cannot_replace_a_missing_sample():
    value=events();value["times"][3]=.25
    with pytest.raises(ValueError,match="COORDINATES_DIFFER"):
        recording_matrices(value,np.array([10,3]),np.array([.25,.5]))


def test_reference_precision_is_not_silently_promoted():
    value=events();value["V_m"]=value["V_m"].astype(np.float32)
    with pytest.raises(ValueError,match="PRECISION_OR_SHAPE"):
        recording_matrices(value,np.array([10,3]),np.array([.25,.5]))
