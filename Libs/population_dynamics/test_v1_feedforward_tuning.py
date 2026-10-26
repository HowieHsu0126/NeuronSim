"""Exact source event projection and unfitted source-angle harmonics."""
import numpy as np
import pytest
from .v1_feedforward_tuning import project_emitted_events,harmonic_alignment,summarize,source_orientation_for_grating,emitted_event_modulation,project_complex_events


def test_source_synapse_multiplicity_weight_and_zero_targets():
    counts=np.array([[2,3],[0,1]])
    actual=project_emitted_events(counts,np.array([0,1,0]),np.array([1,1,2]),np.array([2,1,1]),np.array([.5,2.,0.]),4)
    np.testing.assert_array_equal(actual,[[0,8,0,0],[0,2,0,0]])


def test_preferred_direction_and_silent_denominator_retained():
    response=np.zeros((8,3));response[1,0]=4;response[3,1]=2
    phi=np.array([45.,135.,0.]);labels=np.array(['E','I','E'])
    rows,arrays=summarize(response,phi,labels)
    np.testing.assert_allclose(arrays['harmonic_2_signed_alignment'][:2],[1,1])
    assert np.isnan(arrays['harmonic_2_signed_alignment'][2])
    assert rows[0]['units']==3 and rows[0]['zero_denominator_units']==1
    assert rows[0]['harmonic_2_pooled_alignment']==pytest.approx(1.)


def test_source_separation_axis_matches_reflected_grating_mod_pi():
    # Connector phi=target+270; ON/OFF separation=(sin(-phi),cos(-phi)).
    source=np.deg2rad(np.array([0.,45.,90.,135.]))
    connector=source+1.5*np.pi
    separation=np.stack((np.sin(-connector),np.cos(-connector)),axis=1)
    np.testing.assert_allclose(separation,np.stack((np.cos(source),np.sin(source)),axis=1),atol=1e-14)
    # Source movie thetaRad=pi-theta. Its spatial normal is antiparallel
    # to the source separation axis when theta=-source angle, modulo pi.
    reflected_theta=-source
    grating=np.stack((np.cos(np.pi-reflected_theta),np.sin(np.pi-reflected_theta)),axis=1)
    np.testing.assert_allclose(np.sum(separation*grating,axis=1),-1.,atol=1e-14)
    raw_normal=np.stack((np.cos(np.pi-source),np.sin(np.pi-source)),axis=1)
    np.testing.assert_allclose(np.sum(separation*raw_normal,axis=1),-np.cos(2*source),atol=1e-14)


def test_invalid_negative_weight_rejects():
    with pytest.raises(ValueError,match='BOUND_EDGE'):
        project_emitted_events(np.ones((8,1),dtype=int),np.array([0]),np.array([0]),np.array([1]),np.array([-1.]),1)


def test_geometric_reflection_changes_orientation_without_fitted_offset():
    responses=np.zeros((8,2));responses[7,0]=3;responses[5,1]=2
    source_angle=np.array([45.,135.]);labels=np.array(['E','I'])
    raw,_=summarize(responses,source_angle,labels)
    reflected,_=summarize(responses,np.mod(-source_angle,360),labels)
    assert raw[0]['harmonic_2_pooled_alignment']==pytest.approx(-1.)
    assert reflected[0]['harmonic_2_pooled_alignment']==pytest.approx(1.)


def test_source_reflection_has_no_fitted_offset():
    np.testing.assert_array_equal(source_orientation_for_grating(np.array([0.,45.,135.,315.])),[0.,315.,225.,45.])


def test_phase_cancellation_occurs_after_source_edge_projection():
    values=emitted_event_modulation(np.array([1000.,1250.,3000.]),np.array([0,1,2]),3,
        frequency_hz=2.,start_ms=500.,stop_ms=3000.)
    np.testing.assert_allclose(values,[1.,-1.,1.],atol=1e-14)
    projected=project_complex_events(values[None,:],np.array([0,1]),np.array([0,0]),np.array([1,1]),np.array([2.,2.]),2)
    np.testing.assert_allclose(projected,0.,atol=1e-14)
    assert np.abs(values[:2]).sum()==pytest.approx(2.)


def test_modulation_window_excludes_start_and_preserves_endpoint_and_repeats():
    values=emitted_event_modulation(np.array([500.,1000.,1000.,3000.]),np.array([0,0,0,0]),1,
        frequency_hz=2.,start_ms=500.,stop_ms=3000.)
    np.testing.assert_allclose(values,[3.],atol=1e-14)


def test_complex_projection_rejects_nonfinite_source_values():
    with pytest.raises(ValueError,match='EDGE_AND_VALUE'):
        project_complex_events(np.array([[complex(np.nan,0)]]),np.array([0]),np.array([0]),np.array([1]),np.array([1.]),1)
