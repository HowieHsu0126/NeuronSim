"""Compare source-emitted LGN afferent tuning and complete cortical responses.

The afferent proxy sums emitted event counts times source synapse multiplicity
and source table weight. It is not a membrane current or effective arrival-time
trace. All target units, zero-afferent units and undefined harmonic denominators
remain visible. No angles, subsets or success thresholds are fitted.
"""
from __future__ import annotations

import numpy as np


def project_emitted_events(source_counts: np.ndarray, source_ids: np.ndarray, target_ids: np.ndarray,
                           multiplicities: np.ndarray, weights: np.ndarray, target_units: int) -> np.ndarray:
    counts, sources, targets, nsyns, weight = map(np.asarray,(source_counts,source_ids,target_ids,multiplicities,weights))
    if (counts.ndim != 2 or counts.dtype.kind not in "iu" or np.any(counts < 0)
            or sources.ndim != 1 or sources.dtype.kind not in "iu" or targets.shape != sources.shape or targets.dtype.kind not in "iu"
            or nsyns.shape != sources.shape or nsyns.dtype.kind not in "iu" or np.any(nsyns < 1)
            or weight.shape != sources.shape or weight.dtype.kind != "f" or not np.isfinite(weight).all() or np.any(weight < 0)
            or type(target_units) is not int or target_units < 1 or np.any(sources < 0) or np.any(sources >= counts.shape[1])
            or np.any(targets < 0) or np.any(targets >= target_units)):
        raise ValueError("V1_AFFERENT_BOUND_EDGE_AND_COUNT_ARRAYS_REQUIRED")
    result=np.zeros((counts.shape[0],target_units),dtype=float)
    scale=nsyns.astype(float)*weight
    for condition in range(counts.shape[0]):
        result[condition]=np.bincount(targets,weights=counts[condition,sources]*scale,minlength=target_units)
    return result


def harmonic_alignment(responses: np.ndarray, tuning_angle_deg: np.ndarray) -> dict:
    responses,phi=map(np.asarray,(responses,tuning_angle_deg))
    if (responses.ndim != 2 or responses.shape[0] != 8 or phi.shape != (responses.shape[1],)
            or not np.isfinite(responses).all() or np.any(responses < 0)
            or not np.isfinite(phi).all() or np.any(phi < 0) or np.any(phi >= 360)):
        raise ValueError("V1_AFFERENT_COMPLETE_DIRECTION_RESPONSE_REQUIRED")
    angles=np.deg2rad(np.arange(0,360,45));phi=np.deg2rad(phi)
    denominator=responses.sum(axis=0);defined=denominator>0
    result=dict(denominator=denominator,defined=defined)
    for harmonic in (1,2):
        z=np.exp(1j*harmonic*angles)@responses
        aligned=z*np.exp(-1j*harmonic*phi)
        result[f"harmonic_{harmonic}_signed_alignment"]=np.divide(aligned.real,denominator,out=np.full(len(phi),np.nan),where=defined)
        result[f"harmonic_{harmonic}_quadrature"]=np.divide(aligned.imag,denominator,out=np.full(len(phi),np.nan),where=defined)
    return result


def summarize(responses: np.ndarray, tuning_angle_deg: np.ndarray, populations: np.ndarray) -> tuple[list[dict],dict]:
    arrays=harmonic_alignment(responses,tuning_angle_deg);labels=np.asarray(populations)
    if labels.shape != tuning_angle_deg.shape or labels.dtype.kind not in "US":
        raise ValueError("V1_AFFERENT_SOURCE_POPULATION_MAP_REQUIRED")
    rows=[];angles=np.deg2rad(np.arange(0,360,45));phi=np.deg2rad(tuning_angle_deg)
    for name in ("all",*tuple(dict.fromkeys(labels.tolist()))):
        selected=np.ones(len(labels),dtype=bool) if name=="all" else labels==name
        total=float(arrays['denominator'][selected].sum())
        row=dict(population=name,units=int(selected.sum()),zero_denominator_units=int(np.count_nonzero(selected&~arrays['defined'])))
        for harmonic in (1,2):
            real=sum((responses[index,selected]*np.cos(harmonic*(angles[index]-phi[selected]))).sum() for index in range(8))
            imag=sum((responses[index,selected]*np.sin(harmonic*(angles[index]-phi[selected]))).sum() for index in range(8))
            row[f"harmonic_{harmonic}_pooled_alignment"]=float(real/total) if total>0 else None
            row[f"harmonic_{harmonic}_pooled_quadrature"]=float(imag/total) if total>0 else None
        rows.append(row)
    return rows,arrays


def source_orientation_for_grating(tuning_angle_deg: np.ndarray) -> np.ndarray:
    """Source ON/OFF separation axis versus GratingMovie thetaRad=pi-theta.

    The returned angle is an orientation convention. ON/OFF order and temporal
    filter differences do not identify the sign of preferred drift direction.
    """
    tuning=np.asarray(tuning_angle_deg)
    if tuning.ndim != 1 or not np.isfinite(tuning).all() or np.any(tuning < 0) or np.any(tuning >= 360):
        raise ValueError("V1_SOURCE_ORIENTATION_ANGLE_DOMAIN_REQUIRED")
    return np.mod(-tuning,360.)


def emitted_event_modulation(event_times_ms: np.ndarray, event_sources: np.ndarray, source_units: int,
                             *, frequency_hz: float, start_ms: float, stop_ms: float) -> np.ndarray:
    """Complex emitted-event Fourier sum on (start_ms,stop_ms].

    Keep phase through projection. Summing magnitudes before projection loses
    the source ON/OFF cancellation that this diagnostic tests.
    """
    times,sources=map(np.asarray,(event_times_ms,event_sources))
    if (times.ndim != 1 or times.dtype.kind != "f" or not np.isfinite(times).all()
            or sources.shape != times.shape or sources.dtype.kind not in "iu"
            or type(source_units) is not int or source_units < 1 or np.any(sources < 0) or np.any(sources >= source_units)
            or not np.isfinite(frequency_hz) or frequency_hz <= 0 or not 0 <= start_ms < stop_ms <= 3000):
        raise ValueError("V1_SOURCE_MODULATION_EVENT_DOMAIN_REQUIRED")
    selected=(times > start_ms)&(times <= stop_ms)
    phase=-2j*np.pi*frequency_hz*(times[selected]-start_ms)/1000.
    values=np.exp(phase)
    return np.bincount(sources[selected],weights=values.real,minlength=source_units)+1j*np.bincount(sources[selected],weights=values.imag,minlength=source_units)


def project_complex_events(source_values: np.ndarray, source_ids: np.ndarray, target_ids: np.ndarray,
                           multiplicities: np.ndarray, weights: np.ndarray, target_units: int) -> np.ndarray:
    """Apply the source edge-weight map before taking modulation amplitude."""
    values,sources,targets,nsyns,weight=map(np.asarray,(source_values,source_ids,target_ids,multiplicities,weights))
    if (values.ndim != 2 or values.dtype.kind != "c" or not np.isfinite(values).all()
            or sources.ndim != 1 or sources.dtype.kind not in "iu" or targets.shape != sources.shape or targets.dtype.kind not in "iu"
            or nsyns.shape != sources.shape or nsyns.dtype.kind not in "iu" or np.any(nsyns < 1)
            or weight.shape != sources.shape or weight.dtype.kind != "f" or not np.isfinite(weight).all() or np.any(weight < 0)
            or type(target_units) is not int or target_units < 1 or np.any(sources < 0) or np.any(sources >= values.shape[1])
            or np.any(targets < 0) or np.any(targets >= target_units)):
        raise ValueError("V1_SOURCE_MODULATION_EDGE_AND_VALUE_ARRAYS_REQUIRED")
    result=np.zeros((values.shape[0],target_units),dtype=complex);scale=nsyns.astype(float)*weight
    for condition in range(values.shape[0]):
        weighted=values[condition,sources]*scale
        result[condition]=np.bincount(targets,weights=weighted.real,minlength=target_units)+1j*np.bincount(targets,weights=weighted.imag,minlength=target_units)
    return result
