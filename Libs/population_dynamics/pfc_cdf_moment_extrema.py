"""Sharp finite-support moment screen using saved empirical moments only."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog
import scipy
import yaml

from .pfc_cdf_moment_envelope import bound_inputs, read_csv, write_csv, moment_interval
from .pfc_confirmation_count_scores import require
from .provenance import freeze_json, sha256_file

LP_TOLERANCE = 1e-9
CHECK_TOLERANCE = 1e-8  # numerical LP verification, not a scientific margin


def sharp_interval(mean, second, threshold, maximum):
    moment_interval(mean, second, threshold, maximum, use_second=True)
    require(maximum > 0 and 0 <= threshold < maximum, "NONTRIVIAL_MOMENT_EXTREMUM_REQUIRED")
    support = np.arange(maximum + 1, dtype=float)
    normalized = support / maximum
    matrix = np.stack((np.ones_like(support), normalized, normalized**2))
    rhs = np.array([1., mean/maximum, second/(maximum**2)])
    indicator = (support <= threshold).astype(float)
    solved = []
    for sign in (1., -1.):
        cost = sign * indicator
        result = linprog(cost, A_eq=matrix, b_eq=rhs, bounds=(0, None), method="highs-ds",
                         options={"threads": 1, "parallel": False,
                                  "primal_feasibility_tolerance": LP_TOLERANCE,
                                  "dual_feasibility_tolerance": LP_TOLERANCE})
        require(result.success and result.status == 0, "MOMENT_LP_FAILED:" + result.message)
        probability = result.x
        dual = result.eqlin.marginals
        primal_residual = float(np.max(np.abs(matrix @ probability - rhs)))
        raw_moment_residual = max(abs(float(probability @ support) - mean),
                                  abs(float(probability @ (support**2)) - second))
        dual_violation = max(0., float(np.max(matrix.T @ dual - cost)))
        # Subtracting this violation from the constant polynomial coefficient
        # gives a feasible dual minorant on every integer in physical support.
        corrected = dual.copy(); corrected[0] -= dual_violation
        dual_bound = float(rhs @ corrected)
        attained = float(cost @ probability)
        gap = attained - dual_bound
        require(np.all(probability >= 0) and primal_residual <= CHECK_TOLERANCE
                and raw_moment_residual <= CHECK_TOLERANCE
                and -CHECK_TOLERANCE <= gap <= CHECK_TOLERANCE, "MOMENT_PRIMAL_DUAL_CHECK")
        solved.append(dict(probability=probability, attained_cdf=float(indicator @ probability),
            dual_bound=dual_bound, corrected_dual=corrected.tolist(), primal_residual=primal_residual,
            dual_violation=dual_violation, raw_moment_residual=raw_moment_residual, primal_dual_gap=gap))
    lower = max(0., solved[0]["dual_bound"] - CHECK_TOLERANCE)
    upper = min(1., -solved[1]["dual_bound"] + CHECK_TOLERANCE)
    require(lower <= upper, "MOMENT_LP_CROSSING_INTERVAL")
    return (lower, upper), solved


def run(config_path, project_root, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status="FAILED_PFC_CDF_MOMENT_EXTREMA",source_simulations=0,new_independent_blocks=0,
        new_bootstrap_draws=0,population_moment_intervals_computed=False,method_adopted=False,
        confirmation_execution_released=False,scientific_fidelity_qualified=False,production_episodes=0)
    try:
        cfg = yaml.safe_load(config_path.read_text())
        require(cfg["schema"] == "pfc_cdf_moment_extrema_v1" and cfg["planning_blocks"] == 128
                and cfg["method_adopted"] is False and cfg["confirmation_execution_released"] is False, "MOMENT_REFINEMENT_SCOPE")
        files, reports = bound_inputs(project_root,cfg)
        prior = reports["screen"]
        require(prior["status"] == "COMPLETE_PFC_CDF_MOMENT_ENVELOPE_SCREEN"
                and prior["population_moment_intervals_computed"] is False, "COMPLETED_MOMENT_SCREEN_REQUIRED")
        rows = read_csv(files["screen"]["original_gap_moment_envelopes.csv"])
        require(len(rows) == 200, "ALL_ORIGINAL_GAPS_A_B_REQUIRED")
        cache, solutions, witnesses, output_rows, summaries = {}, [], {}, [], {"A":Counter(),"B":Counter()}
        for row in rows:
            realization = row["realization"]
            counts=summaries[realization]
            require(row["original_planning_status"] == "PLANNING_BAND_EXCEEDS_BUDGET", "ORIGINAL_GAP_IDENTITY")
            counts["original_gaps"]+=1
            out=dict(row)
            out.update(sharp_lower="",sharp_upper="",attainable_difference_lower="",attainable_difference_upper="",left_problem="",right_problem="")
            if row["two_moment_envelope_fits"] == "True":
                counts["prior_moment_envelope_fits"]+=1
                out["refinement_status"]="PRIOR_MOMENT_ENVELOPE_ALREADY_FITS"
            else:
                ident=json.loads(row["endpoint_identity"])
                results=[]
                for side in ("left","right"):
                    params=(float(row[side+"_mean"]),float(row[side+"_second_raw"]),int(ident["count_threshold"]),int(row["physical_maximum_count"]))
                    if params not in cache:
                        bounds, solved = sharp_interval(*params)
                        index=len(solutions)
                        for direction,value in zip(("minimum","maximum"),solved):
                            witnesses[f"problem_{index}_{direction}"]=value["probability"]
                        solutions.append(dict(problem_index=index,mean=params[0],second=params[1],threshold=params[2],maximum=params[3],
                            bounds=bounds,solutions=[{k:v for k,v in x.items() if k!="probability"} for x in solved]))
                        cache[params]=(index,bounds,solved)
                    index,bounds,solved=cache[params]
                    out[side+"_problem"]=index
                    results.append((bounds,solved))
                left,right=results
                lo,hi=left[0][0]-right[0][1],left[0][1]-right[0][0]
                attained_lo=left[1][0]["attained_cdf"]-right[1][1]["attained_cdf"]
                attained_hi=left[1][1]["attained_cdf"]-right[1][0]["attained_cdf"]
                margin=float(row["margin"])
                fits=max(abs(lo),abs(hi))<=margin
                outside=max(abs(attained_lo),abs(attained_hi))>margin+CHECK_TOLERANCE
                require(not (fits and outside), "MOMENT_EXTREMUM_INCONSISTENT_CLASSIFICATION")
                out.update(sharp_lower=lo,sharp_upper=hi,attainable_difference_lower=attained_lo,attainable_difference_upper=attained_hi,
                    refinement_status=("SHARP_EMPIRICAL_MOMENT_ENVELOPE_FITS" if fits else
                        "OUTSIDE_MARGIN_DISTRIBUTIONS_SHARE_THESE_TWO_MOMENTS" if outside else "NUMERICAL_PRECISION_UNRESOLVED"))
                counts["refined_gaps"]+=1
                counts["additional_sharp_fits"]+=int(fits)
                counts["two_moment_compatible_outside_margin"]+=int(outside)
                counts["numerically_unresolved"]+=int(not fits and not outside)
            output_rows.append(out)
        require(all(v["original_gaps"]==100 for v in summaries.values()), "ALL_A_B_GAPS_RETAINED")
        write_csv(output/"moment_extrema_by_endpoint.csv",output_rows)
        freeze_json(output/"moment_optimization_problems.json",solutions)
        np.savez_compressed(output/"extremal_count_distributions.npz",**witnesses)
        report.update(status="COMPLETE_PFC_CDF_MOMENT_EXTREMA_SCREEN",settings=cfg,settings_sha256=sha256_file(config_path),
            source_report_sha256=cfg["inputs"]["screen"]["report_sha256"],summaries=summaries,unique_moment_problems=len(solutions),
            linear_programs=2*len(solutions),scipy_version=scipy.__version__,solver="highs-ds",
            maximum_primal_residual=max(x["primal_residual"] for p in solutions for x in p["solutions"]),
            maximum_raw_moment_residual=max(x["raw_moment_residual"] for p in solutions for x in p["solutions"]),
            maximum_primal_dual_gap=max(x["primal_dual_gap"] for p in solutions for x in p["solutions"]),
            retained_domain_rows=84330,original_independent_blocks=32,maximum_confirmation_blocks=128,
            interpretation="Extremizers are finite-support distributions matching empirical marginal count moments, not simulated neural trajectories or admissible PFC input laws. They diagnose information missing from this two-moment channel. No population coverage, global method impossibility, scientific qualification, or confirmation release follows.",
            implementation_sha256=sha256_file(Path(__file__)),outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p)) for p in output.iterdir() if p.is_file()})
    except Exception as exc:
        report["error"]=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        report["wall_seconds"]=time.perf_counter()-started
        freeze_json(output/"report.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--project-root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();report=run(args.config,args.project_root,args.output)
    print(json.dumps(dict(status=report["status"],summaries=report["summaries"],linear_programs=report["linear_programs"])))


if __name__=="__main__":
    main()
