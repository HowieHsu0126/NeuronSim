#!/bin/bash
#SBATCH --job-name=NrnSim_ClockNoiseCohort
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_ClockNoiseCohort-%j.log
set -euo pipefail
[[ $# == 3 ]] || exit 2
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$NRNSIM_BENCHMARK_PYTHON" -B - "$@" <<'PYTHON'
from pathlib import Path
import csv,hashlib,json,os,shutil,subprocess,sys,tempfile,traceback
extension,software,destination=map(Path,sys.argv[1:])
if destination.exists():
    raise FileExistsError("CLOCK_NOISE_COHORT_DESTINATION_EXISTS")
if os.environ.get("SLURMD_NODENAME") not in ("cpu03","cpu04") or os.environ.get("SLURM_JOB_GPUS"):
    raise ValueError("CLOCK_NOISE_COHORT_CPU_ALLOCATION_REQUIRED")
receipt=dict(status="FAILED_CLOCK_NOISE_COHORT",exit_code=1,job_id=int(os.environ["SLURM_JOB_ID"]),
    extension=str(extension),new_source_simulations=0,new_unique_episodes=0,
    scientific_qualification=False,production_episodes=0,commands=[],pairs=[],
    interpretation="Three simulated circuit identities, one observed noise-block contrast per circuit. Values remain separate by circuit; no sampling-uncertainty or equivalence verdict.")
with tempfile.TemporaryDirectory(prefix="neuronsim-clock-noise-cohort-") as temporary:
    work=Path(temporary);package=work/"Libs/population_dynamics";package.parent.mkdir()
    try:
        submission=json.loads((extension/"submission.json").read_text())
        preparation=json.loads((extension/"preparation.json").read_text())
        combined=Path(submission["combined_panel_job"]["destination"])
        delivery=json.loads((combined/"delivery.json").read_text())
        if delivery["exit_code"]!=0 or delivery["verified_pairs"]!=submission["combined_panel_job"]["expected_pairs"]:
            raise ValueError("CLOCK_NOISE_COHORT_COMPLETE_COMBINED_PANEL_REQUIRED")
        validation=json.loads((software.parent/"delivery.json").read_text())
        if validation["exit_code"]!=0 or not validation["legacy_csv_exact"]:
            raise ValueError("CLOCK_NOISE_COHORT_VERIFIED_LEGACY_RESULTS_REQUIRED")
        receipt["combined_panel"]=str(combined/"panel")
        shutil.copytree(software,package)
        env=dict(os.environ,PYTHONPATH=str(work/"Libs"),TMPDIR=str(work))
        merged={"noise":[],"binning":[]};headers={}
        pairings=preparation["circuit_block_pairs"]
        if len(pairings)!=3 or len({pair[0] for pair in pairings})!=3:
            raise ValueError("CLOCK_NOISE_THREE_DISTINCT_DECLARED_CIRCUITS_REQUIRED")
        for circuit,first,second in pairings:
            circuit_id="v1_visual_circuit_%03d"%circuit
            record=dict(circuit_id=circuit_id,block_indices=[first,second],
                        low_contrast_direction_deg=45*(first%8),sources={})
            reused=[circuit,first,second]==preparation["prior_pair_reused"]
            for label,module in (("noise","v1_clock_noise_comparison"),("binning","v1_clock_binning_comparison")):
                if reused:
                    source=software.parent/("legacy_"+label)
                else:
                    source=work/circuit_id/label
                    command=[sys.executable,"-B","-m","population_dynamics."+module,
                        "--panel",str(combined/"panel"),"--circuit-id",circuit_id,
                        "--first-block",str(first),"--second-block",str(second),"--output",str(source)]
                    receipt["commands"].append(command)
                    with (work/(circuit_id+"_"+label+".log")).open("w") as log:
                        result=subprocess.run(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError("CLOCK_NOISE_COHORT_COMMAND_FAILED:"+circuit_id+":"+label)
                report_path=source/"report.json";report=json.loads(report_path.read_text())
                if (report["exit_code"]!=0 or report["circuit_id"]!=circuit_id
                        or report["block_indices"]!=[first,second] or report["units"]!=230924
                        or report["noise_block_pairs"]!=1):
                    raise ValueError("CLOCK_NOISE_COHORT_REPORT_IDENTITY_DIFFERS")
                if label=="binning" and (report["bin_ms"]!=[1,5,10] or report["lag_ms"]!=[0,10,100]
                        or report["window"]!=dict(start_ms=501,stop_ms_exclusive=2991)):
                    raise ValueError("CLOCK_NOISE_COHORT_BINNING_PROTOCOL_DIFFERS")
                path=source/"comparison.csv";data=path.read_bytes();artifact=report["artifacts"]["comparison.csv"]
                digest=hashlib.sha256(data).hexdigest()
                if len(data)!=artifact["bytes"] or digest!=artifact["sha256"]:
                    raise ValueError("CLOCK_NOISE_COHORT_COMPARISON_CHANGED")
                with path.open(newline="") as stream:
                    reader=csv.DictReader(stream);rows=list(reader);columns=reader.fieldnames
                expected=310 if label=="noise" else 240
                if len(rows)!=expected or report["comparison_rows"]!=expected:
                    raise ValueError("CLOCK_NOISE_COHORT_INCOMPLETE_COMPARISON_ROWS")
                if label in headers and columns!=headers[label]:
                    raise ValueError("CLOCK_NOISE_COHORT_COLUMNS_DIFFER")
                headers[label]=columns
                for row in rows:
                    merged[label].append(dict(circuit_id=circuit_id,first_block_index=first,
                        second_block_index=second,low_contrast_direction_deg=45*(first%8),**row))
                record["sources"][label]=dict(report=str(report_path if reused else destination/report_path.relative_to(work)),report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                    comparison_sha256=digest,rows=len(rows),reused=reused)
            receipt["pairs"].append(record)
        receipt["artifacts"]={}
        for label,rows in merged.items():
            path=work/(label+"_by_circuit.csv")
            with path.open("x",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            receipt["artifacts"][path.name]=dict(rows=len(rows),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        receipt.update(status="COMPLETE_THREE_CIRCUIT_CLOCK_NOISE_COMPARISON",exit_code=0,
                       circuit_identities=3,observed_noise_block_pairs=3,reused_pairs=1,new_pair_analyses=2)
    except BaseException as error:
        receipt.update(error_type=type(error).__name__,error=str(error))
        (work/"error.log").write_text(traceback.format_exc())
    finally:
        (work/"report.json").write_text(json.dumps(receipt,indent=2)+"\n")
        shutil.copytree(work,destination,symlinks=True)
print(json.dumps({key:receipt[key] for key in ("status","exit_code","job_id")}))
raise SystemExit(receipt["exit_code"])
PYTHON
