#!/usr/bin/env python3
"""Build the manuscript with latexmk in a Slurm node's temporary directory."""

import argparse
import gzip
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def compile_on_node(args):
    """Keep TeX writes local; publish outputs only after latexmk succeeds."""
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURMD_NODENAME") != args.node:
        raise RuntimeError("The worker must run inside the requested Slurm allocation")
    source = args.source
    destination = source / "build"
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        [str(args.tex_bin), str(args.perl.parent), environment.get("PATH", "")]
    )
    environment["TMPDIR"] = "/tmp"
    with tempfile.TemporaryDirectory(prefix="neuronsim-latex-", dir="/tmp") as temporary:
        work = Path(temporary) / "source"
        shutil.copytree(source, work, ignore=shutil.ignore_patterns("build", ".git"))
        command = [
            str(args.perl), str(args.tex_bin / "latexmk"),
            "-pdf", "-synctex=1", "-interaction=nonstopmode", "-halt-on-error",
            "-file-line-error", "-outdir=build", "main.tex",
        ]
        print("Compiling in job " + os.environ["SLURM_JOB_ID"], flush=True)
        result = subprocess.run(command, cwd=work, env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # Map diagnostics and source navigation back to the editable files.
        old_path, new_path = os.fsencode(work), os.fsencode(source)
        sys.stdout.buffer.write(result.stdout.replace(old_path, new_path))
        sys.stdout.flush()
        destination.mkdir(exist_ok=True)
        log = work / "build/main.log"
        if log.exists():
            (destination / "main.log").write_bytes(log.read_bytes().replace(old_path, new_path))
        if result.returncode:
            return result.returncode
        for extension in ["pdf", "synctex.gz", "fls", "bbl"]:
            built = work / ("build/main." + extension)
            if not built.exists():
                continue
            output = destination / built.name
            if extension == "synctex.gz":
                with gzip.open(built, "rb") as handle:
                    data = handle.read().replace(old_path, new_path)
                with gzip.open(output, "wb") as handle:
                    handle.write(data)
            elif extension == "fls":
                output.write_bytes(built.read_bytes().replace(old_path, new_path))
            else:
                shutil.copyfile(built, output)
        print("PDF: " + str(destination / "main.pdf"), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tex-bin", type=Path, required=True)
    parser.add_argument("--perl", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ["source", "tex_bin", "perl"]:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not (args.source / "main.tex").is_file():
        parser.error("--source must contain main.tex")
    if args.worker:
        return compile_on_node(args)
    repository = Path(__file__).resolve().parents[2]
    prefix = (repository / "PROJECT_PREFIX").read_text(encoding="utf-8").strip()
    command = [
        "srun", "--partition=" + args.partition, "--nodelist=" + args.node,
        "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=1G",
        "--time=00:05:00", "--job-name=" + prefix + "_latex_build", "--chdir=/tmp",
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--source", str(args.source), "--tex-bin", str(args.tex_bin),
        "--perl", str(args.perl), "--node", args.node, "--partition", args.partition,
    ]
    environment = os.environ.copy()
    environment["TMPDIR"] = "/tmp"
    return subprocess.call(command, env=environment)


if __name__ == "__main__":
    sys.exit(main())
