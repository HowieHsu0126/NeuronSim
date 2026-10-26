# Optional NEST GPU reference candidate

Official repository: https://github.com/nest/nest-gpu; revision 74fc021cad2b4bd5c011c390b3f95a302b663483. Source headers use GPL version 2 or later; the supplied LICENSE is the upstream GPL-2 text. Source archive and intake provenance are in Output/neural_population_dynamics/mechanistic_reference/nest_gpu_archive_source/.

The MPI-disabled upstream build has an unguarded getMPIComm override even though its base declaration and backing member require HAVE_MPI. patches/no_mpi_accessor.patch applies that same guard to the override. Apply it to the pinned source, then configure task-local build/install directories with -Dwith-mpi=OFF, the allocated device's verified architecture, and the verified CUDA compiler. No neuronal, synaptic, event or random-generator source was changed.

The verified build used CUDA 12.4 and architecture 89. Commands, output and patch identities are in nest_gpu_no_mpi_build/no_mpi_build_report.json under the mechanistic-reference output directory. Unpatched-build and absent-MPI failures remain separately archived. Source install/bin/nestgpu_vars.sh and preserve its PYTHONPATH entries when adding the project path.

The candidate remains unqualified: the matched single-cell protocol passes34/42, with continuation and off-grid refractory failures. Native spatial/pairwise-Bernoulli rules are unavailable in this revision; explicit source-target pairs are supported but Mesocircuit connectivity and background noise still need qualification. This optional local runtime does not change a shared environment.

`patches/live_spike_schedule.patch` adds a local future-only device interface for
one zero-origin continuous run. It accepts absolute integer recorded generator
ticks, checks the live iteration, resets the consumed schedule index, supports
an empty replacement, and reads back the device schedule. The Python consumer
is `Libs/population_dynamics/nest_gpu_live_control.py`. Folded TC relays retain
pending arrivals of already committed source emissions. The interface does not
recalibrate, restart the clock, reset random streams or clear network queues.
Historical libraries and receipts remain bound to their original hashes. Native
canary and coupled-scale checks are required before intervention use.
