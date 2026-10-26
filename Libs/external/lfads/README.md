# Original LFADS core

Source: lfads/models, commit fd700f0cb2e104544c445d9fbf3991d8388ff18a,
research/lfads/{lfads.py, distributions.py, utils.py, plot_lfads.py}.
The original Apache-2.0 license and source copyright headers are retained.

Compatibility edits: select tensorflow.compat.v1 in four imports, use package-relative
imports, and materialize the four zip sequences consumed by tf.case. The adapter
uses explicit Graph/Session objects; it does not switch global TensorFlow eager mode.

Adapter edits: derive batch size from the data placeholder (the source's self.batch_size
has no other in-file consumers), and expose unclipped Poisson log rates so the adapter
can reject saturation inconsistent with the source's unclipped likelihood. Source
GRU equations, observation likelihood, feedback, encoders and AR(1) prior are preserved.

The LFADS constructor also accepts optional `random_ops` hooks for the five
dropout sites and posterior Gaussian draws. `DiagonalGaussianFromInput` accepts
an optional normal-draw callable. Omitting these hooks preserves upstream behavior;
the sample and its log probability still share one standard-normal noise tensor.
Source priors and variable initializers are unchanged. The campaign adapter uses
stateless keys tied to training update and call site, enabling exact model/Adam/RNG
resume without depending on graph scheduling order. This changes the training
random-number schedule; it does not retroactively reproduce old training draws.
Old parameter checkpoints retain their verified inference behavior.

The supported project entry is Libs.population_dynamics.lfads. The upstream CLI and
whole-trial smoothing exports are not imported as the forecasting interface. The
original files and earlier compatibility evidence remain under
Output/neural_population_dynamics/baselines/source_runtime_checks/.

The optional model hparam `implicit_identity_readin: true` avoids constructing
an explicit units-by-units identity constant for a single session without an
alignment matrix or alignment bias. The upstream branch computes `x @ I + 0`
with nontrainable constants even when `do_train_readin` is true; the opt-in
branch passes `x` directly to the same encoders. Trainable variable names and
shapes, loss equations, generator and readout remain unchanged. The default
remains the original dense path. Opting in with multiple sessions or alignment
metadata raises an explicit error; those source paths are not modified.

Parameter checkpoint save/restore and posterior/generative output exports use
the existing variables and outputs. Explicit access to
`fns_in_fac_Ws[0]()` in implicit mode raises
`LFADS_IMPLICIT_IDENTITY_READIN_DENSE_MATRIX_EXPORT_UNSUPPORTED`, instead of
materializing an identity matrix. The zero-bias callback remains available.
This is an algebraic compatibility patch, not a new LFADS model or a fidelity
result. Tests in `Libs/population_dynamics/test_lfads_identity_readin.py` compare
the dense and implicit paths with shared assigned variables and fixed training
randomness, and inspect a 100,000-unit short-sequence graph. They do not establish
full six-second training capacity or convergence.

The optional model hparam `gradient_aggregation: tree` selects
`tf.AggregationMethod.EXPERIMENTAL_TREE` for the existing loss gradients;
`add_n` is the default. In TensorFlow 2.21, TREE and
EXPERIMENTAL_ACCUMULATE_N use the same fixed left fold of binary AddN operations.
This can release partial weight gradients earlier than a single high-fan-in
AddN. It does not serialize every gradient producer or guarantee a peak-memory
bound; executor scheduling and the order of available partials still matter.

This option is applied after constructing the forward graph. It preserves the
loss, parameter and Adam-slot shapes, initialized forward variables, and
stateless random-site schedule. Floating-point addition order changes, so
cross-policy gradients and training trajectories are checked numerically, not
claimed to be bitwise identical. Checkpoints record the policy in the adapter's
model config; exact replay is tested within the tree policy on the supported CPU
fixture. That test does not establish TensorFlow-v1 Session determinism for all
hardware, backend versions or workloads. The source commit attribution above
continues to identify the upstream equations; this is a compatibility option.

The 33-unit, 32-bin structure fixture checks binary fan-in and records leaf
producer order. With a two-bin controller lag, both controller encoders' matrix
gradients fold processing steps 30 down to 1; the 16-bin initial-condition
encoders fold 16 down to 1. These orders follow recurrent backpropagation's
dependency direction. The fixture establishes graph structure, not the executor's
peak memory or successful full-length, 100,000-unit training.

Dataset-name placeholders, string equality predicates and their `tf.case`
dispatch execute explicitly on CPU. TensorFlow has no GPU string-equality
kernel; blanket GPU placement therefore failed before numerical execution.
The selected parameter tensors feed the caller-selected numerical device.
Source encoders, generator, likelihood and optimizer retain that device and
strict session placement remains enabled; this is no implicit CPU fallback.

The source Saver and summary builders also use explicit CPU scopes for file
I/O and string-valued summary kernels; numerical training remains on the
selected execution device.

The adapter model config accepts `grappler_optimization: off` (default:
`default`). This disables Grappler's meta optimizer in its sessions and executes
the original graph with its control edges. Native stack samples from full-sequence
first updates identified dependency transitive reduction on CPU and static
inference for the memory-swapping cost estimator on GPU. The option preserves
neural units, bins, gradients and equations; full-sequence feasibility and memory
still require execution. Checkpoints retain the setting. GPU cross-backend checks
use the process setting `NVIDIA_TF32_OVERRIDE=0`; their fixed tolerances were not
met with the default GPU precision behavior in the recorded small fixture.
