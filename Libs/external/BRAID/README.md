# BRAID reference core

Copied from `ShanechiLab/BRAID`, commit
`f9735960fdacff8351829100ca67408dd7e0351f`, `source/BRAID/`.
The original USC educational, research and non-profit terms are retained in
`LICENSE.md`; commercial use has separate terms.

One packaging change changes `from BRAID.tools import MatHelper` in `MainModel.py`
to the equivalent relative import. Other source code is retained unchanged.
The original package's legacy serialization aliases remain in its `__init__.py`.

The campaign adapter is `Libs/population_dynamics/braid.py`. It uses the native
neural-stage regression RNN, separate learned predictor/forward recursions,
known-input feedthrough and the source's Poisson readout option. It does not use
behavioral targets and does not implement the behavior-specific decomposition.
Its outputs are deterministic mean counts per bin, not joint spike samples.

The adapter constructs trial-local training tensors because the reference
`MainModel.fit` concatenates trials before `RNNModel.fit` shifts future targets.
That reference path can cross trial boundaries and drop the final trial/batch.
Every adapter trial contributes `T - max(steps_ahead)` prediction origins per
head, without cross-trial targets; shorter heads omit their extra tail targets.
Native recurrent cells, decoder and prediction-head losses remain in use.
The unused stage-1 prior is disabled in this neural-only configuration. Passing
even a neutral multiplicative prior to the original Poisson readout raises its
unimplemented missing-prior exception. No prior is needed without stage 1.

Use an isolated Python 3.9 environment and the campaign `requirements-braid.txt`.
The original NumPy 1.22.4 and TensorFlow 2.5.0 pins conflict. The adapter runtime
uses NumPy 1.19.5, TensorFlow CPU 2.5.0 and protobuf 3.20.3. Current validation
status belongs to the campaign evidence, not this source provenance note.
