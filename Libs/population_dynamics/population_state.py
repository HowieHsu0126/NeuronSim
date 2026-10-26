"""Source-indexed stochastic population dynamics with full-unit renewal output.

Field states interact through a learned field-to-field operator. Field indices
are supplied by the source observation map; they are not inferred anatomy.
The fixed and recurrent encoders share this exact transition/observation backend.
Closure selection is a separate, held-out decision, not a property asserted by
the existence of a recurrent encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor, nn

from .poisson_state_space import LatentRandomStreams, initial_kl
from .refractory_observation import observation_nll, sample_bin


@dataclass(frozen=True)
class PopulationBoundary:
    mean: Tensor                 # batch, source field, latent channel
    variance: Tensor
    ready_in_ticks: Tensor       # batch, unit
    unit_ids: tuple[int, ...]
    field_names: tuple[str, ...]
    bin_seconds: float
    source_metadata_identity: str | None = None


@dataclass(frozen=True)
class PopulationStep:
    counts: Tensor               # draw, batch, unit
    events: Tensor               # draw, batch, unit index, relative emission tick
    latent: Tensor
    ready_in_ticks: Tensor
    initial_latent: Tensor


class ControlledPopulationState(nn.Module):
    """Nonlinear interacting field SSM; native-clock output support is exact.

    Initial, process and observation randomness use distinct explicit streams.
    Conditional unit independence given the field path is a model assumption.
    """

    def __init__(self, *, unit_ids, unit_field_index, field_names,
                 minimum_interval_ticks, tick_seconds: float, bin_seconds: float,
                 input_size: int, latent_channels: int = 4, encoder_size: int = 16,
                 representation: str = "recurrent", initialization_seed: int = 41,
                 unit_chunk_size: int = 4096, metadata: dict | None = None,
                 control_geometry: str = "static"):
        super().__init__()
        if control_geometry not in ("static", "actuation_only"):
            raise ValueError("registered population control geometry required")
        self.control_geometry = control_geometry
        ids, names = tuple(unit_ids), tuple(field_names)
        if (not ids or any(type(x) is not int or x < 0 for x in ids)
                or len(set(ids)) != len(ids) or not names or len(set(names)) != len(names)
                or any(not isinstance(x, str) or not x for x in names)):
            raise ValueError("unique source unit identities and field names required")
        if (any(type(x) is not int or x < 1 for x in
                (latent_channels, encoder_size, unit_chunk_size))
                or type(input_size) is not int or input_size < 0
                or type(initialization_seed) is not int or initialization_seed not in (41, 42, 43)
                or representation not in ("fixed", "recurrent")):
            raise ValueError("registered representation, dimensions and training seed required")
        if (not all(math.isfinite(x) and x > 0 for x in (tick_seconds, bin_seconds))
                or not math.isclose(bin_seconds / tick_seconds,
                                    round(bin_seconds / tick_seconds), abs_tol=1e-9, rel_tol=0)):
            raise ValueError("positive model bins aligned to the native clock required")
        index, interval = torch.as_tensor(unit_field_index), torch.as_tensor(minimum_interval_ticks)
        if (index.shape != (len(ids),) or interval.shape != index.shape
                or index.dtype not in (torch.int32, torch.int64)
                or interval.dtype not in (torch.int32, torch.int64)
                or torch.any(index < 0) or torch.any(index >= len(names))
                or torch.any(interval < 1)):
            raise ValueError("source field assignment and native minimum intervals required")
        self.unit_ids, self.field_names = ids, names
        self.input_size, self.latent_channels, self.encoder_size = input_size, latent_channels, encoder_size
        self.representation, self.initialization_seed = representation, initialization_seed
        self.bin_seconds, self.tick_seconds = float(bin_seconds), float(tick_seconds)
        self.bin_ticks, self.unit_chunk_size = round(bin_seconds / tick_seconds), unit_chunk_size
        self.metadata = None
        self.source_metadata_identity = None
        if metadata is not None:
            self.metadata = self._metadata(metadata, len(ids), len(names), input_size)
            self.register_buffer("unit_features", torch.tensor(self.metadata["unit_features"], dtype=torch.float32))
            self.register_buffer("field_metadata", torch.tensor(self.metadata["field_features"], dtype=torch.float32))
            source = dict(metadata=self.metadata, unit_ids=ids, field_names=names,
                          unit_field_index=index.tolist(), minimum_interval_ticks=interval.tolist(),
                          tick_seconds=self.tick_seconds, bin_seconds=self.bin_seconds)
            if self.control_geometry != "static":
                source["control_geometry"] = self.control_geometry
            self.source_metadata_identity = hashlib.sha256(
                json.dumps(source, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if self.control_geometry == "actuation_only":
            self._validate_actuation_geometry(index, len(names))
        self.register_buffer("unit_field_index", index.long())
        self.register_buffer("minimum_interval_ticks", interval.long())
        self.register_buffer("field_unit_counts", torch.bincount(index.long(), minlength=len(names)))
        # Fixed source-ordered reductions avoid nondeterministic GPU scatter sums.
        for f in range(len(names)):
            self.register_buffer(f"field_units_{f}", torch.where(index == f)[0])
        f, k, h = len(names), latent_channels, encoder_size
        features = 3 + input_size  # mean rate, active-unit fraction, actual N
        if self.metadata is not None:
            features += self.field_metadata.shape[-1]
        # Construct under a restored RNG context; do not consume the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.encoder = (nn.GRU(features, h, batch_first=True) if representation == "recurrent"
                            else nn.Sequential(nn.Linear(features, h), nn.Tanh()))
            self.posterior_mean = nn.Linear(h, k)
            self.posterior_log_variance = nn.Linear(h, k)
        self.local_weight = nn.Parameter(torch.empty(k, k))
        generator = torch.Generator().manual_seed(initialization_seed)
        if self.metadata is None:
            self.field_mix_logits = nn.Parameter(torch.eye(f) * 2)
            self.control_weight = nn.Parameter(torch.empty(f, input_size, k))
            self.transition_bias = nn.Parameter(torch.zeros(f, k))
            self.decay_logit = nn.Parameter(torch.full((f, k), math.log(.98 / .02)))
            self.process_log_variance = nn.Parameter(torch.full((f, k), math.log(.01)))
            self.initial_mean = nn.Parameter(torch.zeros(f, k))
            self.initial_log_variance = nn.Parameter(torch.zeros(f, k))
            self.unit_loading = nn.Parameter(torch.empty(len(ids), k))
            self.log_hazard_bias = nn.Parameter(torch.zeros(len(ids)))
            initialized = (self.local_weight, self.control_weight, self.unit_loading)
        else:
            # Every fitted parameter has a size independent of source N and F.
            # Source descriptors and reduction indices remain circuit buffers.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(initialization_seed)
                d = self.field_metadata.shape[-1]
                self.shared_field_parameters = nn.Linear(d, (input_size + 5) * k)
                self.shared_query = nn.Linear(d, k, bias=False)
                self.shared_key = nn.Linear(d, k, bias=False)
                self.shared_unit_readout = nn.Linear(self.unit_features.shape[-1], k + 1)
            nn.init.zeros_(self.shared_field_parameters.weight)
            nn.init.zeros_(self.shared_field_parameters.bias)
            with torch.no_grad():
                offsets = (input_size * k, (input_size + 1) * k,
                           (input_size + 2) * k, (input_size + 3) * k)
                self.shared_field_parameters.bias[offsets[1]:offsets[2]].fill_(math.log(.98 / .02))
                self.shared_field_parameters.bias[offsets[2]:offsets[3]].fill_(math.log(.01))
            nn.init.normal_(self.shared_field_parameters.weight[:input_size * k], std=.05, generator=generator)
            nn.init.normal_(self.shared_unit_readout.weight, std=.05, generator=generator)
            nn.init.zeros_(self.shared_unit_readout.bias)
            initialized = (self.local_weight,)
        for parameter in initialized:
            nn.init.normal_(parameter, std=.05, generator=generator)
        nn.init.zeros_(self.posterior_log_variance.weight)
        nn.init.constant_(self.posterior_log_variance.bias, math.log(.1))

    def _tensor(self, value: Any) -> Tensor:
        parameter = self.posterior_mean.weight
        raw = torch.as_tensor(value, device=parameter.device)
        if raw.is_complex():
            raise ValueError("real population observations required")
        return raw.to(parameter.dtype)

    @staticmethod
    def _metadata(value, units, fields, controls):
        """Require explicit static-feature and control measurement contracts."""
        if set(value) != {"unit_feature_schema", "field_feature_schema", "control_features",
                          "unit_features", "field_features", "source_identity"}:
            raise ValueError("explicit static population metadata required")
        result = json.loads(json.dumps(value, allow_nan=False))
        identity = result["source_identity"]
        if (not isinstance(identity, dict)
                or set(identity) != {"circuit_id", "source_revision", "connectivity_identity", "parameter_identity"}
                or any(not isinstance(v, str) or not v for v in identity.values())):
            raise ValueError("explicit source circuit and realization identities required")
        for key, rows in (("unit", units), ("field", fields)):
            schema = result[key + "_feature_schema"]
            if (not schema or any(set(s) != {"name", "unit", "meaning", "transform"}
                or any(not isinstance(v, str) or not v for v in s.values()) for s in schema)
                or len({s["name"] for s in schema}) != len(schema)):
                raise ValueError("named feature units, meanings and transforms required")
            x = torch.as_tensor(result[key + "_features"])
            if x.shape != (rows, len(schema)) or x.is_complex() or not torch.isfinite(x).all():
                raise ValueError("finite aligned source static features required")
        if (len(result["control_features"]) != controls
                or any(not s.get("name") or not s.get("unit") for s in result["control_features"])
                or len({s["name"] for s in result["control_features"]}) != controls):
            raise ValueError("explicit source control feature semantics required")
        return result

    def _validate_actuation_geometry(self, field_index, fields):
        """Require source-indexed actuator masks and the native four controls."""
        if self.metadata is None or self.input_size != 4:
            raise ValueError("actuation_only requires source PFC target/control metadata")
        unit_names = [row["name"] for row in self.metadata["unit_feature_schema"]]
        field_names = [row["name"] for row in self.metadata["field_feature_schema"]]
        unit_targets = ("cue_target", "distractor_target")
        field_targets = ("cue_target_fraction", "distractor_target_fraction")
        if (any(name not in unit_names for name in unit_targets)
                or any(name not in field_names for name in field_targets)):
            raise ValueError("actuation_only requires source target mask feature schema")
        self._unit_target_columns = tuple(unit_names.index(name) for name in unit_targets)
        self._field_target_columns = tuple(field_names.index(name) for name in field_targets)
        controls = self.metadata["control_features"]
        if ([row["name"] for row in controls] !=
                ["cue_current", "distractor_current", "cue_active", "distractor_active"]
                or [row["unit"] for row in controls] !=
                ["nA", "nA", "dimensionless", "dimensionless"]
                or any(controls[index].get("geometry") !=
                       "source_index_target_masks_in_population_metadata" for index in (0, 1))
                or any(controls[index].get("role") != "source_assignment_amplitude" for index in (0, 1))
                or any(controls[index].get("role") != "derived_source_schedule_indicator" for index in (2, 3))):
            raise ValueError("actuation_only requires native four source control channels")
        for schema, columns in ((self.metadata["unit_feature_schema"], self._unit_target_columns),
                                (self.metadata["field_feature_schema"], self._field_target_columns)):
            if any(schema[column]["unit"] != "dimensionless"
                   or schema[column]["transform"] != "identity" for column in columns):
                raise ValueError("actuation_only target masks require dimensionless identity features")
        unit_masks = self.unit_features[:, self._unit_target_columns]
        fractions = self.field_metadata[:, self._field_target_columns]
        if (torch.any((unit_masks != 0) & (unit_masks != 1))
                or torch.any(fractions < 0) or torch.any(fractions > 1)):
            raise ValueError("actuation_only requires binary unit masks and field fractions")
        for field in range(fields):
            selected = field_index == field
            expected = (unit_masks[selected].mean(0) if selected.any() else
                        torch.zeros(2, dtype=unit_masks.dtype))
            if not torch.equal(fractions[field], expected):
                raise ValueError("actuation_only field fractions differ from source unit masks")

    def _intrinsic_field_metadata(self):
        if self.control_geometry == "static":
            return self.field_metadata
        intrinsic = self.field_metadata.clone()
        intrinsic[:, self._field_target_columns] = 0
        return intrinsic

    def _intrinsic_unit_features(self, start, stop):
        features = self.unit_features[start:stop]
        if self.control_geometry == "static":
            return features
        intrinsic = features.clone()
        intrinsic[:, self._unit_target_columns] = 0
        return intrinsic

    def _field_parameters(self):
        if self.metadata is None:
            return (self.control_weight, self.transition_bias, self.decay_logit,
                    self.process_log_variance, self.initial_mean, self.initial_log_variance)
        k = self.latent_channels
        values = self.shared_field_parameters(self._intrinsic_field_metadata())
        control = values[:, :self.input_size * k].reshape(len(self.field_names), self.input_size, k)
        if self.control_geometry == "actuation_only":
            fractions = self.field_metadata[:, self._field_target_columns]
            control = control * fractions[:, (0, 1, 0, 1), None]
        remainder = values[:, self.input_size * k:].split(k, -1)
        return (control, *remainder)

    def _sequence(self, counts, controls):
        y, u = self._tensor(counts), self._tensor(controls)
        if (y.ndim != 3 or min(y.shape) < 1 or y.shape[-1] != len(self.unit_ids)
                or not torch.isfinite(y).all() or torch.any(y < 0) or torch.any(y != y.round())):
            raise ValueError("complete batch/time/unit nonnegative integer counts required")
        if u.shape != (*y.shape[:2], self.input_size) or not torch.isfinite(u).all():
            raise ValueError("aligned finite known control sequence required")
        return y, u

    def field_features(self, counts: Tensor, controls: Tensor) -> Tensor:
        """Preserve empty fields and source population sizes; no soma imputation."""
        rows = []
        for f in range(len(self.field_names)):
            indices = getattr(self, f"field_units_{f}")
            if indices.numel():
                values = counts.index_select(-1, indices)
                mean = torch.log1p(values.mean(-1) / self.bin_seconds)
                active = (values > 0).to(values.dtype).mean(-1)
            else:
                mean = counts.new_zeros(counts.shape[:2]); active = mean
            size = torch.ones_like(mean) * torch.log1p(self.field_unit_counts[f].to(mean.dtype))
            known=controls
            if self.control_geometry=="actuation_only":
                fractions=self.field_metadata[f,list(self._field_target_columns)]
                known=controls*fractions[(0,1,0,1),]
            features = [torch.stack((mean, active, size), -1), known]
            if self.metadata is not None:
                features.append(self._intrinsic_field_metadata()[f].expand(*counts.shape[:2], -1))
            rows.append(torch.cat(features, -1))
        return torch.stack(rows, dim=2)

    def encode(self, counts, controls):
        y, u = self._sequence(counts, controls)
        x = self.field_features(y, u)
        b, t, f, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * f, t, d)
        encoded = self.encoder(x)
        hidden = encoded[0] if self.representation == "recurrent" else encoded
        mean = self.posterior_mean(hidden).reshape(b, f, t, -1).transpose(1, 2)
        variance = self.posterior_log_variance(hidden).exp().reshape(b, f, t, -1).transpose(1, 2)
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or torch.any(variance <= 0):
            raise FloatingPointError("invalid causal posterior")
        return mean, variance

    def transition_mean(self, z: Tensor, controls: Tensor) -> Tensor:
        if self.metadata is None:
            mixing = self.field_mix_logits.softmax(-1)
        else:
            # Learned relative interaction, not an asserted mechanistic graph.
            intrinsic = self._intrinsic_field_metadata()
            query, key = self.shared_query(intrinsic), self.shared_key(intrinsic)
            mixing = (query @ key.T / math.sqrt(self.latent_channels)).softmax(-1)
        context = torch.einsum("fg,...gk->...fk", mixing, z)
        control, bias, decay_logit, _, _, _ = self._field_parameters()
        drive = torch.einsum("...c,fck->...fk", controls, control)
        decay = decay_logit.sigmoid()
        return decay * z + (1 - decay) * torch.tanh(z @ self.local_weight.T + context + drive + bias)

    def log_hazard(self, z: Tensor, start: int = 0, stop: int | None = None) -> Tensor:
        stop = len(self.unit_ids) if stop is None else stop
        latent = z.index_select(-2, self.unit_field_index[start:stop])
        if self.metadata is None:
            loading, bias = self.unit_loading[start:stop], self.log_hazard_bias[start:stop]
        else:
            readout = self.shared_unit_readout(self._intrinsic_unit_features(start, stop))
            loading, bias = readout[:, :-1], readout[:, -1]
        return (latent * loading).sum(-1) + bias

    def sequence_loss(self, counts, controls, eligible_ticks, generator: torch.Generator):
        y, u = self._sequence(counts, controls)
        exposure = self._tensor(eligible_ticks)
        if (exposure.shape != y.shape or torch.any(exposure < y)
                or torch.any(exposure > self.bin_ticks) or torch.any(exposure != exposure.round())
                or not torch.isfinite(exposure).all()):
            raise ValueError("observed native eligible-tick exposure required")
        if torch.device(generator.device) != y.device:
            raise ValueError("variational RNG must be on the model device")
        mean, variance = self.encode(y, u)
        z = mean + variance.sqrt() * torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        normalizer = y.shape[0] * y.shape[1]
        observation = y.new_zeros(())
        for start in range(0, len(self.unit_ids), self.unit_chunk_size):
            stop = min(start + self.unit_chunk_size, len(self.unit_ids))
            observation = observation + observation_nll(
                y[..., start:stop], exposure[..., start:stop], self.log_hazard(z, start, stop),
                self.tick_seconds).sum() / normalizer
        control, _, _, process_log_variance, initial_mean, initial_log_variance = self._field_parameters()
        prior0 = initial_mean + torch.einsum("bc,fck->bfk", u[:, 0], control)
        initial = initial_kl(mean[:, 0], variance[:, 0], prior0,
                             initial_log_variance.exp()).sum() / normalizer
        prior = self.transition_mean(z[:, :-1], u[:, 1:])
        # Conditional KL is analytic; expectation over previous q is Monte Carlo.
        transition = initial_kl(mean[:, 1:], variance[:, 1:], prior,
                                process_log_variance.exp()).sum() / normalizer
        loss = observation + initial + transition
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite population negative ELBO")
        return dict(loss=loss, sampled_observation_nll=observation, initial_kl=initial, transition_kl=transition)

    def _ready(self, ready, batch: int):
        result = torch.as_tensor(ready, device=self.posterior_mean.weight.device)
        if (result.shape != (batch, len(self.unit_ids)) or result.dtype not in (torch.int32, torch.int64)
                or torch.any(result < 0) or torch.any(result >= self.minimum_interval_ticks)):
            raise ValueError("source-identified observed boundary availability required")
        return result.long()

    def infer_state(self, history: dict, metadata: dict | None = None):
        if metadata is not None and (tuple(metadata["unit_ids"]) != self.unit_ids
                                     or tuple(metadata["field_names"]) != self.field_names):
            raise ValueError("history metadata differs from the source observation map")
        mean, variance = self.encode(history["counts"], history["stimulus"])
        return PopulationBoundary(mean[:, -1], variance[:, -1],
                                  self._ready(history["ready_in_ticks"], mean.shape[0]),
                                  self.unit_ids, self.field_names, self.bin_seconds,
                                  self.source_metadata_identity)

    @torch.no_grad()
    def iter_simulation(self, state: PopulationBoundary, stimulus,
                        rng: LatentRandomStreams, n_samples: int, *, record_events: bool = True
                        ) -> Iterator[PopulationStep]:
        if (state.unit_ids != self.unit_ids or state.field_names != self.field_names
                or state.bin_seconds != self.bin_seconds
                or state.source_metadata_identity != self.source_metadata_identity):
            raise ValueError("boundary source/observation map mismatch")
        mean, variance, u = self._tensor(state.mean), self._tensor(state.variance), self._tensor(stimulus)
        b, f, k = mean.shape
        if (mean.shape != (b, len(self.field_names), self.latent_channels) or variance.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or torch.any(variance <= 0)
                or u.ndim != 3 or u.shape[0] != b or u.shape[1] < 1 or u.shape[2] != self.input_size
                or not torch.isfinite(u).all() or type(n_samples) is not int or n_samples < 1):
            raise ValueError("finite boundary, controls and positive draw count required")
        streams = (rng.initial, rng.dynamics, rng.observation)
        if (any(torch.device(g.device) != mean.device for g in streams)
                or len({g.initial_seed() for g in streams}) != 3 or len({id(g) for g in streams}) != 3):
            raise ValueError("three independent explicit device-local random streams required")
        shape = (n_samples, b, f, k)
        z = mean + variance.sqrt() * torch.randn(shape, dtype=mean.dtype, device=mean.device, generator=rng.initial)
        initial = z.clone()
        ready = self._ready(state.ready_in_ticks, b).expand(n_samples, -1, -1).clone()
        process = self._field_parameters()[3].exp()
        if not torch.isfinite(process).all() or torch.any(process <= 0):
            raise FloatingPointError("invalid process variance")
        units = len(self.unit_ids)
        for t in range(u.shape[1]):
            z = self.transition_mean(z, u[:, t]) + process.sqrt() * torch.randn(
                shape, dtype=mean.dtype, device=mean.device, generator=rng.dynamics)
            sampled = sample_bin(self.log_hazard(z), ready, self.minimum_interval_ticks,
                                 self.bin_ticks, self.tick_seconds, rng.observation, record_events=record_events)
            flat, within = sampled.events[:, 0], sampled.events[:, 1]
            events = torch.stack((flat // (b * units), (flat // units) % b,
                                  flat % units, within + t * self.bin_ticks), -1)
            ready = sampled.ready_in_ticks
            yield PopulationStep(sampled.counts, events, z, ready, initial)

    def configuration(self):
        config = dict(unit_ids=list(self.unit_ids), unit_field_index=self.unit_field_index.tolist(),
                    field_names=list(self.field_names), minimum_interval_ticks=self.minimum_interval_ticks.tolist(),
                    tick_seconds=self.tick_seconds, bin_seconds=self.bin_seconds, input_size=self.input_size,
                    latent_channels=self.latent_channels, encoder_size=self.encoder_size,
                    representation=self.representation, initialization_seed=self.initialization_seed,
                    unit_chunk_size=self.unit_chunk_size)
        if self.metadata is not None:
            config["metadata"] = json.loads(json.dumps(self.metadata, allow_nan=False))
        if self.control_geometry != "static":
            config["control_geometry"] = self.control_geometry
        return config

    def for_circuit(self, *, unit_ids, unit_field_index, field_names,
                    minimum_interval_ticks, metadata):
        """Copy fitted parameters into an independent source circuit model."""
        target = self._circuit_model(unit_ids=unit_ids, unit_field_index=unit_field_index,
            field_names=field_names, minimum_interval_ticks=minimum_interval_ticks, metadata=metadata)
        source_parameters = dict(self.named_parameters())
        with torch.no_grad():
            for name, value in target.named_parameters():
                value.copy_(source_parameters[name])
        return target

    def training_for_circuit(self, *, unit_ids, unit_field_index, field_names,
                             minimum_interval_ticks, metadata):
        """Bind source buffers while retaining the canonical optimizer parameters.

        Place the canonical model on its device before creating training views.
        Use one optimizer over the canonical model, and save its state once.
        Views have separate source buffers and modes; rebuild them after loading
        a canonical checkpoint or changing the model device or dtype.
        """
        target = self._circuit_model(unit_ids=unit_ids, unit_field_index=unit_field_index,
            field_names=field_names, minimum_interval_ticks=minimum_interval_ticks, metadata=metadata)
        for name, parameter in self.named_parameters():
            module, _, local = name.rpartition(".")
            target.get_submodule(module).register_parameter(local, parameter)
        return target

    def _circuit_model(self, *, unit_ids, unit_field_index, field_names,
                       minimum_interval_ticks, metadata):
        """Construct a source map without consuming the caller's random stream."""
        if self.metadata is None:
            raise ValueError("source-specific parameters cannot transfer to another circuit")
        for key in ("unit_feature_schema", "field_feature_schema", "control_features"):
            if metadata.get(key) != self.metadata[key]:
                raise ValueError("source feature or control measurement contract changed: " + key)
        config = self.configuration()
        config.update(unit_ids=unit_ids, unit_field_index=unit_field_index, field_names=field_names,
                      minimum_interval_ticks=minimum_interval_ticks, metadata=metadata)
        parameter = self.posterior_mean.weight
        target = type(self)(**config).to(device=parameter.device, dtype=parameter.dtype)
        target.train(self.training)
        return target

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            torch.save(dict(schema=("controlled_population_state_actuation_v1" if self.control_geometry == "actuation_only"
                                    else "controlled_population_state_v1"), configuration=self.configuration(),
                            state_dict=self.state_dict()), stream)

    @classmethod
    def load(cls, path: Path, *, device="cpu"):
        value = torch.load(path, weights_only=True, map_location=device)
        mode = value["configuration"].get("control_geometry", "static")
        expected_schema = ("controlled_population_state_actuation_v1" if mode == "actuation_only"
                           else "controlled_population_state_v1")
        if value["schema"] != expected_schema:
            raise ValueError("population state checkpoint schema mismatch")
        model = cls(**value["configuration"]).to(device=device, dtype=value["state_dict"]["posterior_mean.weight"].dtype)
        if model.metadata is not None:
            for name, expected in model.named_buffers():
                if not torch.equal(value["state_dict"][name], expected):
                    raise ValueError("population metadata checkpoint source buffers changed: " + name)
        model.load_state_dict(value["state_dict"], strict=True)
        return model
