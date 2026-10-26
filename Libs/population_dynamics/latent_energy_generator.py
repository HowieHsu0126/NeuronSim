"""Paper-specified masked latent energy generation with explicit random streams.

The caller encodes the observed prefix separately and initializes this module
under a registered training seed. Every subsequent random operation uses the
supplied generator. Known per-time controls are condition tokens, not responses.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

NORM_EPSILON = 1e-6


def _initialize_linear_norm(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"positive integer {name} required")


def _finite_number(value, name, *, minimum=0.0, strict=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value <= minimum if strict else value < minimum)):
        raise ValueError(f"finite {'positive' if strict else 'nonnegative'} {name} required")


def _dropout(value, probability, generator, training):
    if not training or probability == 0:
        return value
    keep = torch.rand(value.shape, device=value.device, generator=generator) >= probability
    return value * keep.to(value.dtype) / (1.0 - probability)


def energy_score_terms(first, second, target, *, alpha=1.0, temperature=1.0):
    """Two independent draws; the positive repulsion distance is subtracted."""
    _finite_number(alpha, "energy alpha", strict=True)
    _finite_number(temperature, "training temperature", strict=True)
    if alpha >= 2:
        raise ValueError("energy alpha must be strictly below two")
    if (first.shape != second.shape or first.shape != target.shape or first.ndim < 2
            or not first.numel() or first.device != second.device or first.device != target.device
            or any(not x.is_floating_point() or not torch.isfinite(x).all()
                   for x in (first, second, target))):
        raise ValueError("matched finite nonempty latent samples and targets required")
    # Avoid low-precision norm arithmetic under autocast while preserving float64.
    values = [x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
              for x in (first, second, target)]
    first, second, target = values
    attraction = ((first - target).norm(dim=-1).pow(alpha)
                  + (second - target).norm(dim=-1).pow(alpha)).mean()
    repulsion = (first - second).norm(dim=-1).pow(alpha).mean()
    return dict(loss=attraction - temperature * repulsion,
                attraction=attraction, repulsion=repulsion)


def cosine_commit_counts(future_bins, steps):
    """Quantize a cosine remaining-mask schedule to exactly ``steps`` commits.

    Each round commits at least one position and reserves one for each later
    round. This makes integer rounding explicit; the requested step count never
    changes. Positions are selected from one fixed random permutation per draw.
    """
    _positive_integer(future_bins, "future_bins")
    _positive_integer(steps, "steps")
    if steps > future_bins:
        raise ValueError("steps must not exceed future_bins")
    remaining, counts = future_bins, []
    for step in range(steps):
        cosine_remaining = math.floor(future_bins * math.cos(math.pi * (step + 1) / (2 * steps)))
        after = min(remaining - 1, max(steps - step - 1, cosine_remaining))
        counts.append(remaining - after)
        remaining = after
    return tuple(counts)


class _AttentionBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio, attention_dropout, projection_dropout):
        super().__init__()
        self.heads = heads
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout
        self.norm_attention = nn.LayerNorm(width, eps=NORM_EPSILON)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width, eps=NORM_EPSILON)
        self.mlp_in = nn.Linear(width, width * mlp_ratio)
        self.mlp_out = nn.Linear(width * mlp_ratio, width)

    def forward(self, value, *, generator, training):
        batch, length, width = value.shape
        qkv = self.qkv(self.norm_attention(value)).reshape(
            batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        query, key, content = qkv.unbind(0)
        weights = (query @ key.transpose(-2, -1)) * (width // self.heads) ** -0.5
        norm_dtype = torch.float32 if weights.dtype in (torch.float16, torch.bfloat16) else weights.dtype
        weights = weights.softmax(dim=-1, dtype=norm_dtype).to(content.dtype)
        weights = _dropout(weights, self.attention_dropout, generator, training)
        attended = (weights @ content).transpose(1, 2).reshape(batch, length, width)
        value = value + _dropout(self.projection(attended), self.projection_dropout, generator, training)
        hidden = _dropout(F.gelu(self.mlp_in(self.norm_mlp(value))),
                          self.projection_dropout, generator, training)
        return value + _dropout(self.mlp_out(hidden), self.projection_dropout, generator, training)


class _AdaLNBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=NORM_EPSILON)
        self.modulation = nn.Linear(width, 3 * width)
        self.linear_in = nn.Linear(width, width)
        self.linear_out = nn.Linear(width, width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, value, noise, *, training, temperature):
        shift, scale, gate = self.modulation(F.silu(noise)).chunk(3, dim=-1)
        # Inference temperature acts only on this residual block's noise shift.
        if not training:
            shift = shift * temperature
        hidden = self.norm(value) * (1 + scale) + shift
        hidden = F.silu(self.linear_in(hidden))
        residual = self.linear_out(hidden)
        return value + gate * residual


class _EnergyHead(nn.Module):
    def __init__(self, context_width, latent_channels, width, depth, noise_channels):
        super().__init__()
        self.context = nn.Linear(context_width, width)
        self.noise = nn.Linear(noise_channels, width)
        self.blocks = nn.ModuleList([_AdaLNBlock(width) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=NORM_EPSILON)
        self.final_modulation = nn.Linear(width, 2 * width)
        self.output = nn.Linear(width, latent_channels)
        self.apply(_initialize_linear_norm)
        for block in self.blocks:
            nn.init.zeros_(block.modulation.weight)
            nn.init.zeros_(block.modulation.bias)
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, context, noise, *, training, temperature):
        value, noise = self.context(context), self.noise(noise)
        for block in self.blocks:
            value = block(value, noise, training=training, temperature=temperature)
        # Final-layer modulation is deliberately independent of infer temperature.
        shift, scale = self.final_modulation(F.silu(noise)).chunk(2, dim=-1)
        return self.output(self.final_norm(value) * (1 + scale) + shift)


class LatentEnergyGenerator(nn.Module):
    """EAG latent prior; tensors use [batch,time,channels].

    Production defaults implement the paper's MC Maze dimensions. Dropout 0.1
    defaults are source defaults outside the paper's hyperparameter table.
    ``sample`` disables dropout without changing the module's training mode.
    """
    def __init__(self, latent_channels, control_size, max_tokens, *, encoder_width=256,
                 encoder_depth=4, encoder_heads=4, decoder_width=256, decoder_depth=4,
                 decoder_heads=4, mlp_ratio=4, score_mlp_width=768, score_mlp_depth=6,
                 noise_channels=64, alpha=1.0, training_temperature=1.0,
                 mask_ratio_min=0.7, mask_ratio_max=1.0, label_dropout=0.1,
                 attention_dropout=0.1, projection_dropout=0.1, guidance_schedule="linear"):
        super().__init__()
        dimensions = dict(latent_channels=latent_channels, control_size=control_size,
            max_tokens=max_tokens, encoder_width=encoder_width, encoder_depth=encoder_depth,
            encoder_heads=encoder_heads, decoder_width=decoder_width, decoder_depth=decoder_depth,
            decoder_heads=decoder_heads, mlp_ratio=mlp_ratio, score_mlp_width=score_mlp_width,
            score_mlp_depth=score_mlp_depth, noise_channels=noise_channels)
        for name, value in dimensions.items():
            _positive_integer(value, name)
        if encoder_width < 4 or encoder_width % encoder_heads or decoder_width % decoder_heads:
            raise ValueError("attention widths must be divisible by their head counts")
        if guidance_schedule not in ("linear", "constant"):
            raise ValueError("linear or constant guidance schedule required")
        _finite_number(alpha, "energy alpha", strict=True)
        _finite_number(training_temperature, "training temperature", strict=True)
        if alpha >= 2:
            raise ValueError("energy alpha must be strictly below two")
        for name, value in dict(label_dropout=label_dropout, attention_dropout=attention_dropout,
                                projection_dropout=projection_dropout).items():
            _finite_number(value, name)
            if value >= 1:
                raise ValueError("dropout probabilities must be below one")
        if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in (mask_ratio_min, mask_ratio_max))
                or not 0 < mask_ratio_min <= mask_ratio_max <= 1):
            raise ValueError("ordered mask ratios in (0,1] required")
        self.configuration = dict(**dimensions, alpha=alpha, training_temperature=training_temperature,
            mask_ratio_min=mask_ratio_min, mask_ratio_max=mask_ratio_max, label_dropout=label_dropout,
            attention_dropout=attention_dropout, projection_dropout=projection_dropout,
            guidance_schedule=guidance_schedule)
        self.latent_channels, self.control_size, self.max_tokens = latent_channels, control_size, max_tokens
        self.noise_channels, self.alpha = noise_channels, alpha
        self.training_temperature = training_temperature
        self.mask_ratio_min, self.mask_ratio_max = mask_ratio_min, mask_ratio_max
        self.label_dropout = label_dropout
        self.guidance_schedule = guidance_schedule
        self.projection_dropout = projection_dropout
        self.latent_projection = nn.Linear(latent_channels, encoder_width)
        self.control_projection = nn.Sequential(
            nn.Linear(control_size, encoder_width // 4), nn.ReLU(),
            nn.Linear(encoder_width // 4, encoder_width // 2), nn.ReLU(),
            nn.Linear(encoder_width // 2, encoder_width))
        self.null_control = nn.Parameter(torch.empty(encoder_width))
        # Separate condition/latent positions retain the source's two token sets.
        self.encoder_positions = nn.Parameter(torch.empty(2, max_tokens, encoder_width))
        self.decoder_positions = nn.Parameter(torch.empty(2, max_tokens, decoder_width))
        self.energy_positions = nn.Parameter(torch.empty(max_tokens, decoder_width))
        self.mask_token = nn.Parameter(torch.empty(decoder_width))
        for value in (self.null_control, self.encoder_positions, self.decoder_positions,
                      self.energy_positions, self.mask_token):
            nn.init.normal_(value, std=0.02)
        self.input_norm = nn.LayerNorm(encoder_width, eps=NORM_EPSILON)
        self.encoder = nn.ModuleList([_AttentionBlock(encoder_width, encoder_heads, mlp_ratio,
            attention_dropout, projection_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = nn.LayerNorm(encoder_width, eps=NORM_EPSILON)
        self.to_decoder = nn.Linear(encoder_width, decoder_width)
        self.decoder = nn.ModuleList([_AttentionBlock(decoder_width, decoder_heads, mlp_ratio,
            attention_dropout, projection_dropout) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_width, eps=NORM_EPSILON)
        self.apply(_initialize_linear_norm)
        self.energy_head = _EnergyHead(decoder_width, latent_channels, score_mlp_width,
                                      score_mlp_depth, noise_channels)

    def get_extra_state(self):
        return dict(schema="latent_energy_generator_v1", configuration=dict(self.configuration))

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("energy generator checkpoint configuration differs")

    def _validate(self, prefix, controls, future_bins, generator, target=None):
        _positive_integer(future_bins, "future_bins")
        if not isinstance(generator, torch.Generator):
            raise ValueError("explicit torch.Generator required")
        reference = self.encoder_positions
        stream_device = torch.device(generator.device)
        if stream_device.type == 'cuda' and stream_device.index is None:
            stream_device = torch.device('cuda', torch.cuda.current_device())
        if stream_device != reference.device:
            raise ValueError("generator and model must use the same device")
        values = (prefix, controls) if target is None else (prefix, controls, target)
        if any(not isinstance(x, torch.Tensor) or x.ndim != 3 or not x.is_floating_point()
               or x.device != reference.device or x.dtype != reference.dtype
               or not torch.isfinite(x).all() for x in values):
            raise ValueError("finite rank-three inputs with model dtype and device required")
        batch, past, channels = prefix.shape
        if (batch < 1 or channels != self.latent_channels or past + future_bins > self.max_tokens
                or controls.shape != (batch, past + future_bins, self.control_size)
                or (target is not None and target.shape != (batch, future_bins, self.latent_channels))):
            raise ValueError("matched prefix/known-control shapes within max_tokens required")

    def _context(self, values, controls, masked, *, drop_labels, generator, training):
        batch, length, _ = values.shape
        visible_counts = (~masked).sum(dim=1)
        if (masked.dtype != torch.bool or masked.shape != values.shape[:2]
                or not torch.equal(visible_counts, visible_counts[:1].expand_as(visible_counts))):
            raise ValueError("a common visible-token count per batch is required")
        visible = torch.arange(length, device=values.device).expand(batch, -1)[~masked].reshape(batch, -1)
        # Gather before projection: hidden target values never enter the encoder.
        latent = values.gather(1, visible[..., None].expand(-1, -1, self.latent_channels))
        latent = self.latent_projection(latent) + self.encoder_positions[1, visible]
        control = self.control_projection(controls)
        control = torch.where(drop_labels[:, None, None], self.null_control, control)
        control = control + self.encoder_positions[0, :length]
        encoded = self.input_norm(torch.cat((control, latent), dim=1))
        for block in self.encoder:
            encoded = block(encoded, generator=generator, training=training)
        encoded = self.to_decoder(self.encoder_norm(encoded))
        latent = self.mask_token.expand(batch, length, -1).clone()
        latent = latent.scatter(1, visible[..., None].expand(-1, -1, latent.shape[-1]), encoded[:, length:])
        latent = latent + self.decoder_positions[1, :length]
        control = encoded[:, :length] + self.decoder_positions[0, :length]
        decoded = torch.cat((control, latent), dim=1)
        for block in self.decoder:
            decoded = block(decoded, generator=generator, training=training)
        return self.decoder_norm(decoded[:, length:]) + self.energy_positions[:length]

    def _noise(self, shape, reference, generator):
        return torch.rand((*shape, self.noise_channels), device=reference.device,
                          dtype=reference.dtype, generator=generator) - 0.5

    def loss(self, future_target, prefix, controls, *, generator, mask_ratio=None):
        if not isinstance(future_target, torch.Tensor) or future_target.ndim != 3:
            raise ValueError("future_target [batch,future,latent_channels] required")
        future = future_target.shape[1]
        self._validate(prefix, controls, future, generator, future_target)
        batch, past, _ = prefix.shape
        if mask_ratio is None:
            ratio = self.mask_ratio_min + (self.mask_ratio_max - self.mask_ratio_min) * float(
                torch.rand((), device=prefix.device, generator=generator))
        else:
            # Accumulation shares one source mask ratio across a logical batch.
            _finite_number(mask_ratio, "training mask ratio")
            if not self.mask_ratio_min <= mask_ratio <= self.mask_ratio_max:
                raise ValueError("training mask ratio outside the registered interval")
            ratio = mask_ratio
        hidden = max(1, math.ceil(future * ratio))
        order = torch.rand((batch, future), device=prefix.device, generator=generator).argsort(dim=1, stable=True)
        selected = order[:, :hidden]
        masked = torch.zeros((batch, past + future), device=prefix.device, dtype=torch.bool)
        masked.scatter_(1, selected + past, True)
        dropped = (torch.rand((batch,), device=prefix.device, generator=generator) < self.label_dropout
                   if self.training and self.label_dropout else torch.zeros(batch, device=prefix.device, dtype=torch.bool))
        context = self._context(torch.cat((prefix, future_target), dim=1), controls, masked,
                                drop_labels=dropped, generator=generator, training=self.training)
        context = context.gather(1, (selected + past)[..., None].expand(-1, -1, context.shape[-1]))
        target = future_target.gather(1, selected[..., None].expand(-1, -1, self.latent_channels))
        draws = [self.energy_head(context, self._noise(context.shape[:2], context, generator),
                 training=self.training, temperature=1.0) for _ in range(2)]
        terms = energy_score_terms(*draws, target, alpha=self.alpha, temperature=self.training_temperature)
        return dict(**terms, masked_tokens=batch * hidden, mask_ratio=hidden / future)

    def _predict_positions(self, values, controls, masked, positions, *, generator, guidance, temperature):
        batch = values.shape[0]
        conditional = self._context(values, controls, masked,
            drop_labels=torch.zeros(batch, device=values.device, dtype=torch.bool), generator=generator, training=False)
        context = conditional.gather(1, positions[..., None].expand(-1, -1, conditional.shape[-1]))
        if guidance != 1:
            unconditional = self._context(values, controls, masked,
                drop_labels=torch.ones(batch, device=values.device, dtype=torch.bool), generator=generator, training=False)
            unconditioned = unconditional.gather(1, positions[..., None].expand(-1, -1, unconditional.shape[-1]))
            context = unconditioned + guidance * (context - unconditioned)
        # Guidance acts on conditioning features; Uniform noise is not rescaled.
        return self.energy_head(context, self._noise(context.shape[:2], context, generator),
                                training=False, temperature=temperature)

    @torch.no_grad()
    def sample(self, prefix, controls, future_bins, *, generator, steps=50, guidance=4.0, temperature=0.7):
        self._validate(prefix, controls, future_bins, generator)
        commits = cosine_commit_counts(future_bins, steps)
        _finite_number(guidance, "guidance")
        _finite_number(temperature, "inference temperature", strict=True)
        batch, past, _ = prefix.shape
        values = torch.cat((prefix, prefix.new_zeros(batch, future_bins, self.latent_channels)), dim=1)
        masked = torch.zeros((batch, past + future_bins), device=prefix.device, dtype=torch.bool)
        masked[:, past:] = True
        order = torch.rand((batch, future_bins), device=prefix.device, generator=generator).argsort(dim=1, stable=True)
        completed = 0
        for count in commits:
            positions = past + order[:, completed:completed + count]
            # The source linear schedule keeps a final one-token mask sentinel.
            remaining_for_guidance = max(1, future_bins - completed - count)
            scheduled_guidance = (1 + (guidance - 1) * (future_bins - remaining_for_guidance) / future_bins
                                  if self.guidance_schedule == "linear" else guidance)
            proposed = self._predict_positions(values, controls, masked, positions,
                                               generator=generator, guidance=scheduled_guidance, temperature=temperature)
            if not torch.isfinite(proposed).all():
                raise ValueError("nonfinite generated latent values")
            values.scatter_(1, positions[..., None].expand(-1, -1, self.latent_channels), proposed)
            masked.scatter_(1, positions, False)
            completed += count
        return values[:, past:].clone()
