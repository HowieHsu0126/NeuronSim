"""Independent paper reimplementation of MemNO S4FFNO (SSTSS).

Source: ICLR 2025, arXiv:2409.02313v2, Sections 5.3/5.4 and
Appendices B/G.2/K. This is not recovered MemNO author model code.
Four FFNO spatial layers surround one causal residual S4D layer. Source
populations are channels on a complete periodic spatial grid. The caller
declares rate units and any fixed positive normalization; the network does
not change the observation map or create identified neuron outputs.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

# Use the already pinned, MIT-licensed S4 implementation. The runner adds
# Libs/external/LDNS to PYTHONPATH, as the existing LDNS runner does.
from ldns.networks.s4 import FFTConv

PAPER_URL = "https://arxiv.org/html/2409.02313v2"
IMPLEMENTATION = "MemNO-paper-reimplementation-S4FFNO-SSTSS"
HIDDEN_CHANNELS = 64
MLP_EXPANSION = 4
MEMORY_STATE_DIMENSION = 64


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class FactorizedFourierLayer(nn.Module):
    """Sum full-mode Fourier kernels along the two periodic spatial axes."""

    def __init__(self, hidden, grid_shape):
        super().__init__()
        ny, nx = grid_shape
        scale = hidden ** -.5
        self.y_weight = nn.Parameter(scale * torch.randn(hidden, hidden, ny // 2 + 1, 2))
        self.x_weight = nn.Parameter(scale * torch.randn(hidden, hidden, nx // 2 + 1, 2))
        self.mlp = nn.Sequential(nn.Linear(hidden, MLP_EXPANSION * hidden), nn.GELU(),
                                 nn.Linear(MLP_EXPANSION * hidden, hidden))

    def kernel(self, value):
        # [B,T,Y,X,H]; the population axis never enters a Fourier transform.
        ny, nx = value.shape[2:4]
        y_weight = torch.complex(self.y_weight[..., 0], self.y_weight[..., 1])
        x_weight = torch.complex(self.x_weight[..., 0], self.x_weight[..., 1])
        y_spectrum = torch.fft.rfft(value, dim=2, norm="ortho")
        y_mixed = torch.einsum("btkxi,oik->btkxo", y_spectrum, y_weight)
        x_spectrum = torch.fft.rfft(value, dim=3, norm="ortho")
        x_mixed = torch.einsum("btyki,oik->btyko", x_spectrum, x_weight)
        return (torch.fft.irfft(y_mixed, n=ny, dim=2, norm="ortho")
                + torch.fft.irfft(x_mixed, n=nx, dim=3, norm="ortho"))

    def forward(self, value):
        return value + self.mlp(self.kernel(value))


class MemoryNeuralOperator(nn.Module):
    """Nonnegative next-bin mean fields with explicit causal sequence state.

    Fields have shape [B,T,P,Y,X]; controls have shape [B,T,C]. A control
    at index t acts in the bin predicted from field t. Regular cell-center
    positional coordinates are dimensionless grid features, not soma positions.
    ``bin_seconds`` declares the transition clock. Internal learned S4 time
    constants have no mechanistic interpretation. Reset sequence state after
    loading or changing weights. This mean predictor has no joint sampler.
    """

    def __init__(self, populations, grid_shape, control_channels, bin_seconds):
        super().__init__()
        self.populations = _integer(populations, "populations", 1)
        self.control_channels = _integer(control_channels, "control_channels", 0)
        if not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 2:
            raise ValueError("grid_shape must contain exactly Y and X")
        self.grid_shape = tuple(_integer(size, "grid dimension", 1) for size in grid_shape)
        if type(bin_seconds) not in (int, float) or not math.isfinite(bin_seconds) or bin_seconds <= 0:
            raise ValueError("bin_seconds must be positive and finite")
        self.bin_seconds = float(bin_seconds)
        ny, nx = self.grid_shape
        y = (torch.arange(ny, dtype=torch.float32) + .5) / ny
        x = (torch.arange(nx, dtype=torch.float32) + .5) / nx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("grid_coordinates", torch.stack((yy, xx), dim=-1), persistent=False)
        self.lifting = nn.Linear(populations + control_channels + 2, HIDDEN_CHANNELS)
        self.spatial_layers = nn.ModuleList(
            FactorizedFourierLayer(HIDDEN_CHANNELS, self.grid_shape) for _ in range(4))
        self.memory = FFTConv(HIDDEN_CHANNELS, mode="diag", d_state=MEMORY_STATE_DIMENSION,
                              bidirectional=False, activation=None)
        # Positivity is an explicit count-rate observation adaptation. The
        # source PDE framework uses a linear decoder without this constraint.
        self.readout = nn.Linear(HIDDEN_CHANNELS, populations)

    def get_extra_state(self):
        return dict(implementation=IMPLEMENTATION, populations=self.populations,
                    grid_shape=self.grid_shape, control_channels=self.control_channels,
                    bin_seconds=self.bin_seconds, hidden_channels=HIDDEN_CHANNELS,
                    memory_state_dimension=MEMORY_STATE_DIMENSION)

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state != self.get_extra_state():
            raise ValueError("checkpoint model contract differs")

    def _validate(self, field, control):
        expected = (self.populations, *self.grid_shape)
        if field.ndim != 5 or tuple(field.shape[2:]) != expected or min(field.shape[:2]) < 1:
            raise ValueError("complete field [B,T,P,Y,X] required")
        if tuple(control.shape) != (*field.shape[:2], self.control_channels):
            raise ValueError("aligned control [B,T,C] required")
        for name, value in (("field", field), ("control", control)):
            if (not value.is_floating_point() or not torch.isfinite(value).all()
                    or value.dtype != self.lifting.weight.dtype
                    or value.device != self.lifting.weight.device):
                raise ValueError(f"finite {name} with model dtype and device required")
        if (field < 0).any():
            raise ValueError("nonnegative source rate fields required")

    def _features(self, field, control):
        batch, length = field.shape[:2]
        ny, nx = self.grid_shape
        features = [field.permute(0, 1, 3, 4, 2),
                    control[:, :, None, None, :].expand(batch, length, ny, nx, self.control_channels),
                    self.grid_coordinates.to(field).expand(batch, length, ny, nx, 2)]
        value = self.lifting(torch.cat(features, dim=-1))
        for layer in self.spatial_layers[:2]:
            value = layer(value)
        return value

    def _output(self, value):
        for layer in self.spatial_layers[2:]:
            value = layer(value)
        result = F.softplus(self.readout(value)).permute(0, 1, 4, 2, 3)
        if not torch.isfinite(result).all():
            raise FloatingPointError("nonfinite next-bin rates")
        return result

    def forward(self, field, control):
        self._validate(field, control)
        value = self._features(field, control)
        batch, length, ny, nx, hidden = value.shape
        sequence = value.permute(0, 2, 3, 4, 1).reshape(batch * ny * nx, hidden, length)
        output, _ = self.memory(sequence)
        output = output.reshape(batch, ny, nx, hidden, length).permute(0, 4, 1, 2, 3)
        return self._output(value + output)

    def reset_state(self, batchsize):
        batchsize = _integer(batchsize, "batchsize", 1)
        self.memory.setup_step()
        return self.memory.default_state(batchsize * self.grid_shape[0] * self.grid_shape[1])

    def one_step(self, field, control, state):
        if field.ndim != 4 or control.ndim != 2:
            raise ValueError("one_step requires field [B,P,Y,X] and control [B,C]")
        self._validate(field[:, None], control[:, None])
        batch = field.shape[0]
        ny, nx = self.grid_shape
        expected = (batch * ny * nx, HIDDEN_CHANNELS, MEMORY_STATE_DIMENSION // 2)
        if (not isinstance(state, torch.Tensor) or tuple(state.shape) != expected
                or not state.is_complex() or state.device != field.device
                or not torch.isfinite(state).all()):
            raise ValueError("finite matching S4D sequence state required")
        value = self._features(field[:, None], control[:, None])
        output, next_state = self.memory.step(value[:, 0].reshape(batch * ny * nx, HIDDEN_CHANNELS), state)
        output = output.reshape(batch, 1, ny, nx, HIDDEN_CHANNELS)
        return self._output(value + output)[:, 0], next_state
