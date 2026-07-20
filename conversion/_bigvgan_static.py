"""Make BigVGAN's alias-free resampling exportable.

UpSample1d and LowPassFilter1d build their depthwise convolution kernel at call time with
``self.filter.expand(C, -1, -1)``, where ``C`` comes from the input tensor. The tracer
therefore sees a kernel of unknown shape and refuses the convolution.

``C`` is fixed for a given layer, so a single calibration pass records it and the expanded
kernel is materialised as a buffer. The arithmetic is unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticUpSample1d(nn.Module):
    def __init__(self, src, channels: int):
        super().__init__()
        self.ratio, self.stride = src.ratio, src.stride
        self.pad, self.pad_left, self.pad_right = src.pad, src.pad_left, src.pad_right
        self.register_buffer("kernel", src.filter.expand(channels, -1, -1).contiguous())
        self.groups = channels

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.kernel, stride=self.stride,
                                            groups=self.groups)
        return x[..., self.pad_left:-self.pad_right]


class StaticLowPassFilter1d(nn.Module):
    def __init__(self, src, channels: int):
        super().__init__()
        self.pad_left, self.pad_right = src.pad_left, src.pad_right
        self.stride, self.padding, self.padding_mode = src.stride, src.padding, src.padding_mode
        self.register_buffer("kernel", src.filter.expand(channels, -1, -1).contiguous())
        self.groups = channels

    def forward(self, x):
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(x, self.kernel, stride=self.stride, groups=self.groups)


def make_static(model, example_input):
    """Record each resampler's channel count, then swap in fixed-kernel equivalents."""
    from bigvgan.alias_free_torch.filter import LowPassFilter1d
    from bigvgan.alias_free_torch.resample import UpSample1d

    seen = {}
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, (UpSample1d, LowPassFilter1d)):
            handles.append(mod.register_forward_pre_hook(
                lambda m, inp, n=name: seen.__setitem__(n, inp[0].shape[1])))
    with torch.no_grad():
        model(example_input)
    for h in handles:
        h.remove()

    replaced = 0
    for name, channels in seen.items():
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        mod = getattr(parent, parts[-1])
        new = (StaticUpSample1d(mod, channels) if isinstance(mod, UpSample1d)
               else StaticLowPassFilter1d(mod, channels))
        setattr(parent, parts[-1], new.eval())
        replaced += 1
    return replaced
