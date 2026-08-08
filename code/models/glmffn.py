from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _make_tuple(value: int | Sequence[int], length: int) -> Tuple[int, ...]:
    if isinstance(value, int):
        return tuple([value] * length)
    value = tuple(value)
    if len(value) != length:
        raise ValueError(f"Expected {length} values, got {len(value)}.")
    return value



class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor() / keep_prob


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


class PatchMerging(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


def _selective_scan(
    u: Tensor,
    delta: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d: Tensor,
    delta_bias: Tensor,
) -> Tensor:
    u = u.float()
    delta = F.softplus(delta.float() + delta_bias.float().view(1, -1, 1))
    a = a.float()
    b = b.float()
    c = c.float()
    d = d.float()
    batch, total_channels, length = u.shape
    groups = b.shape[1]
    channels_per_group = total_channels // groups
    state_size = a.shape[-1]
    u = u.view(batch, groups, channels_per_group, length)
    delta = delta.view(batch, groups, channels_per_group, length)
    a = a.view(groups, channels_per_group, state_size)
    d = d.view(groups, channels_per_group)
    state = u.new_zeros(batch, groups, channels_per_group, state_size)
    outputs = []
    for index in range(length):
        u_t = u[..., index]
        delta_t = delta[..., index]
        transition = torch.exp(delta_t.unsqueeze(-1) * a.unsqueeze(0))
        state = (
            transition * state
            + delta_t.unsqueeze(-1)
            * b[..., index].unsqueeze(2)
            * u_t.unsqueeze(-1)
        )
        outputs.append(
            (state * c[..., index].unsqueeze(2)).sum(dim=-1)
            + d.unsqueeze(0) * u_t
        )
    return torch.stack(outputs, dim=-1).reshape(batch, total_channels, length)


class SelectiveScan2D(nn.Module):
    def __init__(
        self,
        channels: int,
        d_state: int = 1,
        ssm_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        d_conv: int = 3,
        dropout: float = 0.0,
        bias: bool = False,
        conv_bias: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        if d_state <= 0:
            raise ValueError(f"d_state must be positive, got {d_state}.")
        if ssm_ratio <= 0:
            raise ValueError(f"ssm_ratio must be positive, got {ssm_ratio}.")
        if not 0 < dt_min < dt_max:
            raise ValueError(
                f"Expected 0 < dt_min < dt_max, got {dt_min} and {dt_max}."
            )
        self.d_model = int(channels)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * channels)
        self.dt_rank = math.ceil(channels / 16) if dt_rank == "auto" else int(dt_rank)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.k_group = 4
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.act = nn.SiLU()
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=conv_bias,
        )
        projections = [
            nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
            )
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([projection.weight for projection in projections], dim=0)
        )
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        dt_weight = torch.empty(self.k_group, self.d_inner, self.dt_rank)
        if dt_init == "constant":
            nn.init.constant_(dt_weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"Unsupported dt_init: {dt_init}.")
        self.dt_projs_weight = nn.Parameter(dt_weight)
        dt = torch.exp(
            torch.rand(self.k_group, self.d_inner)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_projs_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_projs_bias._no_reinit = True
        a = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        a = a.view(1, -1).repeat(self.k_group * self.d_inner, 1)
        self.A_logs = nn.Parameter(torch.log(a))
        self.A_logs._no_weight_decay = True
        self.Ds = nn.Parameter(torch.ones(self.k_group * self.d_inner))
        self.Ds._no_weight_decay = True
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        batch, _, height, width = x.shape
        length = height * width
        x = x.permute(0, 2, 3, 1).contiguous()
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        gate = self.act(gate)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        horizontal_vertical = torch.stack(
            [x.flatten(2), x.transpose(2, 3).contiguous().flatten(2)],
            dim=1,
        )
        sequences = torch.cat(
            [horizontal_vertical, torch.flip(horizontal_vertical, dims=[-1])],
            dim=1,
        )
        projected = torch.einsum(
            "b k d l, k c d -> b k c l",
            sequences,
            self.x_proj_weight,
        )
        delta_low_rank, b, c = torch.split(
            projected,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2,
        )
        delta = torch.einsum(
            "b k r l, k d r -> b k d l",
            delta_low_rank,
            self.dt_projs_weight,
        )
        scan_output = _selective_scan(
            sequences.reshape(batch, -1, length).float(),
            delta.reshape(batch, -1, length).float(),
            -torch.exp(self.A_logs.float()),
            b.float().contiguous(),
            c.float().contiguous(),
            self.Ds.float(),
            self.dt_projs_bias.float().reshape(-1),
        ).view(batch, self.k_group, self.d_inner, length)
        inverse = torch.flip(scan_output[:, 2:4], dims=[-1])
        vertical = (
            scan_output[:, 1]
            .view(batch, self.d_inner, width, height)
            .transpose(2, 3)
            .contiguous()
            .view(batch, self.d_inner, length)
        )
        inverse_vertical = (
            inverse[:, 1]
            .view(batch, self.d_inner, width, height)
            .transpose(2, 3)
            .contiguous()
            .view(batch, self.d_inner, length)
        )
        merged = scan_output[:, 0] + inverse[:, 0] + vertical + inverse_vertical
        merged = self.out_norm(merged.transpose(1, 2).contiguous())
        merged = merged.view(batch, height, width, self.d_inner)
        merged = merged.to(gate.dtype) * gate
        output = self.dropout(self.out_proj(merged))
        return output.permute(0, 3, 1, 2).contiguous().to(input_dtype)


class VSSBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        d_state: int = 1,
        ssm_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        drop_path: float = 0.0,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.op = SelectiveScan2D(
            channels,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = LayerNorm2d(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.op(self.norm(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class VSSStage(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int = 1,
        d_state: int = 1,
        ssm_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                VSSBlock(
                    channels,
                    d_state=d_state,
                    ssm_ratio=ssm_ratio,
                    dt_rank=dt_rank,
                    dt_min=dt_min,
                    dt_max=dt_max,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class VSSBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (96, 192, 384, 768),
        depths: Sequence[int] = (1, 1, 1, 1),
        d_state: int = 1,
        ssm_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        channels = _make_tuple(channels, 4)
        depths = _make_tuple(depths, 4)
        self.out_channels = channels
        stage_options = {
            "d_state": d_state,
            "ssm_ratio": ssm_ratio,
            "dt_rank": dt_rank,
            "dt_min": dt_min,
            "dt_max": dt_max,
        }
        self.patch_embed = PatchEmbed(in_channels, channels[0], patch_size=4)
        self.stage1 = VSSStage(channels[0], depths[0], **stage_options)
        self.merge2 = PatchMerging(channels[0], channels[1])
        self.stage2 = VSSStage(channels[1], depths[1], **stage_options)
        self.merge3 = PatchMerging(channels[1], channels[2])
        self.stage3 = VSSStage(channels[2], depths[2], **stage_options)
        self.merge4 = PatchMerging(channels[2], channels[3])
        self.stage4 = VSSStage(channels[3], depths[3], **stage_options)

    def forward(self, x: Tensor) -> List[Tensor]:
        v1 = self.stage1(self.patch_embed(x))
        v2 = self.stage2(self.merge2(v1))
        v3 = self.stage3(self.merge3(v2))
        v4 = self.stage4(self.merge4(v3))
        return [v1, v2, v3, v4]

class ResNet34Encoder(nn.Module):
    out_channels = (64, 128, 256, 512)

    def __init__(self, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet34_Weights, resnet34
        except ImportError as exc:
            raise ImportError(
                "ResNet34Encoder requires torchvision. Install torchvision or "
                "provide an equivalent ResNet34 implementation."
            ) from exc

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)

        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    backbone.conv1.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return [s1, s2, s3, s4]


class CoordinateAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2).contiguous()
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [height, width], dim=2)
        y_w = y_w.permute(0, 1, 3, 2).contiguous()
        attn_h = torch.sigmoid(self.conv_h(y_h))
        attn_w = torch.sigmoid(self.conv_w(y_w))
        return x * attn_h * attn_w


class AdaPool2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        flat = x.flatten(2)

        emax_weights = torch.softmax(flat, dim=-1)
        emax_pool = torch.sum(flat * emax_weights, dim=-1, keepdim=True)

        mean = flat.mean(dim=-1, keepdim=True)
        dice = (2.0 * torch.abs(mean * flat) + self.eps) / (
            mean.pow(2) + flat.pow(2) + self.eps
        )
        edsc_weights = torch.softmax(dice, dim=-1)
        edsc_pool = torch.sum(flat * edsc_weights, dim=-1, keepdim=True)

        beta = torch.sigmoid(self.beta).flatten(2)
        return (beta * edsc_pool + (1.0 - beta) * emax_pool).unsqueeze(-1)


class AdaChannelPool2d(nn.Module):

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        emax_weights = torch.softmax(x, dim=1)
        emax_pool = torch.sum(x * emax_weights, dim=1, keepdim=True)

        mean = x.mean(dim=1, keepdim=True)
        dice = (2.0 * torch.abs(mean * x) + self.eps) / (
            mean.pow(2) + x.pow(2) + self.eps
        )
        edsc_weights = torch.softmax(dice, dim=1)
        edsc_pool = torch.sum(x * edsc_weights, dim=1, keepdim=True)

        beta = torch.sigmoid(self.beta)
        return beta * edsc_pool + (1.0 - beta) * emax_pool


class MultiScaleAdaptiveFusionModule(nn.Module):
    def __init__(
        self,
        aux_channels: int,
        main_channels: int,
        out_channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.aux_proj = ConvBNReLU(aux_channels, out_channels, kernel_size=1, padding=0)
        self.main_proj = ConvBNReLU(main_channels, out_channels, kernel_size=1, padding=0)
        self.aux_ca = CoordinateAttention(out_channels)
        self.main_ca = CoordinateAttention(out_channels)

        branch_channels = out_channels
        self.conv1 = ConvBNReLU(out_channels, branch_channels, kernel_size=1, padding=0)
        self.conv3 = ConvBNReLU(out_channels, branch_channels, kernel_size=3, dilation=1)
        self.dilated_d2 = ConvBNReLU(out_channels, branch_channels, kernel_size=3, dilation=2)
        self.dilated_d3 = ConvBNReLU(out_channels, branch_channels, kernel_size=3, dilation=3)
        self.gate = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, aux_feature: Tensor, main_feature: Tensor) -> Tensor:
        if aux_feature.shape[-2:] != main_feature.shape[-2:]:
            aux_feature = F.interpolate(
                aux_feature,
                size=main_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        aux = self.aux_proj(aux_feature)
        main = self.main_proj(main_feature)

        aux_att = self.aux_ca(aux)
        main_att = self.main_ca(main)
        mixed = aux + main

        gate = self.gate(
            torch.cat(
                [
                    self.conv1(mixed),
                    self.conv3(mixed),
                    self.dilated_d2(mixed),
                    self.dilated_d3(mixed),
                ],
                dim=1,
            )
        )
        return aux_att * (1.0 - gate) + main_att * gate


class MultiPoolingChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.ada_pool = AdaPool2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        ada = self.ada_pool(x)
        maxv = F.adaptive_max_pool2d(x, 1)
        return self.sigmoid(
            self.shared_mlp(avg) + self.shared_mlp(ada) + self.shared_mlp(maxv)
        )


class MultiPoolingSpatialAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.ada_pool = AdaChannelPool2d()
        self.attention = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False),
            nn.Conv2d(3, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        ada = self.ada_pool(x)
        maxv = torch.amax(x, dim=1, keepdim=True)
        return self.attention(torch.cat([avg, ada, maxv], dim=1))


class MultiPoolingChannelSpatialFeatureRefinementModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_attention = MultiPoolingChannelAttention(
            channels,
            reduction=reduction,
        )
        self.spatial_attention = MultiPoolingSpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        channel_refined = x + x * self.channel_attention(x)
        spatial_refined = channel_refined + channel_refined * self.spatial_attention(
            channel_refined
        )
        return spatial_refined


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_channels + skip_channels, out_channels, kernel_size=3),
            ConvBNReLU(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.conv(torch.cat([x, skip], dim=1))


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, kernel_size=3),
            ConvBNReLU(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: Tensor, output_size: Tuple[int, int]) -> Tensor:
        x = self.head(x)
        return F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class GLMFFN(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        resnet_pretrained: bool = True,
        vmamba_pretrained: bool = True,
        vmamba_model_name: str = "vmamba_tiny_s1l8",
        aux_channels: Sequence[int] = (96, 192, 384, 768),
        main_channels: Sequence[int] = (64, 128, 256, 512),
        fusion_channels: Sequence[int] = (96, 192, 384, 768),
        decoder_channels: Sequence[int] = (768, 384, 192),
        vss_depths: Sequence[int] = (1, 1, 1, 1),
        vss_d_state: int = 1,
        vss_ssm_ratio: float = 2.0,
        vss_dt_rank: int | str = "auto",
        vss_dt_min: float = 0.001,
        vss_dt_max: float = 0.1,
        attention_reduction: int = 16,
    ) -> None:
        super().__init__()
        aux_channels = _make_tuple(aux_channels, 4)
        main_channels = _make_tuple(main_channels, 4)
        fusion_channels = _make_tuple(fusion_channels, 4)
        decoder_channels = _make_tuple(decoder_channels, 3)
        vss_depths = _make_tuple(vss_depths, 4)

        self.main_encoder = ResNet34Encoder(
            in_channels=in_channels,
            pretrained=resnet_pretrained,
        )
        self.auxiliary_encoder = VSSBranch(
            in_channels=in_channels,
            channels=aux_channels,
            depths=vss_depths,
            d_state=vss_d_state,
            ssm_ratio=vss_ssm_ratio,
            dt_rank=vss_dt_rank,
            dt_min=vss_dt_min,
            dt_max=vss_dt_max,
        )
        aux_channels = tuple(self.auxiliary_encoder.out_channels)

        self.fusion_stage1 = MultiScaleAdaptiveFusionModule(
            aux_channels[0],
            main_channels[0],
            fusion_channels[0],
            reduction=attention_reduction,
        )
        self.fusion_stage2 = MultiScaleAdaptiveFusionModule(
            aux_channels[1],
            main_channels[1],
            fusion_channels[1],
            reduction=attention_reduction,
        )
        self.fusion_stage3 = MultiScaleAdaptiveFusionModule(
            aux_channels[2],
            main_channels[2],
            fusion_channels[2],
            reduction=attention_reduction,
        )
        self.fusion_stage4 = MultiScaleAdaptiveFusionModule(
            aux_channels[3],
            main_channels[3],
            fusion_channels[3],
            reduction=attention_reduction,
        )
        self.fusion_blocks = nn.ModuleList(
            [
                self.fusion_stage1,
                self.fusion_stage2,
                self.fusion_stage3,
                self.fusion_stage4,
            ]
        )

        self.refine_stage1 = MultiPoolingChannelSpatialFeatureRefinementModule(
            fusion_channels[0],
            reduction=attention_reduction,
        )
        self.refine_stage2 = MultiPoolingChannelSpatialFeatureRefinementModule(
            fusion_channels[1],
            reduction=attention_reduction,
        )
        self.refine_stage3 = MultiPoolingChannelSpatialFeatureRefinementModule(
            fusion_channels[2],
            reduction=attention_reduction,
        )

        self.decoder_stage3 = DecoderBlock(
            fusion_channels[3],
            fusion_channels[2],
            decoder_channels[0],
        )
        self.decoder_stage2 = DecoderBlock(
            decoder_channels[0],
            fusion_channels[1],
            decoder_channels[1],
        )
        self.decoder_stage1 = DecoderBlock(
            decoder_channels[1],
            fusion_channels[0],
            decoder_channels[2],
        )
        self.seg_head = SegmentationHead(decoder_channels[2], num_classes)
    def forward(self, x: Tensor, auxiliary_input: Optional[Tensor] = None) -> Tensor:
        output_size = x.shape[-2:]
        auxiliary_input = x if auxiliary_input is None else auxiliary_input

        main_features = self.main_encoder(x)
        aux_features = self.auxiliary_encoder(auxiliary_input)

        fused = [
            fusion(aux, main)
            for fusion, aux, main in zip(
                self.fusion_blocks,
                aux_features,
                main_features,
            )
        ]

        skip1 = self.refine_stage1(fused[0])
        skip2 = self.refine_stage2(fused[1])
        skip3 = self.refine_stage3(fused[2])
        bottleneck = fused[3]

        x = self.decoder_stage3(bottleneck, skip3)
        x = self.decoder_stage2(x, skip2)
        x = self.decoder_stage1(x, skip1)
        return self.seg_head(x, output_size)


def glmffn(
    num_classes: int = 3,
    in_channels: int = 3,
    **kwargs,
) -> GLMFFN:
    return GLMFFN(num_classes=num_classes, in_channels=in_channels, **kwargs)


if __name__ == "__main__":
    model = GLMFFN(num_classes=3, resnet_pretrained=False, vmamba_pretrained=False)
    image = torch.randn(1, 3, 512, 512)
    mask = model(image)
    print(mask.shape)








