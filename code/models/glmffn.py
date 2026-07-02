from __future__ import annotations

from dataclasses import dataclass
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



def _to_nchw(feature: Tensor, expected_channels: int) -> Tensor:
    if feature.ndim != 4:
        raise ValueError(f"Expected a 4D feature map, got shape {tuple(feature.shape)}.")
    if feature.shape[1] == expected_channels:
        return feature
    if feature.shape[-1] == expected_channels:
        return feature.permute(0, 3, 1, 2).contiguous()
    return feature


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


class SelectiveScan2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.merge = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        self.gain = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.25)

    @staticmethod
    def _scan(x: Tensor, dim: int, reverse: bool = False) -> Tensor:
        if reverse:
            x = torch.flip(x, dims=(dim,))
        out = torch.cumsum(x, dim=dim)
        denom = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype)
        shape = [1] * x.ndim
        shape[dim] = -1
        out = out / denom.view(*shape)
        if reverse:
            out = torch.flip(out, dims=(dim,))
        return out

    def forward(self, x: Tensor) -> Tensor:
        left_to_right = self._scan(x, dim=3, reverse=False)
        right_to_left = self._scan(x, dim=3, reverse=True)
        top_to_bottom = self._scan(x, dim=2, reverse=False)
        bottom_to_top = self._scan(x, dim=2, reverse=True)
        return self.merge(
            torch.cat([left_to_right, right_to_left, top_to_bottom, bottom_to_top], dim=1)
        ) * self.gain


class VSSBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.ss2d = SelectiveScan2D(channels)
        self.norm2 = LayerNorm2d(channels)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        feature, gate = self.in_proj(self.norm1(x)).chunk(2, dim=1)
        feature = F.silu(self.dwconv(feature))
        feature = self.norm2(self.ss2d(feature))
        x = shortcut + self.out_proj(feature * F.silu(gate))
        return x + self.mlp(x)


class VSSStage(nn.Module):
    def __init__(self, channels: int, depth: int = 1) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[VSSBlock(channels) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class VSSBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (96, 192, 384, 768),
        depths: Sequence[int] = (1, 1, 1, 1),
    ) -> None:
        super().__init__()
        channels = _make_tuple(channels, 4)
        depths = _make_tuple(depths, 4)
        self.out_channels = channels

        self.patch_embed = PatchEmbed(in_channels, channels[0], patch_size=4)
        self.stage1 = VSSStage(channels[0], depths[0])
        self.merge2 = PatchMerging(channels[0], channels[1])
        self.stage2 = VSSStage(channels[1], depths[1])
        self.merge3 = PatchMerging(channels[1], channels[2])
        self.stage3 = VSSStage(channels[2], depths[2])
        self.merge4 = PatchMerging(channels[2], channels[3])
        self.stage4 = VSSStage(channels[3], depths[3])

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


@dataclass(frozen=True)
class VMambaConfig:
    model_name: str = "vmamba_tiny_s1l8"
    pretrained: bool = True
    out_indices: Tuple[int, int, int, int] = (0, 1, 2, 3)
    fallback_model_names: Tuple[str, ...] = (
        "vmamba_tiny_s1l8",
        "vmamba_tiny",
        "vssm_tiny",
        "vmambav2_tiny",
    )


class ExternalVMambaTinyEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        config: VMambaConfig = VMambaConfig(),
        expected_channels: Sequence[int] = (96, 192, 384, 768),
    ) -> None:
        super().__init__()
        self.config = config
        self.expected_channels = tuple(expected_channels)

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "ExternalVMambaTinyEncoder requires a VMamba-capable external "
                "package. Install a timm build that includes VMamba/VSS models, "
                "then keep VSSBlock inside that package instead of reimplementing "
                "it in this repository."
            ) from exc

        tried: List[str] = []
        model_names = (config.model_name,) + tuple(
            name for name in config.fallback_model_names if name != config.model_name
        )
        last_error: Optional[Exception] = None
        backbone = None

        for name in model_names:
            tried.append(name)
            try:
                backbone = timm.create_model(
                    name,
                    pretrained=config.pretrained,
                    features_only=True,
                    out_indices=config.out_indices,
                    in_chans=in_channels,
                )
                self.model_name = name
                break
            except Exception as exc:
                last_error = exc

        if backbone is None:
            raise RuntimeError(
                "Could not create a VMamba-tiny feature extractor from timm. "
                f"Tried: {', '.join(tried)}. Install/register VMamba-tiny in "
                "timm, or pass the correct timm model name through "
                "VMambaConfig(model_name=...)."
            ) from last_error

        self.backbone = backbone
        channels = None
        if hasattr(backbone, "feature_info"):
            try:
                channels = tuple(backbone.feature_info.channels())
            except Exception:
                channels = None
        self.out_channels = channels or self.expected_channels

    def forward(self, x: Tensor) -> List[Tensor]:
        features = list(self.backbone(x))
        if len(features) < 4:
            raise RuntimeError(
                "VMamba-tiny feature extractor must return four stage features."
            )
        selected = features[:4]
        return [
            _to_nchw(feat, channels)
            for feat, channels in zip(selected, self.out_channels)
        ]


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        weight = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * self.sigmoid(weight)


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
        self.aux_ca = ChannelAttention(out_channels, reduction=reduction)
        self.main_ca = ChannelAttention(out_channels, reduction=reduction)

        branch_channels = out_channels
        self.dilated_d1 = ConvBNReLU(out_channels, branch_channels, kernel_size=3, dilation=1)
        self.dilated_d2 = ConvBNReLU(out_channels, branch_channels, kernel_size=3, dilation=2)
        self.dilated_d3 = ConvBNReLU(out_channels, branch_channels, kernel_size=3, dilation=3)
        self.dilated_d5 = ConvBNReLU(out_channels, branch_channels, kernel_size=3, dilation=5)
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
                    self.dilated_d1(mixed),
                    self.dilated_d2(mixed),
                    self.dilated_d3(mixed),
                    self.dilated_d5(mixed),
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
        use_external_vmamba: bool = False,
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
        if use_external_vmamba:
            self.auxiliary_encoder = ExternalVMambaTinyEncoder(
                in_channels=in_channels,
                config=VMambaConfig(
                    model_name=vmamba_model_name,
                    pretrained=vmamba_pretrained,
                ),
                expected_channels=aux_channels,
            )
        else:
            self.auxiliary_encoder = VSSBranch(
                in_channels=in_channels,
                channels=aux_channels,
                depths=vss_depths,
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





