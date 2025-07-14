import math
from typing import Optional, Union, Sequence

import jittor as jt
from jittor import nn
from jittor import init
from jittor.utils.pytorch_converter import convert
from jdet.utils.registry import BACKBONES

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# 实现autopad函数
def autopad(kernel_size: int, padding: int = None, dilation: int = 1):
    """自动计算填充值，保持输出尺寸不变"""
    assert kernel_size % 2 == 1, 'if use autopad, kernel size must be odd'
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1
    if padding is None:
        padding = kernel_size // 2
    return padding


# 实现make_divisible函数
def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """确保通道数能被divisor整除"""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # 确保下采样不会减少超过(1-min_ratio)的比例
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class BHWC2BCHW(nn.Module):
    # 维度转换 BHWC -> BCHW
    def execute(self, x):
        return x.permute(0, 3, 1, 2)


class BCHW2BHWC(nn.Module):
    # 维度转换 BCHW -> BHWC
    def execute(self, x):
        return x.permute(0, 2, 3, 1)


class GSiLU(nn.Module):
    """Global Sigmoid-Gated Linear Unit"""

    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def execute(self, x):
        return x * jt.sigmoid(self.adpool(x))


class CAA(nn.Module):
    """Context Anchor Attention"""

    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11
    ):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.03, eps=0.001)
        self.silu1 = nn.SiLU()

        self.h_conv = nn.Conv2d(channels, channels, (1, h_kernel_size), 1, (0, h_kernel_size // 2), groups=channels)
        self.v_conv = nn.Conv2d(channels, channels, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), groups=channels)

        self.conv2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.03, eps=0.001)
        self.act = nn.Sigmoid()

    def execute(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu1(x)

        x = self.h_conv(x)
        x = self.v_conv(x)

        x = self.conv2(x)
        x = self.bn2(x)
        return self.act(x)


class ConvFFN(nn.Module):
    """Multi-layer perceptron implemented with ConvModule"""

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            hidden_channels_scale: float = 4.0,
            hidden_kernel_size: int = 3,
            dropout_rate: float = 0.,
            add_identity: bool = True
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = int(in_channels * hidden_channels_scale)

        self.ffn_layers = nn.Sequential(
            BCHW2BHWC(),
            nn.LayerNorm(in_channels),
            BHWC2BCHW(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels, momentum=0.03, eps=0.001),
            nn.SiLU(),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=hidden_kernel_size,
                      stride=1, padding=hidden_kernel_size // 2, groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels, momentum=0.03, eps=0.001),
            GSiLU(),

            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001),
            nn.Dropout(dropout_rate),
        )
        self.add_identity = add_identity

    def execute(self, x):
        return x + self.ffn_layers(x) if self.add_identity else self.ffn_layers(x)


class Stem(nn.Module):
    """Stem layer"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion: float = 1.0
    ):
        super().__init__()
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels, momentum=0.03, eps=0.001),
            nn.SiLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels, momentum=0.03, eps=0.001),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001),
            nn.SiLU()
        )

    def execute(self, x):
        x = self.down_conv(x)
        x = self.conv1(x)
        return self.conv2(x)


class DownSamplingLayer(nn.Module):
    """Down sampling layer"""

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None
    ):
        super().__init__()
        out_channels = out_channels or (in_channels * 2)
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001),
            nn.SiLU()
        )

    def execute(self, x):
        return self.down_conv(x)


class InceptionBottleneck(nn.Module):
    """Bottleneck with Inception module"""

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 1.0,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0),
            nn.BatchNorm2d(hidden_channels, momentum=0.03, eps=0.001),
            nn.SiLU()
        )

        # 创建多个深度卷积层
        self.dw_convs = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            pad = autopad(k, None, d)
            conv = nn.Conv2d(hidden_channels, hidden_channels, k, 1, pad, d, groups=hidden_channels)
            self.dw_convs.append(conv)

        self.pw_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1, 1, 0),
            nn.BatchNorm2d(hidden_channels, momentum=0.03, eps=0.001),
            nn.SiLU()
        )

        self.with_caa = with_caa
        if with_caa:
            self.caa_factor = CAA(hidden_channels, caa_kernel_size, caa_kernel_size)
        else:
            self.caa_factor = None

        self.add_identity = add_identity and in_channels == out_channels

        self.post_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001),
            nn.SiLU()
        )

    def execute(self, x):
        x = self.pre_conv(x)
        y = x

        # 执行所有深度卷积并相加
        out = self.dw_convs[0](x)
        for conv in self.dw_convs[1:]:
            out += conv(x)

        x = self.pw_conv(out)

        if self.with_caa:
            y = self.caa_factor(y)

        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        return self.post_conv(x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    与 PyTorch 版本完全兼容的实现：
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py

    参数:
        drop_prob (float): 路径被置零的概率，默认为 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def execute(self, x: jt.Var) -> jt.Var:
        """执行 DropPath 操作

        参数:
            x (jt.Var): 输入张量

        返回:
            jt.Var: 应用 DropPath 后的输出张量
        """
        if self.drop_prob == 0. or not self.is_training():
            return x

        keep_prob = 1 - self.drop_prob
        # 处理不同维度的张量，不仅仅是 4D 张量
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        # 生成随机张量（0~1均匀分布）
        random_tensor = jt.rand(shape, dtype=x.dtype)

        # 计算保留概率并二值化
        random_tensor = (random_tensor + keep_prob).floor()

        # 应用缩放和掩码
        if keep_prob > 0:
            output = x / keep_prob * random_tensor
        else:
            output = x * random_tensor

        return output


class PKIBlock(nn.Module):
    """Poly Kernel Inception Block"""

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            expansion: float = 1.0,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale: Optional[float] = 1.0,
            add_identity: bool = True
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.norm1 = nn.BatchNorm2d(in_channels, momentum=0.03, eps=0.001)
        self.norm2 = nn.BatchNorm2d(hidden_channels, momentum=0.03, eps=0.001)

        self.block = InceptionBottleneck(
            in_channels, hidden_channels, kernel_sizes, dilations,
            1.0, True, with_caa, caa_kernel_size
        )
        self.ffn = ConvFFN(hidden_channels, out_channels, ffn_scale,
                           ffn_kernel_size, dropout_rate, add_identity=False)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma1 = init.constant((hidden_channels,), "float32", layer_scale)
            self.gamma2 = init.constant((out_channels,), "float32", layer_scale)
        self.add_identity = add_identity and in_channels == out_channels

    def execute(self, x):
        identity = x
        x = self.norm1(x)
        x = self.block(x)

        if self.layer_scale:
            x = x * self.gamma1.view(1, -1, 1, 1)

        if self.add_identity:
            x = identity + self.drop_path(x)

        identity = x
        x = self.norm2(x)
        x = self.ffn(x)

        if self.layer_scale:
            x = x * self.gamma2.view(1, -1, 1, 1)

        if self.add_identity:
            x = identity + self.drop_path(x)

        return x


class PKIStage(nn.Module):
    """Poly Kernel Inception Stage"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 0.5,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: Union[float, list] = 0.,
            layer_scale: Optional[float] = 1.0,
            shortcut_with_ffn: bool = True,
            shortcut_ffn_scale: float = 4.0,
            shortcut_ffn_kernel_size: int = 5,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11
    ):
        super().__init__()
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.downsample = DownSamplingLayer(in_channels, out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, 2 * hidden_channels, 1, 1, 0),
            nn.BatchNorm2d(2 * hidden_channels, momentum=0.03, eps=0.001),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * hidden_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001),
            nn.SiLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001),
            nn.SiLU()
        )

        if shortcut_with_ffn:
            self.ffn = ConvFFN(
                hidden_channels, hidden_channels,
                shortcut_ffn_scale, shortcut_ffn_kernel_size, 0.
            )
        else:
            self.ffn = None

        # 创建多个PKIBlock
        drop_path_rates = drop_path_rate if isinstance(drop_path_rate, list) else [drop_path_rate] * num_blocks
        self.blocks = nn.ModuleList([
            PKIBlock(
                hidden_channels, hidden_channels, kernel_sizes, dilations, with_caa,
                caa_kernel_size + 2 * i, 1.0, ffn_scale, ffn_kernel_size, dropout_rate,
                drop_path_rates[i], layer_scale, add_identity
            ) for i in range(num_blocks)
        ])

    def execute(self, x):
        x = self.downsample(x)
        x = self.conv1(x)

        # 分割特征图
        x, y = jt.chunk(x, 2, dim=1)

        if self.ffn is not None:
            x = self.ffn(x)

        # 处理y分支
        t = jt.zeros_like(y)
        for block in self.blocks:
            t = t + block(y)

        # 合并特征
        z = jt.concat([x, t], dim=1)
        z = self.conv2(z)
        z = self.conv3(z)

        return z


class PKINet(nn.Module):
    """Poly Kernel Inception Network"""
    arch_settings = {
        'T': [[16, 32, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 5, True, True, 11],
              [32, 64, 14, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 7, True, True, 11],
              [64, 128, 22, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 9, True, True, 11],
              [128, 256, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 11, True, True, 11]],

        'S': [[32, 64, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 5, True, True, 11],
              [64, 128, 12, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 7, True, True, 11],
              [128, 256, 20, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 9, True, True, 11],
              [256, 512, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 11, True, True, 11]],

        'B': [[40, 80, 6, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 5, True, True, 11],
              [80, 160, 16, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 7, True, True, 11],
              [160, 320, 24, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 9, True, True, 11],
              [320, 640, 6, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 11, True, True, 11]],
    }

    def __init__(
            self,
            arch: str = 'S',
            img_size: int = 224,
            num_stages: int = 4,
            out_indices: Sequence[int] = (2, 3, 4),
            drop_path_rate: float = 0.1,
            frozen_stages: int = -1,
            norm_eval: bool = False,
            arch_setting: Optional[Sequence[list]] = None
    ):
        super().__init__()
        arch_setting = arch_setting or self.arch_settings[arch]
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # 创建网络阶段
        self.stages = nn.ModuleList()
        self.stem = Stem(3, arch_setting[0][0])
        self.stages.append(self.stem)

        # 计算drop path概率
        depths = [x[2] for x in arch_setting]
        total_blocks = sum(depths)
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, total_blocks)]

        # 构建各阶段
        start_idx = 0
        for i, settings in enumerate(arch_setting):
            in_ch, out_ch, num_blocks, kernel_sizes, dilations, expansion, \
                ffn_scale, ffn_kernel, dropout, layer_scale, shortcut_ffn, \
                shortcut_ffn_scale, shortcut_ffn_kernel, add_identity, with_caa, caa_kernel = settings

            stage_dpr = dpr[start_idx:start_idx + num_blocks]
            start_idx += num_blocks

            stage = PKIStage(
                in_ch, out_ch, num_blocks, kernel_sizes, dilations, expansion,
                ffn_scale, ffn_kernel, dropout, stage_dpr, layer_scale, shortcut_ffn,
                shortcut_ffn_scale, shortcut_ffn_kernel, add_identity, with_caa, caa_kernel
            )
            self.stages.append(stage)

    def execute(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def init_weights(self):
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.gauss_(m.weight, 0, 0.02)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # Kaiming初始化
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                init.gauss_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def train(self):
        super().train()
        self._freeze_stages()
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):  # +1 包含stem
                m = self.stages[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


@BACKBONES.register_module()
def PKINet_s(pretrained=False, **kwargs):
    model = PKINet(
        arch='S',
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        model.load(pretrained)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 10, 'input_size': (3, 1024, 1024), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }