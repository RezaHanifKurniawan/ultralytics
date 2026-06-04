# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
    "SEBlock",
    "CBAM",
    "GAM",
    "CoordAtt",
    "SimAM",
    "EMA",
    "BiFormer",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply convolution transpose and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, H, W) and output shape is (B, c2, H/2, W/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel Attention Module for CBAM.

    Aggregates spatial context via global adaptive max and average pooling to dynamically
    re-weight structural feature channels using a shared multi-layer perceptron.
    """

    def __init__(self, c1, reduction=16):
        """Initialize Channel Attention sub-module."""
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)                                 # Kompresi fitur spasial ke bentuk rata-rata global 1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)                                 # Kompresi fitur spasial ke bentuk maksimum global 1x1

        mid = max(8, c1 // reduction)
        self.mlp = nn.Sequential(
            nn.Flatten(1),                                                      # Konversi format matriks 4D ke bentuk vektor 2D linear
            nn.Linear(c1, mid, bias=False),                                     # Lapisan kompresi dimensi internal saluran (Bottleneck)
            nn.ReLU(inplace=True),                                              # Lapisan aktivasi non-linear internal jaringan
            nn.Linear(mid, c1, bias=False)                                      # Lapisan ekspansi kembali ke jumlah saluran semula
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Compute and multiply channel-wise attention weights using shared MLP."""
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x))                                    # Pemrosesan vektor pooling rata-rata lewat MLP linear
        max_out = self.mlp(self.max_pool(x))                                    # Pemrosesan vektor pooling maksimum lewat MLP linear
        channel_attention = self.act(avg_out + max_out).view(b, c, 1, 1)        # Penggabungan nilai respon ke format dimensi matriks 4D
        return x * channel_attention                                            # Multiplikasi elemen bobot dengan fitur input utama


class SpatialAttention(nn.Module):
    """Spatial Attention Module for CBAM.

    Highlights target object boundaries and handles overlapping regions by
    projecting inter-channel pooling statistics onto a spatial coordinate matrix.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial Attention sub-module."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = kernel_size // 2
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False) # Konvolusi penentu peta spasial koordinat 2D besar
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Compute and multiply positionally-guided spatial attention weights."""
        avg_spatial = torch.mean(x, dim=1, keepdim=True)                        # Operasi rata-rata nilai lintas dimensi saluran
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)                      # Operasi nilai maksimum murni lintas dimensi saluran
        y = torch.cat([avg_spatial, max_spatial], dim=1)                        # Penggabungan fitur deskriptif statistik ruang 2D
        spatial_attention = self.act(self.spatial_conv(y))                      # Transformasi konvolusi menjadi bobot spasial
        return x * spatial_attention                                            # Multiplikasi bobot spasial ke matriks fitur utama


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) Block.

    Recalibrates channel-wise feature responses using global context aggregation.

    Attributes:
        avg_pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        fc (nn.Sequential): Two-layer bottleneck MLP implemented as linear layers.
        act (nn.Sigmoid): Sigmoid activation for channel weights.

    References:
        Hu et al., 2018 (https://ieeexplore.ieee.org/document/8578843)
    """

    def __init__(self, c1, reduction=16):
        """Initialize Squeeze-and-Excitation module.

        Args:
            c1 (int): Number of input channels.
            reduction (int): Channel reduction ratio. Defaults to 16.
        """
        super().__init__()
        mid = max(8, c1 // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)                                 # Kompresi fitur spasial ke bentuk global 1x1
        self.fc = nn.Sequential(
            nn.Flatten(1),                                                      # Meratakan matriks 4D ke 2D untuk input linear
            nn.Linear(c1, mid, bias=False),                                     # Lapisan reduksi dimensi saluran
            nn.ReLU(inplace=True),                                              # Fungsi aktivasi non-linear pertama
            nn.Linear(mid, c1, bias=False)                                      # Lapisan ekspansi kembali dimensi saluran
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel-wise feature recalibration to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        identity = x
        b, c, _, _ = x.size()
        
        w = self.avg_pool(x)                                                    # Operasi agregasi fitur global (Squeeze)
        w = self.fc(w)                                                          # Operasi perhitungan interdependensi (Excitation)
        w = self.act(w).view(b, c, 1, 1)                                        # Rekonstruksi dimensi bobot ke format 4D asli
        
        return identity * w                                                     # Perkalian elemen bobot dengan matriks fitur utama


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    Sequentially applies channel attention and spatial attention for adaptive feature refinement.

    Attributes:
        channel_attention (nn.Module): Channel attention submodule with shared MLP.
        spatial_attention (nn.Module): Spatial attention submodule with large receptive field.

    References:
        Woo et al., 2018 (https://link.springer.com/chapter/10.1007/978-3-030-01234-2_1)
    """

    def __init__(self, c1, reduction=16, kernel_size=7):
        """Initialize Convolutional Block Attention Module.

        Args:
            c1 (int): Number of input channels.
            reduction (int): Channel reduction ratio for shared MLP. Defaults to 16.
            kernel_size (int): Size of the convolutional kernel for spatial attention. Defaults to 7.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1, reduction)                # Inisialisasi komponen atensi dimensi saluran
        self.spatial_attention = SpatialAttention(kernel_size)                  # Inisialisasi komponen atensi dimensi ruang/spasial

    def forward(self, x):
        """Apply sequential channel and spatial attention refinement to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Sequentially refined output tensor.
        """
        out = self.channel_attention(x)                                         # Tahap pertama: Ekstraksi fitur bobot saluran
        out = self.spatial_attention(out)                                       # Tahap kedua: Ekstraksi fitur bobot ruang/spasial
        return out


class GAM(nn.Module):
    """Global Attention Mechanism (GAM).

    Preserves global cross-dimension interactions without spatial information loss
    by sequentially optimizing channel and spatial submodules with permutation.

    Attributes:
        channel_attention (nn.Sequential): Three-dimensional permutation and full-linear MLP.
        spatial_attention (nn.Sequential): Dual-layer non-pooling convolution module.
        act (nn.Sigmoid): Sigmoid activation for feature masking.

    References:
        Liu et al., 2021 (https://arxiv.org/abs/2112.05561)
    """

    def __init__(self, c1, reduction=16, kernel_size=7):
        """Initialize Global Attention Mechanism module.

        Args:
            c1 (int): Number of input channels.
            reduction (int): Channel reduction ratio for shared MLP. Defaults to 16.
            kernel_size (int): Receptive field size for spatial adaptation. Defaults to 7.
        """
        super().__init__()
        mid = max(8, c1 // reduction)

        self.channel_attention = nn.Sequential(
            nn.Linear(c1, mid),                                                 # Lapisan kompresi dimensi internal saluran
            nn.ReLU(inplace=True),                                              # Fungsi aktivasi non-linear internal pertama
            nn.Linear(mid, c1)                                                  # Lapisan ekspansi kembali ke dimensi semula
        )
        
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = kernel_size // 2
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1, mid, kernel_size, padding=padding, bias=False),      # Reduksi dimensi saluran pada domain spasial
            nn.BatchNorm2d(mid),                                                # Normalisasi batch lapisan internal pertama
            nn.ReLU(inplace=True),                                              # Fungsi aktivasi non-linear internal kedua
            nn.Conv2d(mid, c1, kernel_size, padding=padding, bias=False),      # Ekspansi kembali saluran pada domain spasial
            nn.BatchNorm2d(c1)                                                  # Normalisasi batch lapisan internal kedua
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply global channel-spatial attention interactions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Globally-attended output tensor.
        """
        b, c, h, w = x.shape
        
        x_permute = x.permute(0, 2, 3, 1).reshape(b, -1, c)                    # Mengubah urutan matriks 4D ke bentuk sekuensial 3D [B, H*W, C]
        c_att = self.channel_attention(x_permute)                              # Penghitungan interdependensi saluran melalui MLP linear
        c_att = c_att.reshape(b, h, w, c).permute(0, 3, 1, 2)                  # Mengembalikan struktur ke format dimensi asli [B, C, H, W]
        x_c = x * self.act(c_att)                                               # Multiplikasi elemen bobot dengan fitur input utama

        spatial_attention = self.act(self.spatial_attention(x_c))               # Transformasi konvolusional ganda menjadi bobot spasial

        return x_c * spatial_attention                                          # Multiplikasi bobot spasial ke matriks fitur saluran


class CoordAtt(nn.Module):
    """Coordinate Attention Module.

    Embeds positional coordinate signals into channel attention maps using directional
    adaptive average pooling layers to capture long-range spatial dependencies.

    Attributes:
        pool_h (nn.AdaptiveAvgPool2d): Directional pooling along the vertical axis.
        pool_w (nn.AdaptiveAvgPool2d): Directional pooling along the horizontal axis.
        conv1 (nn.Conv2d): Shared transformation layer for unified feature mapping.
        bn1 (nn.BatchNorm2d): Normalization layer for joint horizontal-vertical features.
        act (nn.Hardswish): Non-linear activation for unified spatial mappings.
        conv_h (nn.Conv2d): Directional transformation to isolate vertical channel weights.
        conv_w (nn.Conv2d): Directional transformation to isolate horizontal channel weights.

    References:
        Hou et al., 2021 (https://arxiv.org/abs/2103.02907)
    """

    def __init__(self, c1, reduction=16):
        """Initialize Coordinate Attention module.

        Args:
            c1 (int): Number of input channels.
            reduction (int): Downsampling ratio for bottleneck squeeze. Defaults to 16.
        """
        super().__init__()
        mip = max(8, c1 // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))                           # Pooling adaptif untuk melacak fitur sumbu vertikal (Y)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))                           # Pooling adaptif untuk melacak fitur sumbu horizontal (X)

        self.conv1 = nn.Conv2d(c1, mip, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)

        self.conv_h = nn.Conv2d(mip, c1, 1, 1, 0, bias=False)
        self.conv_w = nn.Conv2d(mip, c1, 1, 1, 0, bias=False)

    def forward(self, x):
        """Apply positionally-guided directional attention weights to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Coordinate-attended output tensor.
        """
        identity = x
        b, c, h, w = x.size()

        x_h = self.pool_h(x)                                                    # Kompresi fitur gambar ke lajur vertikal [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)                                # Kompresi lajur horizontal dan reposisi tensor [B, C, W, 1]

        y = self.conv1(torch.cat([x_h, x_w], dim=2))                            # Penggabungan dua koordinat aksial pada dimensi tinggi
        y = self.act(self.bn1(y))                                               # Normalisasi batch dan aktivasi non-linear gabungan fitur

        x_h, x_w = torch.split(y, [h, w], dim=2)                                # Pemisahan kembali fitur gabungan berdasarkan nilai asli H dan W
        x_w = x_w.permute(0, 1, 3, 2)                                           # Mengembalikan orientasi memori tensor horizontal ke bentuk semula

        a_h = torch.sigmoid(self.conv_h(x_h))                                   # Perhitungan matriks probabilitas atensi vertikal (Y)
        a_w = torch.sigmoid(self.conv_w(x_w))                                   # Perhitungan matriks probabilitas atensi horizontal (X)

        return identity * a_h * a_w                                             # Multiplikasi interaksi dua sumbu koordinat ke fitur input
    

class SimAM(nn.Module):
    """Simple Attention Module (SimAM).
    
    A parameter-free attention module that infers 3D weights based on energy functions.
    
    References:
        Yang et al., 2021 (https://proceedings.mlr.press/v139/yang21o.html)
    """

    def __init__(self, e_lambda=1e-4):
        """Initialize SimAM module.
        
        Args:
            e_lambda (float): Hyperparameter for numerical stability. Defaults to 1e-4.
        """
        super().__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        """Apply energy-based attention weights to input tensor."""
        b, c, h, w = x.size()
        
        mu = x.mean(dim=[2, 3], keepdim=True)                                   # Menghitung nilai rata-rata spasial tiap saluran
        x_minus_mu_sq = (x - mu).pow(2)                                         # Menghitung nilai kuadrat selisih fitur terhadap rata-rata
        
        var = x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / (h * w)             # Menghitung nilai variansi spasial populasi murni
        y = x_minus_mu_sq / (4 * (var + self.e_lambda)) + 0.5                   # Penghitungan fungsi energi minimal tiap piksel 3D
        
        return x * self.act(y)                                                  # Multiplikasi bobot fungsi energi ke fitur input utama


class EMA(nn.Module):
    """Efficient Multi-Scale Attention (EMA).
    
    Uses multi-scale parallel structures and cross-spatial learning for boundary-aware attention.
    
    Attributes:
        groups (int): Number of feature groups.
        pool_h (nn.AdaptiveAvgPool2d): Directional adaptive pooling along the vertical axis.
        pool_w (nn.AdaptiveAvgPool2d): Directional adaptive pooling along the horizontal axis.
        conv1x1 (nn.Conv2d): Dimensionality transformation for pooled features.
        conv3x3 (nn.Conv2d): Spatial feature extraction for parallel branch.
        softmax (nn.Softmax): Normalization layer for 2D spatial attention maps.
    
    References:
        Ouyang et al., 2023 (https://arxiv.org/abs/2305.13563)
    """

    def __init__(self, c1, groups=8):
        """Initialize EMA module.
        
        Args:
            c1 (int): Number of input channels.
            groups (int): Grouping factor for parallel processing. Defaults to 8.
        """
        super().__init__()
        self.groups = groups
        self.softmax = nn.Softmax(dim=-1)
        
        group_ch = c1 // groups

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))                           # Kompresi adaptif untuk melacak koordinat sumbu vertikal
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))                           # Kompresi adaptif untuk melacak koordinat sumbu horizontal

        self.conv1x1 = nn.Conv2d(group_ch, group_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(group_ch, group_ch, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        """Apply multi-scale parallel attention weights to input tensor with groups=8."""
        b, c, h, w = x.size()
        group_ch = c // self.groups

        x_g = x.view(b * self.groups, group_ch, h, w)                           # Pembagian dimensi saluran ke dalam sub-grup paralel

        x_h = self.pool_h(x_g)                                                  # Kompresi fitur sub-grup ke lajur vertikal rahang
        x_w = self.pool_w(x_g).permute(0, 1, 3, 2)                              # Kompresi lajur horizontal dan reposisi dimensi matriks

        y = self.conv1x1(torch.cat([x_h, x_w], dim=2))                          # Penggabungan koordinat aksial lewat konvolusi 1x1
        y_h, y_w = torch.split(y, [h, w], dim=2)                                # Pemisahan kembali fitur koordinat gabungan aksial
        y_w = y_w.permute(0, 1, 3, 2)                                           # Mengembalikan orientasi memori tensor horizontal

        a_h = torch.sigmoid(y_h)                                                # Perhitungan matriks probabilitas atensi vertikal
        a_w = torch.sigmoid(y_w)                                                # Perhitungan matriks probabilitas atensi horizontal

        y_local = self.conv3x3(x_g)                                             # Ekstraksi variansi struktural mikro lewat konvolusi 3x3

        g_h = torch.mean(y_local * a_h, dim=3, keepdim=True).flatten(2)         # Agregasi fitur vertikal linear ukuran [B*G, C_g, H]
        g_w = torch.mean(y_local * a_w, dim=2, keepdim=True).flatten(2)         # Agregasi fitur horizontal linear ukuran [B*G, C_g, W]
        
        attn = self.softmax(torch.matmul(g_h.transpose(-1, -2), g_w))           # Perkalian spasial menghasilkan peta atensi [B*G, H, W]
        
        y_local_permute = y_local.permute(0, 2, 3, 1).view(b * self.groups, h * w, group_ch) # Isolasi channel grup ke posisi belakang matriks
        attn_reshape = attn.view(b * self.groups, h, w).flatten(1).unsqueeze(-1)             # Perataan dimensi spasial peta atensi gambar
        
        out = y_local_permute * attn_reshape.view(b * self.groups, h * w, 1)     # Multiplikasi bobot spasial dengan channel tetap utuh
        out = out.view(b * self.groups, h, w, group_ch).permute(0, 3, 1, 2)     # Pengembalian bentuk tensor ke format 4D sub-grup asli

        out = torch.sigmoid(out) * x_g                                          # Multiplikasi elemen pas antara out dan x_g [B*G, 64, H, W]

        return out.view(b, c, h, w)                                             # Mengembalikan bentuk tensor ke format 4D semula
    

class BiFormer(nn.Module):
    """Bi-Level Routing Attention (BRA).

    Preserves global cross-dimension interactions without spatial information loss
    by dynamically routing attention queries to the most relevant regional keys.

    Attributes:
        c2 (int): Number of output channels matched to the input tensor.
        n_win (int): Number of windows for local block partitioning.
        num_heads (int): Number of multi-head self-attention paths.
        topk (int): Number of top-k regions selected by the dynamic router.

    References:
        Zhu et al., 2023 (https://arxiv.org/abs/2303.08810)
    """

    def __init__(self, c1, c2=None, n_win=7, num_heads=8, topk=4):
        """Initialize Bi-Level Routing Attention module."""
        super().__init__()
        self.c2 = c2 if c2 is not None else c1
        self.n_win = n_win
        self.num_heads = num_heads
        self.topk = topk
        
        self.proj_in = nn.Conv2d(c1, self.c2, kernel_size=1) if c1 != self.c2 else nn.Identity()
        self.qkv = nn.Conv2d(self.c2, self.c2 * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(self.c2, self.c2, kernel_size=1)
        
        self.router_conv = nn.Sequential(
            nn.Conv2d(self.c2, self.c2, kernel_size=3, padding=1, groups=self.c2),
            nn.GELU(),
            nn.Conv2d(self.c2, self.c2, kernel_size=1)
        )
        
    def forward(self, x):
        """Apply dynamic sparse routing attention across the spatial domain."""
        x = self.proj_in(x)                                                     # Penyesuaian dimensi awal saluran jika ada perbedaan
        B, C, H, W = x.shape
        
        qkv = self.qkv(x)                                                       # Transformasi fitur linear untuk membangkitkan QKV
        q, k, v = torch.chunk(qkv, 3, dim=1)                                    # Pemisahan matriks tensor menjadi komponen Q, K, dan V
        
        r_h, r_w = H // self.n_win, W // self.n_win
        if r_h > 0 and r_w > 0:
            q_reg = F.adaptive_avg_pool2d(q, (self.n_win, self.n_win))          # Kompresi wilayah query berdasarkan ukuran jendela
            k_reg = F.adaptive_avg_pool2d(k, (self.n_win, self.n_win))          # Kompresi wilayah key berdasarkan ukuran jendela
        else:
            q_reg, k_reg = q, k
            
        q_reg = self.router_conv(q_reg)                                         # Ekstraksi korelasi lokal regional melalui konvolusi
        
        q_reg_flat = q_reg.flatten(2).transpose(1, 2)                           # Perataan dimensi spasial regional untuk query
        k_reg_flat = k_reg.flatten(2)                                           # Perataan dimensi spasial regional untuk key
        attn_reg = torch.matmul(q_reg_flat, k_reg_flat)                         # Perkalian titik matriks korelasi antar wilayah rahang
        
        actual_topk = min(self.topk, attn_reg.size(-1))
        _, indices = torch.topk(attn_reg, actual_topk, dim=-1)                  # Pemilihan koordinat wilayah dengan tumpang tindih tertinggi
        
        mask = torch.zeros_like(attn_reg).scatter_(-1, indices, 1.0)            # Pembuatan masker biner berdasarkan indeks koordinat top-k
        mask_spatial = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
        
        if mask_spatial.shape[-2:] == v.shape[-2:]:
            v_weighted = v * mask_spatial.mean(dim=1, keepdim=True).softmax(dim=-1) # Multiplikasi bobot wilayah dinamis ke komponen value (V)
        else:
            v_weighted = v

class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]
