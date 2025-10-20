# model.py
'''
提取多层特征 → 分割成 patch → 统一 patch 尺寸 → 统一 patch 向量长度 → 融合多层特征 → 输出固定维度的表征
'''

import torch
import torch.nn.functional as F
import copy

# 这个常量在 FeatureExtractor 类中被使用，所以需要在这里定义
out_channels = 384


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone, layers_to_extract_from, device, input_shape):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.device = device
        self.input_shape = input_shape
        self.patch_maker = PatchMaker(3, stride=1)
        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = Preprocessing(feature_dimensions, 1024)
        self.forward_modules["preprocessing"] = preprocessing

        preadapt_aggregator = Aggregator(target_dim=out_channels)

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.forward_modules.eval()

    @torch.no_grad()
    def embed(self, images):
        """Returns feature embeddings for images."""

        _ = self.forward_modules["feature_aggregator"].eval()
        features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]   #把多层特征图转换成列表

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]

        patch_shapes = [x[1] for x in features]       #提取每层的 patch 尺寸
        features = [x[0] for x in features]           #提取每层的 patch 特征数据
        ref_num_patches = patch_shapes[0]             #选取参考的 patch 数量

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

#型要在空间维度（patch grid）上进行插值（F.interpolate）。但插值要求输入像图像那样有“高×宽”两个维度。
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1],
                *_features.shape[2:]       #把 patch 列表重新 reshape 成 2D patch 网格   (B, H_patches, W_patches, C, h_patch, w_patch)
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)           #(B, C, h_patch, w_patch, H_patches, W_patches)
            perm_base_shape = _features.shape                            #保存当前形状，稍后要 reshape 回来。
            _features = _features.reshape(-1, *_features.shape[-2:])     #“前面所有的维度”合并成一个新的 batch 维度；“后面两个维度” (H_patches, W_patches) 保留下来。

#双线性插值统一 patch的数量
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1,
                                          *_features.shape[-3:])
            features[i] = _features                  #此步已经还原到与features形状一致
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]   #[B*N, C, H, W]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        features = torch.reshape(features, (-1, 64, 64, out_channels))   #[B, H, W, C=out_channels]
        features = torch.permute(features, (0, 3, 1, 2))            #[B, C, H, W]

        return features


# CNN 的特征图拆分成重叠的局部 patch（小块），方便后续做 patch-level 特征建模。

class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize                    # 每个 patch（小块）的边长大小
        self.stride = stride                          # 相邻 patch 之间的滑动步长，决定是否重叠

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            features: [torch.Tensor, bs x c x h x w]   # 输入特征图（批量、通道、高、宽）
        Returns:
            unfolded_features: [torch.Tensor, bs, num_patches, c, patchsize, patchsize]
        """

        # 计算 padding，使边缘像素也能被包含在 patch 中
        padding = int((self.patchsize - 1) / 2)

        # 定义 Unfold 操作，将特征图按滑动窗口展开成 patch 向量
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize,  # 每个 patch 的尺寸
            stride=self.stride,          # 滑动步长
            padding=padding,             # 边缘填充
            dilation=1                   # 膨胀系数（默认1，不改变卷积核间距）
        )

        # 执行 Unfold 操作
        # 输入: [B, C, H, W]
        # 输出: [B, C * patchsize * patchsize, N]，其中 N 是 patch 总数
        unfolded_features = unfolder(features)

        # 计算在每个空间维度（高、宽）上 patch 的数量
        number_of_total_patches = []
        for s in features.shape[-2:]:
            # 根据卷积输出尺寸公式计算 patch 数量：
            # n = (输入尺寸 + 2*padding - (kernel_size-1) - 1)/stride + 1
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))

        # 调整张量形状
        # 从 [B, C*patchsize^2, N] → [B, C, patchsize, patchsize, N]
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )

        # 调整维度顺序，使每个 patch 独立出来
        # 最终形状：[B, N, C, patchsize, patchsize]
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        # 如果需要返回空间排列信息（例如重建时用到）
        if return_spatial_info:
            return unfolded_features, number_of_total_patches

        # 默认只返回 patch 化后的特征张量
        return unfolded_features

#统一每个patch中特征向量的长度
class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.preprocessing_modules = torch.nn.ModuleList()  #创建一个空的 ModuleList，类似 Python 的列表，但专门存放子神经网络模块

        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)     #[batch, num_layers, output_dim]


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    #    features: [B*N, C, H, W]
    def forward(self, features):
        features = features.reshape(len(features), 1, -1)   # [B*N, 1, C*H*W]
        return F.adaptive_avg_pool1d(features,
                                     self.preprocessing_dim).squeeze(1)     #[B*N, output_dim]

#把多层、多 patch 特征融合成单一向量，长度固定
class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

 #   features: [B, num_layers, input_dim]
    def forward(self, features):

        features = features.reshape(len(features), 1, -1)   #[B, 1, num_layers * input_dim]
        features = F.adaptive_avg_pool1d(features, self.target_dim)   #[B, 1, target_dim]
        return features.reshape(len(features), -1)         #[B, target_dim]

#从一个预训练的骨干网络（如 ResNet）的多个指定中间层提取特征图（feature maps）。
class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):    #列表专门用来保存注册过的所有 hook 的句柄。
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()                           #调用 handle.remove()，把之前注册的钩子全部**注销（取消）**掉。
        self.outputs = {}

        for extract_layer in layers_to_extract_from:    #遍历 layers_to_extract_from 列表为每个要提取的层创建一个 ForwardHook 实例
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )

            #段负责定位要挂钩的子模块对象（即 network_layer）
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]

            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]  #取出对应层次的特征图

            #判断找到的模块是否为 Sequential（一种有序容器，支持索引）
            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )   #是 Sequential，则实际注册hook在network_layer[-1]   原因通常是想获取 Sequential 内最后一层的输出
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            try:
                _ = self.backbone(images)                   #这一行执行主干网络（backbone，例如 ResNet）的前向传播
            except LastLayerToExtractReachedException:      #如果一切正常，运行完整个 backbone；如果触发异常，就捕获并忽略（pass 表示什么都不做），
                pass
        return self.outputs                                 #所有 hook 捕获的中间特征图都存在 self.outputs 字典


#计算 backbone 网络中指定提取层的特征通道数，通过给网络输入一个假的图像张量，执行一次前向传播来推断每层输出大小。
    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in
                self.layers_to_extract_from]

#hook（钩子） 的实现本体用来取网络中间层输出
class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        # 判断当前层是不是最后一个要提取的层
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):     #这个方法让 ForwardHook 类的对象变成“可调用的
        self.hook_dict[self.layer_name] = output
        # 最后一层提取后提前中断网络前向传播
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass