## Our model was revised from https://github.com/zczcwh/PoseFormer/blob/main/common/model_poseformer.py

import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath

from common.opt import opts

opt = opts().parse()
device = torch.device("cuda")


#######################################################################################################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


#######################################################################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#######################################################################################################################
class CVA_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.Qnorm = nn.LayerNorm(dim)
        self.Knorm = nn.LayerNorm(dim)
        self.Vnorm = nn.LayerNorm(dim)
        self.QLinear = nn.Linear(dim, dim)
        self.KLinear = nn.Linear(dim, dim)
        self.VLinear = nn.Linear(dim, dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, CVA_input):
        B, N, C = x.shape
        q = self.QLinear(self.Qnorm(CVA_input)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.KLinear(self.Knorm(CVA_input)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.VLinear(self.Vnorm(x)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#######################################################################################################################
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


#######################################################################################################################
class Multi_Out_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        MSA = self.drop_path(self.attn(self.norm1(x)))
        x = x + MSA
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, MSA


#######################################################################################################################
class Multi_In_Out_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cva_attn = CVA_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, CVA_input):
        MSA = self.drop_path(self.cva_attn(x, CVA_input))
        x = x + MSA

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, MSA


#######################################################################################################################
class First_view_Spatial_features(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.block1 = Multi_Out_Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
                                      norm_layer=norm_layer)
        self.block2 = Multi_Out_Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],
                                      norm_layer=norm_layer)
        self.block3 = Multi_Out_Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[2],
                                      norm_layer=norm_layer)
        self.block4 = Multi_Out_Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[3],
                                      norm_layer=norm_layer)

        self.Spatial_norm = norm_layer(embed_dim_ratio)

    def forward(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        x, MSA1 = self.block1(x)
        x, MSA2 = self.block2(x)
        x, MSA3 = self.block3(x)
        x, MSA4 = self.block4(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x, MSA1, MSA2, MSA3, MSA4


#######################################################################################################################
class Spatial_features(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.block1 = Multi_In_Out_Block(
            dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        self.block2 = Multi_In_Out_Block(
            dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer)
        self.block3 = Multi_In_Out_Block(
            dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[2], norm_layer=norm_layer)
        self.block4 = Multi_In_Out_Block(
            dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[3], norm_layer=norm_layer)

        self.Spatial_norm = norm_layer(embed_dim_ratio)

    def forward(self, x, MSA1, MSA2, MSA3, MSA4):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        x, MSA1 = self.block1(x, MSA1)
        x, MSA2 = self.block2(x, MSA2)
        x, MSA3 = self.block3(x, MSA3)
        x, MSA4 = self.block4(x, MSA4)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x, MSA1, MSA2, MSA3, MSA4


#######################################################################################################################
class TemTemporal__features(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3
        ### Temporal patch embedding
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Temporal_norm = norm_layer(embed_dim)
        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

    def forward(self, x):
        b = x.shape[0]
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        # x = self.weighted_mean(x)
        x = x.view(b, opt.frames, -1)
        return x


#######################################################################################################################
class hmvformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        embed_dim = embed_dim_ratio * num_joints
        out_dim = num_joints * 3  #### output dimension is num_joints * 3
        ##Spatial_features
        self.SF1 = First_view_Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                               num_heads, mlp_ratio, qkv_bias, qk_scale,
                                               drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF2 = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                    num_heads, mlp_ratio, qkv_bias, qk_scale,
                                    drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF3 = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                    num_heads, mlp_ratio, qkv_bias, qk_scale,
                                    drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF4 = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                    num_heads, mlp_ratio, qkv_bias, qk_scale,
                                    drop_rate, attn_drop_rate, drop_path_rate, norm_layer)

        ## MVF
        self.view_pos_embed = nn.Parameter(torch.zeros(1, 4, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(4, momentum=0.1),
            nn.Conv2d(4, 1, kernel_size=opt.mvf_kernel, stride=1, padding=int(opt.mvf_kernel // 2), bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv_norm = nn.LayerNorm(embed_dim)

        # Time Serial
        self.TF = TemTemporal__features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                        num_heads, mlp_ratio, qkv_bias, qk_scale,
                                        drop_rate, attn_drop_rate, drop_path_rate, norm_layer)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        self.hop_w0 = nn.Parameter(torch.ones(17, 17))
        self.hop_w1 = nn.Parameter(torch.ones(17, 17))
        self.hop_w2 = nn.Parameter(torch.ones(17, 17))
        self.hop_w3 = nn.Parameter(torch.ones(17, 17))
        self.hop_w4 = nn.Parameter(torch.ones(17, 17))

        self.hop_global = nn.Parameter(torch.ones(17, 17))

        self.linear_hop = nn.Linear(8, 2)

    def forward(self, x, hops):
        b, f, v, j, c = x.shape

        ##############local feature#################
        x_hop1 = torch.einsum('bfvjc, jj -> bfvjc', torch.einsum('bjj, bfvjc -> bfvjc', hops[:, 0], x), self.hop_w1)
        x_hop2 = torch.einsum('bfvjc, jj -> bfvjc', torch.einsum('bjj, bfvjc -> bfvjc', hops[:, 1], x), self.hop_w2)
        # x_hop3 = torch.einsum('bfvjc, jj -> bfvjc', torch.einsum('bjj, bfvjc -> bfvjc', hops[:,2], x), self.hop_w3)
        # x_hop4 = torch.einsum('bfvjc, jj -> bfvjc', torch.einsum('bjj, bfvjc -> bfvjc', hops[:,3], x), self.hop_w4)

        ###############golbal feature#################
        x_hop_global = x.unsqueeze(3).repeat(1, 1, 1, 17, 1, 1)
        x_hop_global = x_hop_global - x_hop_global.permute(0, 1, 2, 4, 3, 5)
        x_hop_global = torch.sum(x_hop_global ** 2, dim=-1)
        hop_global = x_hop_global / torch.sum(x_hop_global, dim=-1).unsqueeze(-1)
        x_hop_global = torch.einsum('bfvjc, jj -> bfvjc', torch.einsum('bfvjj, bfvjc -> bfvjc', hop_global, x),
                                    self.hop_global)

        # x = self.linear_hop(torch.cat([x, x_hop1, x_hop2, x_hop3, x_hop4, x_hop_global], dim=-1))
        x = self.linear_hop(torch.cat([x, x_hop1, x_hop2, x_hop_global], dim=-1))

        x1 = x[:, :, 0]
        x2 = x[:, :, 1]
        x3 = x[:, :, 2]
        x4 = x[:, :, 3]

        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        x3 = x3.permute(0, 3, 1, 2)
        x4 = x4.permute(0, 3, 1, 2)

        x1, MSA1, MSA2, MSA3, MSA4 = self.SF1(x1)
        x2, MSA1, MSA2, MSA3, MSA4 = self.SF2(x2, MSA1, MSA2, MSA3, MSA4)
        x3, MSA1, MSA2, MSA3, MSA4 = self.SF3(x3, MSA1, MSA2, MSA3, MSA4)
        x4, MSA1, MSA2, MSA3, MSA4 = self.SF4(x4, MSA1, MSA2, MSA3, MSA4)

        x = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)), dim=1) + self.view_pos_embed
        x = self.pos_drop(x)

        x = self.conv_norm(self.conv(x).squeeze(1) + x1 + x2 + x3 + x4)

        x = self.TF(x)

        x = self.head(x)

        x = x.view(b, opt.frames, j, -1)

        return x

# x = torch.rand((8, 27, 4, 17 , 2))
# mvft = MultiPoseTransformer(num_frame=opt.frames, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
#                             num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
# print(mvft(x).shape)