import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import torch.nn.functional as F

EARLY_ATTN_FUSION = "early_fuse"
LATE_ATTN_FUSION = "late_fuse"
HIER_ATTN_FUSION = "hier_fuse"

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )


class LearnedPositionEmbedding(nn.Module):
    
    def __init__(self, seq_len: int, embed_dim: int) -> None:
        """
        Adds position embedding to the input vectors.
        """
        super(LearnedPositionEmbedding, self).__init__()
        self.pos_embed = nn.parameter.Parameter(torch.Tensor(seq_len, embed_dim))
        torch.nn.init.kaiming_normal_(self.pos_embed)
    

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 3, "Number of dimensions should be 3 [batch, seq_len, f_len]"
        
        # pos embedding shoud be broadcastable.
        out = input + self.pos_embed  # (batch, seq_len, f_len)
        return out




# Attention-Fusion for the visual and imu features:
class EarlyFusionAttention(nn.Module):
    """
    Refer to: 'Wayformer: Motion Forecasting via Simple & Efficient Attention Networks'
    https://arxiv.org/pdf/2207.05844.pdf
    For more general architecture refer to DeepMind's 'Perceiver: General Perception with Iterative Attention'.
    https://arxiv.org/abs/2103.03206
    """

    def __init__(
            self, 
            opt, 
            num_blocks: int = 1, 
            num_heads: int = 2, 
            pos_embed_req: bool = True,
        ) -> None:
        """
        This is more like cross-attention. 
        Cross-attends features from different modalities.

        Args:
            - opt: config args required for training.
            - num_blocks: number of attention blocks
            - num_heads: number of attention head per block 
        """
        super(EarlyFusionAttention, self).__init__()
        embed_dims = opt.v_f_len + opt.i_f_len
        self.num_blocks = num_blocks

        self.attn_blocks = []
        for _ in range(self.num_blocks):
            self.attn_blocks.append(
                nn.MultiheadAttention(
                    embed_dim=embed_dims,
                    num_heads=num_heads,
                    batch_first=True,
                )
            )
        
        self.pos_embed = None
        if pos_embed_req:
            self.pos_embed = LearnedPositionEmbedding(
                seq_len=opt.seq_len,
                embed_dim=embed_dims
            )
        

    def forward(
            self, 
            v_features: torch.Tensor, 
            i_features: torch.Tensor,
        ) -> torch.Tensor:
        """
        Concatenate the features and apply attention.
        """
        concat_vi = torch.cat((v_features, i_features), dim=-1)   # (batch, seq_len, v_len+i_len)
        
        x = self.pos_embed(concat_vi) if self.pos_embed else concat_vi  # (batch, seq_len, v_len+i_len)

        for attn_block in self.attn_blocks:
            x, _ = attn_block(query=x, key=x, value=x)
        
        return x  # (batch, seq_len, v_len+i_len)


# Attention-Fusion for the visual and imu features:
class LateFusionAttention(nn.Module):
    """
    Refer to: 'Wayformer: Motion Forecasting via Simple & Efficient Attention Networks'
    https://arxiv.org/pdf/2207.05844.pdf
    For more general architecture refer to DeepMind's 'Perceiver: General Perception with Iterative Attention'.
    https://arxiv.org/abs/2103.03206
    """

    def __init__(
            self, 
            opt, 
            num_blocks: int = 1, 
            num_heads: int = 2,
        ) -> None:
        """
        This is more like self-attention.
        The features from each modality self-attend before fusing together.

        Args:
            - opt: config args required for training.
            - num_blocks: number of attention blocks
            - num_heads: number of attention head per block
        """
        super(LateFusionAttention, self).__init__()
        v_embed_dims = opt.v_f_len
        i_embed_dims = opt.i_f_len
        self.num_blocks = num_blocks

        self.v_pos_embed = LearnedPositionEmbedding(
            seq_len=opt.seq_len,
            embed_dim=v_embed_dims
        )

        self.i_pos_embed = LearnedPositionEmbedding(
            seq_len=opt.seq_len,
            embed_dim=i_embed_dims
        )

        self.v_attn_blocks = []
        self.i_attn_blocks = []

        # Same number of blocks for both visual and imu features
        for _ in range(self.num_blocks):
            self.v_attn_blocks.append(
                nn.MultiheadAttention(
                    embed_dim=v_embed_dims,
                    num_heads=num_heads,
                    batch_first=True,
                )
            )
        
        for _ in range(self.num_blocks):
            self.i_attn_blocks.append(
                nn.MultiheadAttention(
                    embed_dim=i_embed_dims,
                    num_heads=num_heads,
                    batch_first=True,
                )
            )

    def forward(
            self, 
            v_features: torch.Tensor, 
            i_features: torch.Tensor, 
            concat: bool = True,
        ) -> torch.Tensor:
        """
        Apply self-attention over each modality and then concatenate,
          or simply return the transformed vectors.
        """
        v_features = self.v_pos_embed(v_features)  # (batch, seq_len, v_len)
        i_features = self.i_pos_embed(i_features)  # (batch, seq_len, i_len)

        for attn_block in self.v_attn_blocks:
            v_features, _ = attn_block(query=v_features, key=v_features, value=v_features)  # (batch, seq_len, v_len)
        
        for attn_block in self.i_attn_blocks:
            i_features, _ = attn_block(query=i_features, key=i_features, value=i_features)  # (batch, seq_len, i_len)
        
        if concat:
            return torch.cat((v_features, i_features), dim=-1)  # (batch, seq_len, v_len+i_len)
        else:
            return (v_features, i_features)



# Attention-Fusion for the visual and imu features:
class HierarchicalFusionAttention(nn.Module):
    """
    Refer to: 'Wayformer: Motion Forecasting via Simple & Efficient Attention Networks'
    https://arxiv.org/pdf/2207.05844.pdf
    For more general architecture refer to DeepMind's 'Perceiver: General Perception with Iterative Attention'.
    https://arxiv.org/abs/2103.03206
    """

    def __init__(self, opt) -> None:
        """
        Interleaves/combines late and early fusion.
        1. Apply self-attention over each modality and concatenate - LateFusion,
        2. Apply attention for cross-modality features - EarlyFusion over the concatenated vector.

        Args:
            - opt: config args required for training.
        """
        super(HierarchicalFusionAttention, self).__init__()
        self.late_fusion = LateFusionAttention(opt)
        self.cross_attn_fusion = EarlyFusionAttention(opt, pos_embed_req=False)

    def forward(
            self, 
            v_features: torch.Tensor, 
            i_features: torch.Tensor,
        ) -> torch.Tensor:

        v_features, i_features = self.late_fusion(
            v_features, 
            i_features, 
            concat=False,
        )  # (batch, seq_len, v_len), (batch, seq_len, i_len)
        
        out = self.cross_attn_fusion(v_features, i_features)
        return out  # (batch, seq_len, v_len+i_len)


        


# The inertial encoder for raw imu data
class Inertial_encoder(nn.Module):
    def __init__(self, opt):
        super(Inertial_encoder, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout))
        self.proj = nn.Linear(256 * 1 * 11, opt.i_f_len)

    def forward(self, x):
        # x: (N, seq_len, 11, 6)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))    # x: (N x seq_len, 11, 6)
        x = self.encoder_conv(x.permute(0, 2, 1))                 # x: (N x seq_len, 64, 11)
        out = self.proj(x.view(x.shape[0], -1))                   # out: (N x seq_len, 256)
        return out.view(batch_size, seq_len, 256)

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        # CNN
        self.opt = opt
        self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5)
        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
        __tmp = self.encode_image(__tmp)

        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len)
        self.inertial_encoder = Inertial_encoder(opt)

    def forward(self, img, imu):
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)

        # image CNN
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v = self.encode_image(v)
        v = v.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        v = self.visual_head(v)  # (batch, seq_len, 256)
        
        # IMU CNN
        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        imu = self.inertial_encoder(imu)
        return v, imu

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6


# The fusion module
class Fusion_module(nn.Module):
    def __init__(self, opt):
        super(Fusion_module, self).__init__()
        self.fuse_method = opt.fuse_method
        self.f_len = opt.i_f_len + opt.v_f_len
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, self.f_len))
        elif self.fuse_method == 'hard':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, 2 * self.f_len))  
        elif self.fuse_method == EARLY_ATTN_FUSION:
            self.net = EarlyFusionAttention(opt)
        elif self.fuse_method == LATE_ATTN_FUSION:
            self.net = LateFusionAttention(opt)
        elif self.fuse_method == HIER_ATTN_FUSION:
            self.net = HierarchicalFusionAttention(opt)

    def forward(self, v, i):
        if self.fuse_method == 'cat':
            return torch.cat((v, i), -1)
        elif self.fuse_method == 'soft':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights
        elif self.fuse_method == 'hard':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[:, :, :, 0]
        else:
            # Any of the attention modules:
            return self.net(v, i)

# The policy network module
class PolicyNet(nn.Module):
    def __init__(self, opt):
        super(PolicyNet, self).__init__()
        in_dim = opt.rnn_hidden_size + opt.i_f_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2))

    def forward(self, x, temp):
        logits = self.net(x)
        hard_mask = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        return logits, hard_mask

# The pose estimation network
class Pose_RNN(nn.Module):
    def __init__(self, opt):
        super(Pose_RNN, self).__init__()

        # The main RNN network
        f_len = opt.v_f_len + opt.i_f_len
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=opt.rnn_hidden_size,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True)

        # The output networks
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 12)) # we need the transformation matrix [3x4]

    def forward(self, fused_vi, prev=None):
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())
        
        out, hc = self.rnn(fused_vi) if prev is None else self.rnn(fused_vi, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return pose, hc



class DeepVIO(nn.Module):
    def __init__(self, opt):
        super(DeepVIO, self).__init__()

        self.Feature_net = Encoder(opt)
        self.fuse = Fusion_module(opt)
        self.Pose_net = Pose_RNN(opt)
        self.Policy_net = PolicyNet(opt)
        self.opt = opt
        
        initialization(self)

    def forward(self, img, imu, is_first=True, hc=None, temp=5, selection='gumbel-softmax', p=0.5):

        fv, fi = self.Feature_net(img, imu)
        fused_vi = self.fuse(fv, fi)
        batch_size = fv.shape[0]
        seq_len = fv.shape[1]

        poses, decisions, logits= [], [], []
        hidden = torch.zeros(batch_size, self.opt.rnn_hidden_size).to(fv.device) if hc is None else hc[0].contiguous()[:, -1, :]
        fv_alter = torch.zeros_like(fv) # zero padding in the paper, can be replaced by other 
        
        for i in range(seq_len):
            pose, hc = self.Pose_net(fused_vi[:, i:i+1, :], hc)
            poses.append(pose)
            hidden = hc[0].contiguous()[:, -1, :]

        poses = torch.cat(poses, dim=1)
        # decisions = torch.cat(decisions, dim=1)
        # logits = torch.cat(logits, dim=1)
        # probs = torch.nn.functional.softmax(logits, dim=-1)

        return poses, hc


def initialization(net):
    #Initilization
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(0)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n//4, n//2
                    param.data[start:end].fill_(1.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
