"""Structure-aware residual CNN; adapted from iteration_1/model_architecture/model.py.
Shared model implementation for training and inference. No probability clipping.
"""
import torch
from torch import nn

ARCHITECTURE = {
    'name': 'pair_message_residual_cnn',
    'input_channels': ['A','C','G','U','(',')','.', 'depth_after/100',
                       'partner_offset/199','partner_A','partner_C','partner_G','partner_U'],
    'sequence_length': 200, 'stem_channels': 64, 'stem_kernel_size': 7,
    'residual_stages': [{'channels':64,'dilations':[1,2]},
                        {'channels':96,'dilations':[1,2]}],
    'residual_kernel_size': 5, 'residual_dropout': 0.10,
    'residual_padding': '2*dilation', 'residual_convolutions_per_block': 2,
    'projection_kernel_size': 1, 'pool_after_each_stage': {'type':'MaxPool1d','kernel_size':2},
    'final_channels':128, 'final_kernel_size':5, 'adaptive_pool_bins':10,
    'normalization':'BatchNorm1d(defaults, running statistics)',
    'activation':'GELU(default approximate=none)', 'conv_bias':False,
    'head_input_features':1281, 'hidden_units':128, 'head_dropout':0.40,
    'mfe_features':1, 'output':'one raw logit', 'initialization':'PyTorch defaults',
}
TRAINING = {
    'seed':42, 'batch_size':256, 'optimizer':'AdamW',
    'learning_rate':0.0005, 'weight_decay':0.001, 'optimizer_betas':[0.9,0.999],
    'optimizer_eps':1e-8, 'loss':'BCEWithLogitsLoss', 'pos_weight':None,
    'gradient_clip_norm':5.0, 'max_epochs':100, 'shuffle':True,
    'augmentation':'minibatch latent mixup after pair message', 'oversampling':False, 'label_smoothing':0.0,
    'mixup_alpha':0.2, 'mixup_lambda':'max(Beta(0.2,0.2), 1-lambda)',
    'mixup_scope':'training only; same lambda and device permutation for post-message hidden states, MFE, and float targets; partner indices never mixed',
    'initialization_policy':'from scratch; no pretrained weights',
    'selection_metric':'validation_average_precision',
    'checkpoint_selection':'highest exact AP (strict improvement)',
    'early_stopping_patience':15, 'early_stopping_min_delta':0.0001,
    'early_stopping_reference':'update only on improvement >= min_delta',
    'scheduler':'ReduceLROnPlateau', 'scheduler_mode':'max',
    'scheduler_factor':0.5, 'scheduler_patience':3, 'scheduler_threshold':1e-4,
    'scheduler_threshold_mode':'rel', 'scheduler_cooldown':0, 'scheduler_eps':1e-8,
    'minimum_learning_rate':0.00001, 'prediction_threshold':0.5,
    'training_device':'cuda', 'ensemble':False,
}

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2*dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(0.10),
            nn.Conv1d(channels, channels, 5, padding=2*dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.branch(x))

class PairMessage(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv1d(128,64,1,bias=False), nn.BatchNorm1d(64),
            nn.GELU(), nn.Dropout(0.10),
            nn.Conv1d(64,64,1,bias=False), nn.BatchNorm1d(64))
        self.activation = nn.GELU()

    @staticmethod
    def gather_partner(h, partner_indices, paired_mask):
        return torch.gather(h, 2, partner_indices[:,None,:].expand(-1,h.shape[1],-1)) * paired_mask[:,None,:]

    def forward(self, h, partner_indices, paired_mask):
        message = self.gather_partner(h, partner_indices, paired_mask)
        return self.activation(h + self.branch(torch.cat((h,message),dim=1)))


def latent_mixup(hidden, mfe, targets, lam, permutation):
    if not 0.5 <= float(lam) <= 1.0:
        raise ValueError('Use max(lambda, 1-lambda) in [0.5,1]')
    return tuple(lam*t + (1-lam)*t[permutation] for t in (hidden,mfe,targets))


class PreMiRNACNN(nn.Module):
    """Clean forward accepts channels [B,13,200], MFE [B,1], int64 partners [B,200]."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(13,64,7,padding=3,bias=False), nn.BatchNorm1d(64), nn.GELU(),
            ResidualBlock(64,1), ResidualBlock(64,2))
        self.pair_message = PairMessage()
        self.encoder = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(64,96,1,bias=False), nn.BatchNorm1d(96), nn.GELU(),
            ResidualBlock(96,1), ResidualBlock(96,2), nn.MaxPool1d(2),
            nn.Conv1d(96,128,5,padding=2,bias=False), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(10), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(1281,128), nn.GELU(), nn.Dropout(0.40), nn.Linear(128,1))

    def encode_pairs(self, sequence, mfe, partner_indices):
        if sequence.ndim != 3 or sequence.shape[1:] != (13,200):
            raise ValueError('Expected sequence shape [batch,13,200]')
        if mfe.shape != (sequence.shape[0],1):
            raise ValueError('Expected MFE shape [batch,1]')
        if partner_indices.shape != (sequence.shape[0],200) or partner_indices.dtype != torch.int64:
            raise ValueError('Expected int64 partner indices [batch,200]')
        paired_mask = (sequence[:,4,:] + sequence[:,5,:]) > 0
        return self.pair_message(self.stem(sequence),partner_indices,paired_mask)

    def classify(self, hidden, mfe):
        return self.head(torch.cat((self.encoder(hidden),mfe),dim=1)).squeeze(-1)

    def forward(self, sequence, mfe, partner_indices):
        return self.classify(self.encode_pairs(sequence,mfe,partner_indices),mfe)

    def forward_train(self, sequence, mfe, partner_indices, targets, lam=None, permutation=None):
        if not self.training:
            raise RuntimeError('Training path requires train mode')
        hidden = self.encode_pairs(sequence,mfe,partner_indices)
        if lam is not None:
            if permutation is None:
                permutation = torch.randperm(len(sequence),device=sequence.device)
            hidden, mfe, targets = latent_mixup(hidden,mfe,targets,lam,permutation)
        elif permutation is not None:
            raise ValueError('Permutation requires lambda')
        return self.classify(hidden,mfe), targets


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

ARCHITECTURE['pair_message'] = {
    'location':'after two 64-channel residual blocks, before first pool',
    'gather':'original-example partner hidden state; zero at unpaired positions',
    'branch':'concat(h,masked_partner) -> Conv1d(128,64,1,bias=False) -> BN -> GELU -> Dropout(0.10) -> Conv1d(64,64,1,bias=False) -> BN',
    'output':'GELU(h + branch)', 'modules':1}


class TwoMessagePreMiRNACNN(PreMiRNACNN):
    """Independent messages before and after the second 64-channel block."""
    def __init__(self):
        super().__init__()
        self.early_pair_message = PairMessage()

    def encode_pairs(self, sequence, mfe, partner_indices):
        if sequence.ndim != 3 or sequence.shape[1:] != (13,200):
            raise ValueError('Expected sequence shape [batch,13,200]')
        if mfe.shape != (sequence.shape[0],1):
            raise ValueError('Expected MFE shape [batch,1]')
        if partner_indices.shape != (sequence.shape[0],200) or partner_indices.dtype != torch.int64:
            raise ValueError('Expected int64 partner indices [batch,200]')
        paired_mask = (sequence[:,4,:] + sequence[:,5,:]) > 0
        hidden = sequence
        for layer in self.stem[:4]:
            hidden = layer(hidden)
        hidden = self.early_pair_message(hidden, partner_indices, paired_mask)
        hidden = self.stem[4](hidden)
        return self.pair_message(hidden, partner_indices, paired_mask)


class GridResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(6, channels), nn.GELU(), nn.Dropout2d(0.10),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(6, channels))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.branch(x))


class FoldedGridPreMiRNACNN(PreMiRNACNN):
    """Full-window one-message encoder plus independently encoded folded grid."""
    def __init__(self):
        super().__init__()
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(9,24,5,stride=2,padding=2,bias=False),
            nn.GroupNorm(6,24), nn.GELU(), GridResidualBlock(24), nn.MaxPool2d(2),
            nn.Conv2d(24,48,1,bias=False), nn.GroupNorm(6,48), nn.GELU(),
            GridResidualBlock(48), nn.MaxPool2d(2),
            nn.Conv2d(48,64,3,padding=1,bias=False), nn.GroupNorm(8,64), nn.GELU())
        self.grid_pool = nn.AdaptiveAvgPool2d((4,4))
        self.grid_projection = nn.Sequential(nn.Linear(1088,128), nn.GELU(), nn.Dropout(0.10))
        self.head = nn.Sequential(nn.Linear(1409,128), nn.GELU(), nn.Dropout(0.40), nn.Linear(128,1))

    def encode_grid(self, sequence, partner_indices):
        from pair_grid import build_pair_grid
        grid = self.grid_encoder(build_pair_grid(sequence, partner_indices))
        return self.grid_projection(torch.cat((self.grid_pool(grid).flatten(1), grid.amax(dim=(2,3))), dim=1))

    def classify_grid(self, hidden, grid_embedding, mfe):
        return self.head(torch.cat((self.encoder(hidden), grid_embedding, mfe),dim=1)).squeeze(-1)

    def forward(self, sequence, mfe, partner_indices):
        hidden = self.encode_pairs(sequence, mfe, partner_indices)
        grid_embedding = self.encode_grid(sequence, partner_indices)
        return self.classify_grid(hidden, grid_embedding, mfe)

    def forward_train(self, sequence, mfe, partner_indices, targets, lam=None, permutation=None):
        if not self.training:
            raise RuntimeError('Training path requires train mode')
        hidden = self.encode_pairs(sequence, mfe, partner_indices)
        grid_embedding = self.encode_grid(sequence, partner_indices)
        if lam is not None:
            if permutation is None:
                permutation = torch.randperm(len(sequence), device=sequence.device)
            hidden, mfe, targets = latent_mixup(hidden, mfe, targets, lam, permutation)
            grid_embedding = lam*grid_embedding + (1-lam)*grid_embedding[permutation]
        elif permutation is not None:
            raise ValueError('Permutation requires lambda')
        return self.classify_grid(hidden, grid_embedding, mfe), targets
