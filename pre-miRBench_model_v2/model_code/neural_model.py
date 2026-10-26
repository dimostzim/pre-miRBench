import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self,d,p):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv1d(64,64,5,padding=2*d,dilation=d),nn.GroupNorm(8,64),nn.GELU(),nn.Dropout1d(p),nn.Conv1d(64,64,5,padding=2*d,dilation=d),nn.GroupNorm(8,64))
    def forward(self,x): return torch.nn.functional.gelu(x+self.layers(x))

class Classifier(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.position=nn.Sequential(nn.Conv1d(18,64,7,padding=3),nn.GroupNorm(8,64),nn.GELU(),*[Residual(d,p) for d in [1,2,4,8]],nn.AvgPool1d(2),nn.Flatten(),nn.Linear(6400,128),nn.GELU(),nn.Dropout(p))
        self.global_branch=nn.Sequential(nn.Linear(1837,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(p),nn.Linear(128,64),nn.GELU())
        self.head=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Dropout(p),nn.Linear(64,1))
    def forward(self,g,x): return self.head(torch.cat([self.position(x),self.global_branch(g)],dim=1)).squeeze(1)
