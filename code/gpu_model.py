"""Small CUDA training networks; no pretrained parameters or inference dependency."""
import torch
from torch import nn

class Verifier(nn.Module):
    def __init__(self,d,architecture,width=64,second=32,dropout=.3):
        super().__init__(); self.architecture=architecture
        self.first=nn.Linear(d,width)
        self.second=nn.Linear(width*(4 if 'context' in architecture else 1),second)
        self.output=nn.Linear(second,1)
        self.drop=nn.Dropout(dropout)
        if architecture.startswith('residual'):
            nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self,x,same,baseline):
        h=torch.relu(self.first(x))
        if 'context' in self.architecture:
            mean=h.mean(dim=1,keepdim=True).expand_as(h)
            same_mean=torch.bmm(same,h)/same.sum(dim=2,keepdim=True).clamp_min(1.)
            h=torch.cat([h,mean,same_mean,h*mean],dim=2)
        h=self.drop(h); h=torch.relu(self.second(h)); h=self.drop(h)
        logits=self.output(h).squeeze(-1)
        if self.architecture.startswith('residual'): logits=logits+baseline
        return logits

def loss_value(logits,y,valid,architecture):
    if architecture.endswith('bce'):
        values=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
        return (values*valid).sum()/valid.sum().clamp_min(1.)
    z=logits.masked_fill(~valid.bool(),-1e9)
    positive=logits.masked_fill(~y.bool(),-1e9)
    return (torch.logsumexp(z,dim=1)-torch.logsumexp(positive,dim=1)).mean()
