"""Causal, bounded edge priors added to the existing directed attention logits."""
import math
import torch
from torch import nn

from full_data import build as static_build
from v7_prepare import FEATURES


class EdgeGate(nn.Module):
    def __init__(self, cfg, seed):
        super().__init__()
        spec=cfg['dynamic_edge']
        self.columns=tuple(FEATURES.index(name) for name in spec['fields'])
        self.missing_column=len(FEATURES)+FEATURES.index('lag20_relative_speed')
        self.log_bound=math.log(spec['bound'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            size=len(self.columns)*2
            self.net=nn.Sequential(nn.Linear(3*size,spec['hidden']),nn.GELU(),
                                   nn.Linear(spec['hidden'],cfg['heads']))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,adj,valid):
        # x: normalized values then explicit missing bits, [B,N,104].
        values=x[...,self.columns].clamp(-5,5)
        missing=x[...,tuple(c+len(FEATURES) for c in self.columns)]
        node=torch.cat((values,missing),-1)
        n=x.shape[1]
        # Attention coordinates are [target query, predecessor source key].
        source=node[:,None,:,:].expand(-1,n,-1,-1)
        target=node[:,:,None,:].expand(-1,-1,n,-1)
        descriptor=torch.cat((source,target,(source-target).abs()),-1)
        log_prior=self.log_bound*torch.tanh(self.net(descriptor)).permute(0,3,1,2)
        observed=x[...,self.missing_column]<.5
        usable=valid & observed
        active=adj.transpose(1,2).bool() & usable[:,:,None] & usable[:,None,:]
        active=active & ~torch.eye(n,device=x.device,dtype=torch.bool)[None]
        return log_prior*active[:,None]


def weighted_attention(block,h,adj,valid,log_prior):
    b,n,w=h.shape
    q,k,v=block.qkv(h).chunk(3,-1)
    q,k,v=[t.reshape(b,n,block.heads,block.d).transpose(1,2) for t in (q,k,v)]
    score=(q@k.transpose(-2,-1))/math.sqrt(block.d)+log_prior
    eye=torch.eye(n,device=h.device,dtype=torch.bool)[None]
    allowed=(adj.transpose(1,2).bool()|eye)&valid[:,None,:]
    allowed=allowed|((~valid)[:,:,None]&eye)
    attention=torch.softmax(score.masked_fill(~allowed[:,None],float('-inf')),-1)
    z=(attention@v).transpose(1,2).reshape(b,n,w)
    h=block.norm1(h+block.out(z))
    h=block.norm2(h+block.ff(h))
    return h*valid[:,:,None]


class DynamicGraphTransformer(nn.Module):
    def __init__(self,backbone,cfg,seed):
        super().__init__()
        self.backbone=backbone
        self.gates=nn.ModuleList([EdgeGate(cfg,seed+8191+i) for i in range(cfg['layers'])])
        self.edge_enabled=True

    def forward(self,x,adj,valid,center):
        b,n,f=x.shape
        if f!=104 or valid.shape!=(b,n) or adj.shape!=(b,n,n):
            raise ValueError('Invalid graph tensor shape')
        if center.shape!=(b,) or ((center<0)|(center>=n)).any():
            raise ValueError('Invalid target mapping')
        rows=torch.arange(b,device=x.device)
        if not valid[rows,center].all():
            raise ValueError('Target is padded or graph is empty')
        if not torch.isfinite(x).all():
            raise ValueError('Missing inputs must be imputed with masks')
        h=self.backbone.input(x)*valid[:,:,None]
        for core,gate in zip(self.backbone.blocks,self.gates,strict=True):
            bias=gate(x,adj,valid)
            if not self.edge_enabled:
                bias=bias*0
            h=weighted_attention(core,h,adj,valid,bias)
        pooled=self.backbone.pool(h,valid)
        return self.backbone.classifier(torch.cat((h[rows,center],pooled),-1)).squeeze(-1)


def build_model(features,cfg,model,seed):
    pooling=model.split('_')[-1]
    backbone=static_build(features,cfg,pooling,seed)
    return DynamicGraphTransformer(backbone,cfg,seed) if model.startswith('dynamic_') else backbone
