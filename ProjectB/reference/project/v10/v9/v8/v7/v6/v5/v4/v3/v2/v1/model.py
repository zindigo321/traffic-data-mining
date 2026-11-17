"""Directed, masked multi-head Graph Transformer. No graph library required."""
import math
import torch
from torch import nn

class GraphAttention(nn.Module):
    def __init__(self, width=32, heads=4):
        super().__init__()
        if width % heads: raise ValueError('width must be divisible by heads')
        self.heads, self.d = heads, width//heads
        self.qkv=nn.Linear(width,3*width)
        self.out=nn.Linear(width,width)
        self.norm1=nn.LayerNorm(width)
        self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))

    def forward(self,h,adj,valid):
        # h [B,N,D]; adj[b,i,j] is physical directed edge i -> j.
        b,n,w=h.shape
        q,k,v=self.qkv(h).chunk(3,-1)
        q,k,v=[t.reshape(b,n,self.heads,self.d).transpose(1,2) for t in (q,k,v)]
        scores=(q @ k.transpose(-2,-1))/math.sqrt(self.d) # [B,H,N,N]
        # A target attends its physical predecessors plus itself. No cross-batch keys.
        eye=torch.eye(n,device=h.device,dtype=torch.bool)[None]
        allowed=(adj.transpose(1,2).bool() | eye) & valid[:,None,:]
        # Padded queries get a dummy diagonal solely to avoid all -inf softmax.
        allowed=allowed | ((~valid)[:,:,None] & eye)
        weights=torch.softmax(scores.masked_fill(~allowed[:,None],float('-inf')),-1)
        z=(weights @ v).transpose(1,2).reshape(b,n,w)
        h=self.norm1(h+self.out(z))
        h=self.norm2(h+self.ff(h))
        return h*valid[:,:,None]

class Pool(nn.Module):
    def __init__(self,width,kind):
        super().__init__()
        if kind not in ('mean','max','attention'):raise ValueError(kind)
        self.kind=kind
        self.score=nn.Linear(width,1) if kind=='attention' else None
    def forward(self,h,valid):
        if (~valid.any(1)).any():raise ValueError('Empty graphs must be rejected')
        if self.kind=='mean':return (h*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
        if self.kind=='max':return h.masked_fill(~valid[:,:,None],float('-inf')).max(1).values
        a=self.score(h).squeeze(-1).masked_fill(~valid,float('-inf')).softmax(1)
        return (h*a[:,:,None]).sum(1)

class GraphTransformer(nn.Module):
    def __init__(self,features,width=32,heads=4,layers=2,pooling='mean'):
        super().__init__()
        self.input=nn.Linear(features,width)
        self.blocks=nn.ModuleList([GraphAttention(width,heads) for _ in range(layers)])
        # Same backbone/classifier initial values for paired seeds across pooling choices.
        self.classifier=nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,1))
        self.pool=Pool(width,pooling)
    def forward(self,x,adj,valid,center):
        b,n,f=x.shape
        if valid.shape!=(b,n) or adj.shape!=(b,n,n):raise ValueError('Invalid graph tensor shape')
        if center.shape!=(b,) or ((center<0)|(center>=n)).any():raise ValueError('Invalid target mapping')
        if not valid[torch.arange(b,device=x.device),center].all():raise ValueError('Target is padded')
        if not torch.isfinite(x).all():raise ValueError('Missing values need imputation and masks')
        h=self.input(x)*valid[:,:,None]
        for block in self.blocks:h=block(h,adj,valid)
        target=h[torch.arange(b,device=x.device),center]
        return self.classifier(torch.cat((target,self.pool(h,valid)),dim=-1)).squeeze(-1)
