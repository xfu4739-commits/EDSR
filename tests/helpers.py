from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
torch.set_num_threads(2)
from model.edsr import EDSR

def tiny_model():
    return EDSR(SimpleNamespace(n_resblocks=1,n_feats=8,res_scale=0.1,n_colors=2,
                               scale=[12],rgb_range=1,shift_mean=False))

def fixture(root,split='train',value=1.):
    size={'train':16,'val':16,'test_16':16,'test_32':32,'test_64':64}[split]
    for kind,factor in [('LR',1),('HR',12)]:
        folder=Path(root)/split/kind; folder.mkdir(parents=True,exist_ok=True)
        a=np.stack([np.full((size*factor,size*factor),value,dtype=np.float32),
                    np.full((size*factor,size*factor),value+2,dtype=np.float32)])
        a[:,0,0]=[0,0]
        np.save(folder/'0000_x12.npy',a)
