"""Read paired CHW float arrays without RGB conversion or synthetic degradation."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

SIZES={'train':16,'val':16,'test_16':16,'test_32':32,'test_64':64}

def discover(root, split):
    if split not in SIZES:
        raise ValueError(f'Unknown split: {split}')
    folders=[Path(root)/split/k for k in ('LR','HR')]
    if not all(p.is_dir() for p in folders):
        raise FileNotFoundError(f'Missing LR/HR directories: {folders}')
    groups=[{p.stem:p for p in folder.glob('*.npy')} for folder in folders]
    if not groups[0] or groups[0].keys()!=groups[1].keys():
        raise ValueError(f'Empty or unmatched LR/HR split: {split}')
    return [(stem,groups[0][stem],groups[1][stem]) for stem in sorted(groups[0])]

def read_pair(pair, split):
    arrays=[np.load(p,allow_pickle=False) for p in pair[1:]]
    size=SIZES[split]
    for a,expected in zip(arrays,[(2,size,size),(2,12*size,12*size)]):
        if a.shape!=expected or not np.issubdtype(a.dtype,np.number) or np.iscomplexobj(a):
            raise ValueError(f'{pair[0]}: expected real CHW {expected}, got {a.shape}/{a.dtype}')
        if not np.isfinite(a).all():
            raise ValueError(f'{pair[0]}: nonfinite input')
    arrays=[np.ascontiguousarray(a,dtype=np.float32) for a in arrays]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError(f'{pair[0]}: float32 overflow')
    return arrays

class Normalizer:
    def __init__(self, lower, upper):
        self.lower=np.asarray(lower,dtype=np.float32)
        self.upper=np.asarray(upper,dtype=np.float32)
        if (self.lower.shape!=(2,) or self.upper.shape!=(2,) or
            not np.isfinite([self.lower,self.upper]).all() or not (self.upper>self.lower).all()):
            raise ValueError('Expected two finite increasing per-band ranges')

    @classmethod
    def fit(cls, root):
        lower=np.full(2,np.inf); upper=np.full(2,-np.inf)
        for pair in discover(root,'train'):
            for a in read_pair(pair,'train'):
                lower=np.minimum(lower,a.min(axis=(1,2)))
                upper=np.maximum(upper,a.max(axis=(1,2)))
        return cls(lower,upper)

    def transform(self, a):
        # Preserve out-of-range validation/test values; no silent clipping.
        return np.ascontiguousarray((a-self.lower[:,None,None])/
                                    (self.upper-self.lower)[:,None,None],dtype=np.float32)

    def record(self):
        return {'lower':self.lower.tolist(),'upper':self.upper.tolist(),
                'fit_split':'train','method':'full_train_joint_LR_HR_per_band_minmax','clip':False}

class Pairs(Dataset):
    def __init__(self, root, split, normalizer):
        self.pairs=discover(root,split)
        self.split=split
        self.normalizer=normalizer

    def __len__(self): return len(self.pairs)

    def __getitem__(self,index):
        pair=self.pairs[index]
        lr,hr=read_pair(pair,self.split)
        return torch.from_numpy(self.normalizer.transform(lr)),torch.from_numpy(self.normalizer.transform(hr)),pair[0]
