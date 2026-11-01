"""Paper-size network, floating-point metrics, and checkpoint-safe training."""
from types import SimpleNamespace
import math
from pathlib import Path
import numpy as np
import torch
from skimage.metrics import structural_similarity
from model.edsr import EDSR

MODEL_CONFIG=dict(n_resblocks=32,n_feats=256,res_scale=0.1,n_colors=2,
                  scale=[12],rgb_range=1,shift_mean=False)

def build_model():
    return EDSR(SimpleNamespace(**MODEL_CONFIG))

def train_epoch(model,loader,optimizer,device,scaler):
    model.train(); total=0.; count=0
    for lr,hr,_ in loader:
        lr,hr=lr.to(device),hr.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=scaler.is_enabled()):
            sr=model(lr)
            if sr.shape!=hr.shape: raise ValueError('Prediction/target shape mismatch')
            loss=torch.nn.functional.l1_loss(sr,hr)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if not finite:
            if not scaler.is_enabled(): raise FloatingPointError('Nonfinite gradients')
            # GradScaler skips overflowed steps and decreases its scale.
        scaler.step(optimizer); scaler.update()
        total+=loss.detach().item()*len(lr); count+=len(lr)
    if not count: raise ValueError('Empty training loader')
    return total/count

class Metrics:
    def __init__(self, span):
        self.span=np.asarray(span,dtype=np.float64)
        self.sse=np.zeros(2); self.sae=np.zeros(2)
        self.ssim=np.zeros(2); self.pixels=0; self.images=0

    def add(self, sr,hr):
        if sr.shape!=hr.shape or sr.ndim!=4 or sr.shape[1]!=2:
            raise ValueError('Metrics require matching B,2,H,W arrays')
        if not np.isfinite(sr).all() or not np.isfinite(hr).all():
            raise FloatingPointError('Nonfinite evaluation values')
        diff=sr.astype(np.float64)-hr.astype(np.float64)
        self.sse+=(diff**2).sum(axis=(0,2,3)); self.sae+=np.abs(diff).sum(axis=(0,2,3))
        self.pixels+=sr.shape[0]*sr.shape[2]*sr.shape[3]
        for pred,target in zip(sr,hr):
            for b in range(2):
                self.ssim[b]+=structural_similarity(target[b],pred[b],data_range=1.)
            self.images+=1

    def result(self):
        if not self.pixels: raise ValueError('Empty evaluation loader')
        mse=self.sse/self.pixels; mae=self.sae/self.pixels
        return {'bands':[{'band':b+1,'mse_normalized':float(mse[b]),
                 'psnr_db':None if mse[b]==0 else float(-10*math.log10(mse[b])),
                 'perfect_reconstruction':bool(mse[b]==0),
                 'ssim':float(self.ssim[b]/self.images),
                 'mae_original_units':float(mae[b]*self.span[b]),
                 'rmse_original_units':float(np.sqrt(mse[b])*self.span[b])} for b in range(2)],
                 'mean_mse_normalized':float(mse.mean()),'images':self.images,
                 'metric_protocol':'No quantization, clipping, RGB conversion, or border crop. '
                  'PSNR uses per-band aggregate pixel MSE and normalized data_range=1; '
                  'SSIM is averaged per image. Null PSNR means exact reconstruction (+infinity).'}

@torch.no_grad()
def evaluate(model,loader,device,normalizer,save_dir=None):
    model.eval(); span=normalizer.upper-normalizer.lower
    metrics=Metrics(span)
    if save_dir is not None: Path(save_dir).mkdir(parents=True,exist_ok=True)
    for lr,hr,names in loader:
        # Evaluate float32, even when training uses AMP.
        sr=model(lr.to(device)).float().cpu().numpy(); hr=hr.numpy()
        metrics.add(sr,hr)
        if save_dir is not None:
            raw=sr*span[None,:,None,None]+normalizer.lower[None,:,None,None]
            for name,a in zip(names,raw):
                if Path(name).name!=name: raise ValueError('Invalid output sample name')
                np.save(Path(save_dir)/(name+'.npy'),a.astype(np.float32),allow_pickle=False)
    return metrics.result()

def save_checkpoint(path, payload):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temp); temp.replace(path)

def load_checkpoint(path,model,device):
    # Only load checkpoints created by this trainer and from trusted storage.
    state=torch.load(path,map_location=device,weights_only=False)
    if state.get('model_config')!=MODEL_CONFIG:
        raise ValueError('Checkpoint is not paper-size dual-band x12 EDSR')
    model.load_state_dict(state['model'],strict=True)
    return state
