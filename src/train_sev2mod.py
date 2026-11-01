"""Train from scratch on train/val; evaluate held-out splits explicitly."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sev2mod_edsr.data import Normalizer,Pairs,SIZES
from sev2mod_edsr.runtime import MODEL_CONFIG,build_model,train_epoch,evaluate,save_checkpoint,load_checkpoint

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','evaluate'])
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--amp',action='store_true',help='CUDA mixed-precision training; evaluation stays float32')
    p.add_argument('--resume',type=Path,help='Continue this trainer\'s latest.pt, not pretrained RGB weights')
    p.add_argument('--checkpoint',type=Path,help='Required for evaluate')
    p.add_argument('--split',choices=['val','test_16','test_32','test_64'],help='evaluate only; default test_16')
    p.add_argument('--save-predictions',action='store_true',help='Save original-unit two-band float32 NPY predictions')
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    if min(args.batch_size,args.epochs)<=0 or args.workers<0 or args.lr<=0:
        raise ValueError('Invalid batch size, epochs, workers or learning rate')
    if args.command=='train' and args.checkpoint:
        raise ValueError('Fresh training has no pretrained weights; use --resume for a training checkpoint')
    if args.command=='train' and (args.split or args.save_predictions):
        raise ValueError('Training selects on val only; --split/--save-predictions belong to evaluate')
    if args.command=='evaluate' and (not args.checkpoint or args.resume):
        raise ValueError('evaluate requires --checkpoint and disallows --resume')
    root=args.data_root.resolve(); out=args.output.resolve()
    if out==root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('Output and dataset directories must not overlap')
    if out.exists() and any(out.iterdir()) and not (args.command=='train' and args.resume):
        raise FileExistsError('Output is not empty; use a new directory')
    if args.command=='train' and args.resume and args.resume.resolve().parent!=out:
        raise ValueError('Resume in the checkpoint\'s original output directory')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu explicitly for CPU checks')
    if args.amp and args.device!='cuda': raise ValueError('--amp requires CUDA')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device(args.device)
    model=build_model().to(device)
    out.mkdir(parents=True,exist_ok=True)
    if args.command=='evaluate':
        args.split=args.split or 'test_16'
        state=load_checkpoint(args.checkpoint,model,device)
        norm=Normalizer(**{k:state['normalization'][k] for k in ('lower','upper')})
        dataset=Pairs(root,args.split,norm)
        loader=DataLoader(dataset,batch_size=1,num_workers=args.workers)
        report=evaluate(model,loader,device,norm,out/'predictions' if args.save_predictions else None)
        report.update(split=args.split,checkpoint=str(args.checkpoint.resolve()),epoch=state['epoch'])
        write_json(out/'metrics.json',report); print(json.dumps(report,indent=2)); return
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.5)
    scaler=torch.amp.GradScaler('cuda',enabled=args.amp)
    generator=torch.Generator().manual_seed(args.seed)
    start=0; best=float('inf')
    if args.resume:
        state=load_checkpoint(args.resume,model,device)
        if (state['training']['amp']!=args.amp or state['training']['batch_size']!=args.batch_size
                or state['training']['workers']!=args.workers):
            raise ValueError('Resume must preserve AMP, batch size and worker count')
        if state['training']['data_root']!=str(root): raise ValueError('Resume data root changed')
        norm=Normalizer(**{k:state['normalization'][k] for k in ('lower','upper')})
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']); start=state['epoch']; best=state['best_val_mse']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu()); generator.set_state(state['loader_rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    else:
        print('Fitting fixed normalization on all TRAIN LR/HR arrays only...',flush=True)
        norm=Normalizer.fit(root)
    if args.epochs<=start: raise ValueError('--epochs must exceed completed checkpoint epoch')
    train=Pairs(root,'train',norm); val=Pairs(root,'val',norm)
    training={'data_root':str(root),'amp':args.amp,'batch_size':args.batch_size,
              'workers':args.workers,'seed':args.seed,'initial_lr':args.lr,
              'loss':'L1','optimizer':'Adam','scheduler':'StepLR(step_size=200,gamma=0.5)',
              'augmentation':False,'train_samples':len(train),'val_samples':len(val)}
    if args.resume:
        # Effective optimizer/scheduler/seed come from the checkpoint, not new CLI defaults.
        training=state['training']
    write_json(out/'config.json',{'model':MODEL_CONFIG,'training':training,'normalization':norm.record(),
        'initialization':'random weights; no RGB checkpoint','selection':'minimum val mean normalized MSE'})
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,
                            pin_memory=device.type=='cuda',generator=generator)
    val_loader=DataLoader(val,batch_size=1,num_workers=args.workers)
    for epoch in range(start+1,args.epochs+1):
        begin=time.perf_counter()
        loss=train_epoch(model,train_loader,optimizer,device,scaler)
        train_seconds=time.perf_counter()-begin
        metrics=evaluate(model,val_loader,device,norm)
        improved=metrics['mean_mse_normalized']<best
        best=min(best,metrics['mean_mse_normalized'])
        scheduler.step()
        payload={'model_config':MODEL_CONFIG,'model':model.state_dict(),'normalization':norm.record(),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'epoch':epoch,'best_val_mse':best,'training':training,
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'loader_rng':generator.get_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        save_checkpoint(out/'latest.pt',payload)
        if improved: save_checkpoint(out/'best.pt',payload)
        record={'epoch':epoch,'train_l1':loss,'train_seconds':train_seconds,
                'samples_per_second':len(train)/train_seconds,'val':metrics,'best_val_mse':best}
        with (out/'history.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)

if __name__=='__main__': main()
