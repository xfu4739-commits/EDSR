"""Opt-in full paper-size CUDA CLI test on one synthetic pair per split.

Run from repository root: python -B -m tests.training.smoke_cli
Creates only temporary fixtures and outputs, never touches real datasets.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from tests.helpers import fixture

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for this optional integration test')
    repo=Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='edsr-cli-') as d:
        base=Path(d); data=base/'data'; out=base/'train'
        fixture(data,'train'); fixture(data,'val'); fixture(data,'test_16')
        entry=[sys.executable,'-B',str(repo/'src/train_sev2mod.py')]
        common=['--data-root',str(data),'--batch-size','1','--workers','0','--device','cuda','--amp']
        for epoch,resume in [(1,[]),(2,['--resume',str(out/'latest.pt')])]:
            subprocess.run(entry+['train','--output',str(out),'--epochs',str(epoch)]+common+resume,check=True,cwd=repo)
        subprocess.run(entry+['evaluate','--data-root',str(data),'--output',str(base/'test'),
            '--checkpoint',str(out/'best.pt'),'--split','test_16','--save-predictions'],check=True,cwd=repo)
        history=[json.loads(line) for line in (out/'history.jsonl').read_text().splitlines()]
        assert [record['epoch'] for record in history]==[1,2]
        config=json.loads((out/'config.json').read_text())
        assert config['model']['n_resblocks']==32 and config['model']['n_feats']==256
        assert (base/'test/predictions/0000_x12.npy').exists()
        print('PASS: full paper-size CUDA AMP train -> val/best -> resume -> float32 held-out eval -> NPY export')

if __name__=='__main__': main()
