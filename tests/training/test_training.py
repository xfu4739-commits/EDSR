import tempfile
import unittest
from pathlib import Path
import torch
from tests.helpers import tiny_model,fixture
from sev2mod_edsr.data import Normalizer,Pairs
from sev2mod_edsr.runtime import train_epoch,evaluate,save_checkpoint,load_checkpoint,MODEL_CONFIG

class TrainingTests(unittest.TestCase):
    def test_real_loader_update_evaluation_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(d); fixture(d,'val')
            norm=Normalizer.fit(d); model=tiny_model()
            loader=torch.utils.data.DataLoader(Pairs(d,'train',norm),batch_size=1)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            before=next(model.parameters()).detach().clone()
            loss=train_epoch(model,loader,optimizer,torch.device('cpu'),scaler)
            self.assertTrue(loss>=0); self.assertFalse(torch.equal(before,next(model.parameters())))
            result=evaluate(model,loader,torch.device('cpu'),norm)
            self.assertEqual(result['images'],1)
            file=Path(d)/'latest.pt'
            save_checkpoint(file,{'model_config':MODEL_CONFIG,'model':model.state_dict(),'optimizer':optimizer.state_dict()})
            copy=tiny_model(); state=load_checkpoint(file,copy,torch.device('cpu'))
            for a,b in zip(model.parameters(),copy.parameters()): self.assertTrue(torch.equal(a,b))
            state['model_config']={'scale':[4]}; save_checkpoint(file,state)
            with self.assertRaises(ValueError): load_checkpoint(file,copy,torch.device('cpu'))
