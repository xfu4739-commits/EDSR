import unittest
import torch
from tests.helpers import tiny_model
from model.common import Upsampler,default_conv
from sev2mod_edsr.runtime import MODEL_CONFIG

class ModelTests(unittest.TestCase):
    def test_x12_shape_gradient_and_no_rgb_mean(self):
        model=tiny_model(); x=torch.rand(1,2,16,16)
        y=model(x); self.assertEqual(tuple(y.shape),(1,2,192,192))
        y.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertIsInstance(model.sub_mean,torch.nn.Identity)

    def test_paper_config_and_legacy_scales(self):
        self.assertEqual((MODEL_CONFIG['n_resblocks'],MODEL_CONFIG['n_feats'],MODEL_CONFIG['res_scale']),(32,256,0.1))
        for scale in (2,3,4,8):
            self.assertEqual(Upsampler(default_conv,scale,4)(torch.rand(1,4,2,2)).shape[-1],2*scale)

    def test_strict_checkpoint_rejects_missing_and_wrong_tail(self):
        model=tiny_model(); state=model.state_dict(); state.pop(next(iter(state)))
        with self.assertRaises(RuntimeError): model.load_state_dict(state,strict=True)
        state=model.state_dict(); name=next(n for n in state if n.startswith('tail') and n.endswith('weight'))
        state[name]=state[name][:1]
        with self.assertRaises(RuntimeError): model.load_state_dict(state,strict=True)
