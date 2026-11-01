import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from tests.helpers import fixture
from sev2mod_edsr.data import Normalizer
from sev2mod_edsr.runtime import Metrics,evaluate

class EvaluationTests(unittest.TestCase):
    def test_known_error_per_band_and_float_precision(self):
        hr=np.zeros((1,2,8,8),dtype=np.float32)
        sr=hr.copy(); sr[:,0]=0.1; sr[:,1]=0.0001
        m=Metrics([10,20]); m.add(sr,hr); result=m.result()
        self.assertAlmostEqual(result['bands'][0]['psnr_db'],20,places=4)
        self.assertAlmostEqual(result['bands'][1]['psnr_db'],80,places=3)
        self.assertAlmostEqual(result['bands'][0]['mae_original_units'],1,places=5)
        self.assertGreater(result['bands'][1]['mse_normalized'],0)

    def test_perfect_metrics_and_original_float_npy_export(self):
        with tempfile.TemporaryDirectory() as d:
            x=torch.full((1,2,8,8),0.123456)
            norm=Normalizer([10,20],[30,60])
            result=evaluate(torch.nn.Identity(),[(x,x,['sample'])],torch.device('cpu'),norm,d)
            self.assertTrue(result['bands'][0]['perfect_reconstruction'])
            saved=np.load(Path(d)/'sample.npy')
            self.assertEqual(saved.dtype,np.float32); self.assertEqual(saved.shape,(2,8,8))
            np.testing.assert_allclose(saved[:,0,0],[10+20*.123456,20+40*.123456],rtol=1e-6)
