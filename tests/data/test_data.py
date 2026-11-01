import tempfile
import unittest
from pathlib import Path
import numpy as np
from tests.helpers import fixture
from sev2mod_edsr.data import Normalizer,Pairs,discover

class DataTests(unittest.TestCase):
    def test_train_only_bounds_no_clipping_and_chw(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(d,'train',1); fixture(d,'val',100)
            norm=Normalizer.fit(d)
            self.assertEqual(norm.upper.tolist(),[1,3])
            lr,hr,name=Pairs(d,'val',norm)[0]
            self.assertEqual(tuple(lr.shape),(2,16,16)); self.assertEqual(tuple(hr.shape),(2,192,192))
            self.assertGreater(lr.max().item(),1)
            self.assertEqual(name,'0000_x12')

    def test_missing_pair_invalid_shape_and_nan(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(d); norm=Normalizer.fit(d)
            hr=Path(d)/'train/HR/0000_x12.npy'
            np.save(hr,np.zeros((192,192,2),dtype=np.float32))
            with self.assertRaises(ValueError): Pairs(d,'train',norm)[0]
            fixture(d); a=np.load(hr); a[0,0,0]=np.nan; np.save(hr,a)
            with self.assertRaises(ValueError): Pairs(d,'train',norm)[0]
            hr.unlink()
            with self.assertRaises(ValueError): discover(d,'train')

    def test_test_sizes(self):
        with tempfile.TemporaryDirectory() as d:
            norm=Normalizer([0,0],[1,3])
            for split,size in [('test_16',16),('test_32',32),('test_64',64)]:
                fixture(d,split)
                lr,hr,_=Pairs(d,split,norm)[0]
                self.assertEqual(lr.shape[-1],size); self.assertEqual(hr.shape[-1],size*12)
