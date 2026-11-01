import tempfile
import unittest
from pathlib import Path
from tests.helpers import fixture
from train_sev2mod import main

class CLIGuardTests(unittest.TestCase):
    def test_pretrained_and_test_selection_rejected_for_training(self):
        args=['train','--data-root','unused-data','--output','unused-output','--device','cpu']
        with self.assertRaises(ValueError): main(args+['--checkpoint','rgb_x4.pt'])
        with self.assertRaises(ValueError): main(args+['--split','test_64'])

    def test_evaluate_requires_checkpoint(self):
        with self.assertRaises(ValueError):
            main(['evaluate','--data-root','unused-data','--output','unused-output'])

    def test_output_cannot_modify_dataset_or_overwrite_existing_results(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'data'; fixture(root)
            with self.assertRaises(ValueError):
                main(['train','--data-root',str(root),'--output',str(root/'out')])
            output=Path(d)/'results'; output.mkdir(); (output/'keep').write_text('existing')
            with self.assertRaises(FileExistsError):
                main(['train','--data-root',str(root),'--output',str(output)])
            self.assertEqual((output/'keep').read_text(),'existing')
