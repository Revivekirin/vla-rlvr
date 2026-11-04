"""Safety regressions; fixture results are not dataset evidence."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.dataset_analysis import cleanup
from experiments.shot_scaling.core import digest


class CleanupTests(unittest.TestCase):
    def setup_tree(self, root):
        for p in (*cleanup.GUARDS, *('outputs/'+t for t in cleanup.TARGETS)):
            (root/p).mkdir(parents=True, exist_ok=True)
            (root/p/'record').write_text(p)

    def test_symlink_and_escape_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve(); (root/'link').symlink_to('/tmp')
            with self.assertRaises(ValueError): cleanup.inventory(root, root/'link')
            with self.assertRaises(ValueError): cleanup.safe(root, root/'../outside')

    def test_archive_tamper_approval_process_and_protected_guards(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve(); self.setup_tree(root)
            with patch.object(cleanup,'process_check',return_value={'clear':True}):
                p=cleanup.plan(root); dest=root/'archive'; cleanup.archive(p,dest)
                with self.assertRaises(ValueError): cleanup.apply(p,dest,'wrong approval')
                with patch.object(cleanup,'process_check',return_value={'clear':False}):
                    with self.assertRaises(ValueError): cleanup.apply(p,dest,digest(p))
                original=(dest/'artifacts/dataset_analysis/record').read_text()
                (dest/'artifacts/dataset_analysis/record').write_text('corrupt')
                with self.assertRaises(ValueError): cleanup.verify(dest)
                (dest/'artifacts/dataset_analysis/record').write_text(original)
                (root/'data/record').write_text('changed')
                with self.assertRaises(ValueError): cleanup.apply(p,dest,digest(p))
                (root/'data/record').write_text('data')
                cleanup.apply(p,dest,digest(p))
                self.assertTrue((root/'checkpoints/record').exists())
                self.assertTrue((root/'outputs/goal/record').exists())
                self.assertTrue((root/'outputs/libero/record').exists())
                self.assertFalse((root/'outputs/dataset_analysis').exists())
                cleanup.verify(dest)


if __name__=='__main__': unittest.main()
