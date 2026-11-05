"""CPU contract for the pinned OFT accumulation and optimizer-step semantics."""
import ast
import importlib.util
import unittest

from experiments.fewshot_sft.worker import instrument
from experiments.fewshot_sft.common import OFT


class AccumulationTests(unittest.TestCase):
    def test_pinned_trainer_scales_loss_and_steps_only_at_boundary(self):
        patched = instrument((OFT / 'vla-scripts/finetune.py').read_text())
        tree = ast.parse(patched)
        found_scale = any(isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'normalized_loss'
                    for target in node.targets)
            and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Div)
            and isinstance(node.value.right, ast.Attribute)
            and node.value.right.attr == 'grad_accumulation_steps'
            for node in ast.walk(tree))
        self.assertTrue(found_scale, 'OFT no longer divides microbatch loss by accumulation steps')
        self.assertIn('(batch_idx + 1) % cfg.grad_accumulation_steps == 0', patched)
        self.assertIn('gradient_step_idx = (batch_idx + 1) // cfg.grad_accumulation_steps', patched)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable on this host')
    def test_four_microbatches_match_one_full_batch_update(self):
        import torch
        torch.manual_seed(4)
        full = torch.nn.Linear(3, 2)
        accumulated = torch.nn.Linear(3, 2)
        accumulated.load_state_dict(full.state_dict())
        x = torch.randn(4, 3)
        y = torch.randn(4, 2)
        full_optim = torch.optim.SGD(full.parameters(), lr=0.03)
        accum_optim = torch.optim.SGD(accumulated.parameters(), lr=0.03)
        full_optim.zero_grad()
        torch.nn.functional.l1_loss(full(x), y).backward()
        full_gradients = [p.grad.detach().clone() for p in full.parameters()]
        full_optim.step()
        accum_optim.zero_grad()
        optimizer_steps = 0
        for index in range(4):
            loss = torch.nn.functional.l1_loss(accumulated(x[index:index + 1]), y[index:index + 1])
            (loss / 4).backward()
            if (index + 1) % 4 == 0:
                for got, expected in zip(accumulated.parameters(), full_gradients):
                    self.assertTrue(torch.allclose(got.grad, expected, atol=1e-7, rtol=1e-6))
                accum_optim.step()
                optimizer_steps += 1
        self.assertEqual(optimizer_steps, 1)
        for got, expected in zip(accumulated.parameters(), full.parameters()):
            self.assertTrue(torch.allclose(got, expected, atol=1e-7, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
