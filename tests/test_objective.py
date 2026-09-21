import unittest
import torch
from oc_opd import signal, surrogate, backward_chunked


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)

    def test_gate_uses_default_tau_001(self):
        z = torch.randn(5, 19, dtype=torch.float64)
        t = z + 0.2 * torch.randn_like(z)
        g, d = signal(z, t)
        expected = 0.5 * (t.softmax(-1).sqrt() - z.softmax(-1).sqrt()).square().sum(-1, keepdim=True)
        torch.testing.assert_close(d['gate'], (expected / .01).clamp(0, 1))
        self.assertFalse(torch.allclose(d['gate'], (expected / .05).clamp(0, 1)))
        torch.testing.assert_close(g.sum(-1), torch.zeros(5, dtype=g.dtype), atol=1e-14, rtol=0)
        self.assertFalse(g.requires_grad)
        _, old = signal(z, t, tau=.05)
        torch.testing.assert_close(old['gate'], (expected / .05).clamp(0, 1))

    def test_identical_distributions_do_not_update(self):
        z = torch.randn(3, 17, dtype=torch.float64)
        g, d = signal(z, z)
        self.assertEqual(g.count_nonzero().item(), 0)
        self.assertEqual(d['gate'].count_nonzero().item(), 0)

    def test_masked_surrogate_gradient_and_frozen_teacher(self):
        z = torch.randn(2, 5, 23, dtype=torch.float64, requires_grad=True)
        t = torch.randn_like(z, requires_grad=True)
        mask = torch.tensor([[0, 1, 1, 0, 1], [0, 0, 1, 1, 0]], dtype=torch.bool)
        g, _ = signal(z[mask], t[mask])
        loss, _ = surrogate(z, t, mask)
        loss.backward()
        torch.testing.assert_close(z.grad[mask], g / mask.sum(), atol=1e-12, rtol=1e-10)
        self.assertEqual(z.grad[~mask].count_nonzero().item(), 0)
        self.assertIsNone(t.grad)

    def test_chunked_matches_direct_backbone_gradient(self):
        model = torch.nn.Linear(7, 11, dtype=torch.float64)
        head = torch.nn.Linear(11, 29, dtype=torch.float64).requires_grad_(False)
        teacher_head = torch.nn.Linear(13, 29, dtype=torch.float64).requires_grad_(False)
        x = torch.randn(2, 5, 7, dtype=torch.float64)
        th = torch.randn(2, 5, 13, dtype=torch.float64)
        mask = torch.tensor([[0, 1, 1, 0, 1], [0, 1, 1, 1, 1]], dtype=torch.bool)
        loss, _ = surrogate(head(model(x)), teacher_head(th), mask)
        loss.backward()
        expected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        backward_chunked(model(x), th, head, teacher_head, mask, chunk_size=3)
        for p, target in zip(model.parameters(), expected):
            torch.testing.assert_close(p.grad, target, atol=1e-12, rtol=1e-10)

    def test_validation_and_underflow(self):
        z = torch.tensor([[0., -200.]])
        g, _ = signal(z, -z)
        self.assertEqual(g.dtype, torch.float64)
        self.assertTrue(torch.isfinite(g).all())
        with self.assertRaises(ValueError):
            signal(z, z, epsilon=float('nan'))
        with self.assertRaises(ValueError):
            signal(z, z, tau=0)
        with self.assertRaises(ValueError):
            surrogate(z, z, torch.zeros(1, dtype=torch.bool))
        with self.assertRaises(ValueError):
            signal(z, torch.full_like(z, float('inf')))


if __name__ == '__main__':
    unittest.main()
