"""Small CPU regression tests for padding visibility and comparison semantics."""
import unittest

import torch
from omegaconf import OmegaConf

from validate_wmt14_kv_fix import ROOT, make_variants, stats
from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE


def tiny_model(pad_id=0):
    cfg = OmegaConf.load(ROOT / "upstream/ncm/config/eval/wmt14.yaml")
    cfg.model.n_embd = 24
    cfg.model.n_head = 2
    cfg.model.n_layer = 2
    cfg.model.prefix_len = 6
    cfg.model.context_length = 10
    cfg.model.vocab_size = [16, 16]
    cfg.model.prefix_pad_token_id = pad_id
    cfg.tokenizer.vqvae.n_layers = 2
    cfg.tokenizer.vqvae.embed_dim = 24
    cfg.tokenizer.vqvae.in_vocab_size = 32
    cfg.tokenizer.vqvae.context_length = 4
    cfg.tokenizer.vqvae.vqvae_vocab_size = [16, 16, 16]
    cfg.tokenizer.vqvae.vector_quantizer_config.levels.manual_levels = [1, 2, 4]
    cfg.tokenizer.vqvae.vector_quantizer_config.learned_l1_cascade_base = 2
    torch.manual_seed(17)
    vq = VQVAE(**cfg.tokenizer.vqvae).eval()
    return NextConceptTransformer(vq, cfg.model).eval()


class CachePaddingTest(unittest.TestCase):
    def test_cached_padding_visibility_cfg_zero_and_one(self):
        net = tiny_model()
        prefix = torch.tensor([[2, 3, 0, 0, 0, 0], [4, 5, 6, 7, 8, 9]])
        for cfg in (0., 1.):
            with self.subTest(cfg=cfg):
                masks, logits = [], []

                def inspect_mask(module, args, kwargs):
                    if module.attn.caching:
                        masks.append(kwargs["attn_mask"].clone())

                handle = net.transformer.blocks[0].register_forward_pre_hook(inspect_mask, with_kwargs=True)
                original = net._sample_level

                def capture(value, *args):
                    logits.append(value.clone())
                    return original(value, *args)

                net._sample_level = capture
                try:
                    torch.manual_seed(42)
                    eager = net.generate(None, prefix, cfg_scale=cfg, cfg_null_eos=1, top_k=1, top_p=0.)
                    eager_logits = logits[:]
                    logits.clear()
                    torch.manual_seed(42)
                    cached = net.generate(None, prefix, cfg_scale=cfg, cfg_null_eos=1,
                                          top_k=1, top_p=0., use_kv_cache=True)
                finally:
                    handle.remove()
                    del net._sample_level
                self.assertTrue(torch.equal(eager, cached))
                for a, b in zip(eager_logits, logits):
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
                self.assertEqual(len(masks), 3)  # prefill plus two target levels
                expected = prefix.eq(0)
                if cfg:
                    null = torch.ones_like(expected)
                    null[:, 0] = False
                    expected = torch.cat([expected, null])
                for mask in masks:
                    wanted = expected[:, None, None, :].expand(-1, 1, mask.shape[-2], -1)
                    self.assertTrue(torch.equal(torch.isneginf(mask[..., :6]), wanted))
                    self.assertTrue(torch.all(mask[..., 6:] == 0))
                self.assertTrue(all(not block.attn.caching and block.attn.cached_k is None
                                    for block in net.transformer.blocks))

    def test_no_pad_id_keeps_legacy_behavior(self):
        net = tiny_model(pad_id=None)
        variants, _ = make_variants(net)
        prefix = torch.tensor([[1, 2, 3, 4, 5, 6]])
        outputs = []
        for name in ("legacy", "both"):
            torch.manual_seed(42)
            outputs.append(variants[name](None, prefix, use_kv_cache=True, top_k=1, top_p=0.))
        self.assertTrue(torch.equal(*outputs))

    def test_bitwise_detects_signed_zero_and_nonfinite(self):
        a = torch.tensor([[0., 1.]])
        b = torch.tensor([[-0., 1.]])
        result = stats(a, b)
        self.assertEqual(result["bitwise_different_elements"], 1)
        self.assertEqual(result["numerically_different_elements"], 0)
        self.assertEqual(result["outside_tolerance"], 0)
        self.assertFalse(stats(a, torch.tensor([[float("nan"), 1.]]))["finite"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
