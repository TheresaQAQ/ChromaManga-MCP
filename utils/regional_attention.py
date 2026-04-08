"""
Regional Attention Processor
Route different text embeddings to different spatial regions during SDXL cross-attention.

Each region has independent prompt but shares same ControlNet + LoRA.
Suitable for multi-character single-LoRA scenarios (different trigger words activate different characters).
"""

import torch
import torch.nn.functional as F
from typing import Optional


class RegionalAttnProcessor:
    """
    Replace all cross-attention processors in UNet.
    During inference, use different encoder_hidden_states for different regions based on latent space masks.
    
    Supports non-square images and CFG (batch=2).
    """

    def __init__(self, region_masks, region_embeds, base_embed, latent_h, latent_w):
        self.region_masks = region_masks
        self.region_embeds = region_embeds
        self.base_embed = base_embed
        self.latent_h = latent_h
        self.latent_w = latent_w

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self-attention: use default
        if encoder_hidden_states is None:
            return self._default_attn(attn, hidden_states, hidden_states, attention_mask)

        input_dtype = hidden_states.dtype
        batch, seq_len, dim = hidden_states.shape

        # Infer current layer's spatial resolution
        h, w = self._infer_hw(seq_len)
        if h is None:
            return self._default_attn(attn, hidden_states, encoder_hidden_states, attention_mask)

        # CFG mode: batch=2 (conditional + unconditional)
        if batch == 2:
            hs_uncond, hs_cond = hidden_states.chunk(2)
            enc_uncond, enc_cond = encoder_hidden_states.chunk(2)

            out_uncond = self._default_attn(attn, hs_uncond, enc_uncond, attention_mask)
            out_cond = self._regional_attn(attn, hs_cond, enc_cond, attention_mask, h, w)

            return torch.cat([out_uncond, out_cond], dim=0).to(input_dtype)
        else:
            return self._regional_attn(attn, hidden_states, encoder_hidden_states, attention_mask, h, w)

    def _regional_attn(self, attn, hidden_states, encoder_hidden_states, attention_mask, h, w):
        """Execute regional routing attention (optimized: Q computed once)"""
        input_dtype = hidden_states.dtype
        hw = h * w

        # Q computed once, shared by all regions
        q = attn.to_q(hidden_states)
        q = attn.head_to_batch_dim(q)

        # Base (background) attention
        base_enc = self.base_embed[:1].to(hidden_states.device, hidden_states.dtype)
        result = self._attn_with_q(attn, q, hidden_states, base_enc, attention_mask)

        # Region attention
        for mask_2d, embed in zip(self.region_masks, self.region_embeds):
            region_enc = embed[:1].to(hidden_states.device, hidden_states.dtype)
            out_region = self._attn_with_q(attn, q, hidden_states, region_enc, attention_mask)

            # Interpolate mask to current layer resolution
            m = F.interpolate(
                mask_2d.to(hidden_states.device).to(input_dtype),
                size=(h, w),
                mode="nearest",
            )
            m = m.reshape(1, hw, 1)

            # Blend: region uses region output, outside keeps previous result
            result = result * (1 - m) + out_region * m

        return result.to(input_dtype)

    @staticmethod
    def _attn_with_q(attn, q, hidden_states, encoder_hidden_states, attention_mask):
        """Cross-attention with pre-computed Q"""
        input_dtype = hidden_states.dtype

        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)

        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        attn_weights = attn.get_attention_scores(q, k, attention_mask)
        out = torch.bmm(attn_weights, v)
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out.to(input_dtype)

    def _infer_hw(self, seq_len: int):
        """Infer h, w from seq_len and known latent aspect ratio"""
        ratio = self.latent_h / self.latent_w

        # Try various downsampling factors
        for divisor in [1, 2, 4, 8]:
            h = self.latent_h // divisor
            w = self.latent_w // divisor
            if h * w == seq_len:
                return h, w

        # Fallback: use aspect ratio
        h_est = int((seq_len * ratio) ** 0.5)
        w_est = seq_len // h_est if h_est > 0 else 0
        if h_est * w_est == seq_len and h_est > 0 and w_est > 0:
            return h_est, w_est

        # Try nearby values
        for dh in range(-2, 3):
            h_try = h_est + dh
            if h_try > 0 and seq_len % h_try == 0:
                w_try = seq_len // h_try
                if w_try > 0:
                    return h_try, w_try

        return None, None

    @staticmethod
    def _default_attn(attn, hidden_states, encoder_hidden_states, attention_mask):
        """Standard cross-attention"""
        input_dtype = hidden_states.dtype

        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        attn_weights = attn.get_attention_scores(q, k, attention_mask)
        out = torch.bmm(attn_weights, v)
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out.to(input_dtype)


def set_regional_attn(pipe, region_masks, region_embeds, base_embed, latent_h, latent_w):
    """Inject RegionalAttnProcessor to all cross-attention layers"""
    processor = RegionalAttnProcessor(region_masks, region_embeds, base_embed, latent_h, latent_w)
    attn_procs = {}
    for name, module in pipe.unet.attn_processors.items():
        if "attn2" in name:
            attn_procs[name] = processor
        else:
            attn_procs[name] = module
    pipe.unet.set_attn_processor(attn_procs)


def reset_attn(pipe):
    """Restore default attention processor"""
    from diffusers.models.attention_processor import AttnProcessor2_0
    pipe.unet.set_attn_processor(AttnProcessor2_0())
