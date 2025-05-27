"""
PyTorch implementation of a multimodal text-audio language model that can process both text and audio tokens.
The model is based on a pre-trained language model backbone (e.g. Pythia) and adds audio-specific layers and embeddings.

Key features:
- Supports both text-only, audio-only and multimodal inputs/outputs
- Uses codebook patterns for audio token embeddings
- Configurable audio input/output layers
- Layer-wise attention weighting for audio processing
- Supports various pre-trained backbones via HuggingFace Transformers

Code adapted from https://github.com/karpathy/llama2.c

Classes:
    ModelArgs: Configuration dataclass for model hyperparameters
    TextAudioLMFromPretrained: Main model class implementing the multimodal architecture

Author: Anonymous
"""

import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional
from torch.nn import functional as F
from transformers import AutoModelForCausalLM


@dataclass
class ModelArgs:
    """Configuration dataclass containing model architecture parameters and training options.
    
    Args:
        audio_vocabsize (int, optional): Size of audio vocabulary
        txt_vocabsize (int, optional): Size of text vocabulary  
        block_size (int, optional): Maximum sequence length. Defaults to 2048
        audio_pad_token (int, optional): Token ID used for audio padding
        bop_token (int, optional): Beginning of prompt token ID (ASR & TTS only)
        eop_token (int, optional): End of prompt token ID (ASR & TTS only)
        swt_token (int, optional): Switch token ID for modality changes
        n_codebooks (int, optional): Number of audio codebooks. Defaults to 1
        tie_audio_embeddings (bool, optional): Whether to tie audio input/output embeddings
        backbone (str): HuggingFace model ID for pretrained backbone. Defaults to "EleutherAI/pythia-410m-deduped"
        warm_init (bool, optional): Whether to initialize only new weights
        freeze_backbone (bool, optional): Whether to freeze pretrained backbone weights
        freeze_txt_inout (bool, optional): Whether to freeze text input/output embeddings
        n_audio_in_layers (int, optional): Number of audio-specific input layers
        n_audio_out_layers (int, optional): Number of audio-specific output layers
        layer_wa_audio (bool, optional): Whether to use layer-wise attention for audio
        layer_selwa_audio (bool, optional): Whether to use selective layer-wise attention
        selwa_in_layer (int, optional): Input layer for selective attention
        selwa_linear (bool, optional): Whether to use linear layer for selective attention\
        raw_speech_residual (bool, optional): Whether to add residual connections to raw audio
        entropy_reg (float, optional): Entropy regularization coefficient
        torch_dtype (torch.dtype, optional): Model dtype. Defaults to bfloat16
        attn_implementation (str, optional): Attention implementation. Defaults to "flash_attn"
    """
    # i/o args
    audio_vocabsize: Optional[int] = None
    txt_vocabsize: Optional[int] = None
    block_size: Optional[int] = 2048
    audio_pad_token: Optional[int] = None
    bop_token: Optional[int] = None
    eop_token: Optional[int] = None
    swt_token: Optional[int] = None
    n_codebooks: Optional[int] = 1
    tie_audio_embeddings: Optional[int] = True
    backbone: str = "EleutherAI/pythia-410m-deduped"
    warm_init: Optional[bool] = True
    freeze_backbone: Optional[bool] = False
    freeze_txt_inout: Optional[bool] = False
    n_audio_in_layers: Optional[int] = 0
    n_audio_out_layers: Optional[int] = 0
    layer_wa_audio: Optional[bool] = False
    layer_selwa_audio: Optional[bool] = False
    selwa_in_layer: Optional[int] = None
    selwa_linear: Optional[bool] = True
    raw_speech_residual: Optional[bool] = False
    entropy_reg: Optional[float] = 0.
    torch_dtype: Optional[torch.dtype] = torch.bfloat16
    attn_implementation: Optional[str] = "flash_attn"


class TextAudioLMFromPretrained(nn.Module):
    """
    A multimodal language model that can process both text and audio tokens.
    Built on top of a pretrained language model backbone with additional audio-specific components.
    
    The model supports:
    - Text-only inputs/outputs
    - Audio-only inputs/outputs  
    - Mixed text-audio sequences

    Args:
        params (ModelArgs): Configuration object containing model parameters
        
    Attributes:
        last_loss (torch.Tensor): Loss from most recent forward pass
        losses (dict): Dictionary tracking different loss components
    """
    last_loss: Optional[torch.Tensor]
    losses: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        # define the model type
        models_audio = params.audio_vocabsize is not None
        models_txt = params.txt_vocabsize is not None
        backbone_model = AutoModelForCausalLM.from_pretrained(
            params.backbone,
            torch_dtype=params.torch_dtype,
            attn_implementation=params.attn_implementation
        )
        # get required config info
        backbone_config = backbone_model.config
        self.dim = backbone_config.hidden_size
        self.n_layers = backbone_config.num_hidden_layers
        self.n_heads = backbone_config.num_attention_heads
        self.block_size = params.block_size
        # according to model type set params
        self.n_codebooks = 1
        if models_audio:
            self.audio_pad_token = params.audio_pad_token
        if models_txt:
            self.txt_pad_token = backbone_config.pad_token_id if backbone_config.pad_token_id is not None else \
                backbone_config.bos_token_id
            # move the embeddings from the backbone to the root module 
            # (just to have in/out txt/audio embeds in the same module)
            self.txt_embed = backbone_model.get_input_embeddings()
            self.txt_unembed = backbone_model.get_output_embeddings()
            if backbone_config.tie_word_embeddings:
                self.txt_unembed.weight = self.txt_embed.weight
            # expand embedding table if needed
            backbone_num_embs = self.txt_embed.weight.size(0)
            if params.txt_vocabsize > backbone_num_embs:
                num_new_embds = params.txt_vocabsize - backbone_num_embs
                updated_txt_embed = self.expand_pretrained_embs(self.txt_embed.weight.data, num_new_embds)
                self.txt_embed = nn.Embedding.from_pretrained(updated_txt_embed, freeze=False)
                self.txt_unembed = nn.Linear(self.dim, updated_txt_embed.size(0), bias=False)
                if backbone_config.tie_word_embeddings:
                    self.txt_unembed.weight = self.txt_embed.weight
                else:
                    updated_txt_unembed = self.expand_pretrained_embs(self.txt_unembed.weight.data, num_new_embds)
                    self.txt_unembed.weight.data = updated_txt_unembed

        # remove embeddings and output projections from the backbone
        backbone_model.set_input_embeddings(None)
        backbone_model.set_output_embeddings(None)
        # depending on the architecture, the transformer layers are stored in different attributes.
        if hasattr(backbone_model, "transformer"):  # For GPT-style models
            self.context_model = backbone_model.transformer
        elif hasattr(backbone_model, "gpt_neox"): # For GPT-NeoX models
            self.context_model = backbone_model.gpt_neox
        elif hasattr(backbone_model, "model"):  # For LLaMA and other similar architectures
            self.context_model = backbone_model.model
        else:
            raise ValueError(f"Unknown model structure for {params.backbone}")
        del backbone_model
        self.audio_in_layers, self.audio_out_layers = None, None
        if models_audio:
            # audio in/outs
            self.audio_embed = nn.ModuleList([nn.Embedding(params.audio_vocabsize, self.dim) for _ in range(self.n_codebooks)])
            self.audio_unembed = nn.ModuleList([nn.Linear(self.dim, params.audio_vocabsize, bias=False) for _ in range(self.n_codebooks)])
            if params.tie_audio_embeddings:
                # share the unembedding parameters with the embedding parameters
                for k in range(self.n_codebooks):
                    self.audio_embed[k].weight = self.audio_unembed[k].weight # https://paperswithcode.com/method/weight-tying
            if params.n_audio_in_layers:
                modal_layers_config_in = deepcopy(self.context_model.config)
                modal_layers_config_in._attn_implementation = "sdpa" # input layers use custom attention masks not supported by FlashAttn
                self.audio_in_layers = nn.ModuleList([
                    type(self.context_model.layers[0])(modal_layers_config_in, l_ix) for l_ix in range(params.n_audio_in_layers)
                ])
            if params.n_audio_out_layers:
                modal_layers_config_out = deepcopy(self.context_model.config)
                self.audio_out_layers = nn.ModuleList([ 
                    type(self.context_model.layers[0])(modal_layers_config_out, l_ix) for l_ix in range(params.n_audio_out_layers)
                ])
            if params.layer_wa_audio:
                self.layer_weights = nn.Parameter(0.01 * torch.randn((self.n_layers, 1, 1, 1)))
            if params.layer_selwa_audio:
                sel_in_dim = self.dim
                if params.selwa_linear:
                    self.layer_selector = nn.Linear(sel_in_dim, self.n_layers)
                else:
                    self.layer_selector = nn.Sequential(
                        nn.Linear(sel_in_dim, self.dim),
                        nn.GELU(),
                        nn.Linear(self.dim, self.n_layers)
                    )
        # initialization
        if not params.warm_init:
            self.apply(self._init_weights)
            # Special initialization for residual projections, scaled per GPT-2 paper
            for name, param in self.named_parameters():
                if name.endswith('w3.weight') or name.endswith('wo.weight'):
                    nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * self.params.n_layers))
        elif models_audio:
            # Initialize only the new weights
            for module in [self.audio_embed, self.audio_unembed]:
                module.apply(self._init_weights)
        if params.freeze_backbone:
            for param in self.context_model.parameters():
                param.requires_grad = False
        if models_txt and params.freeze_txt_inout:
            self.txt_embed.weight.requires_grad = False
            self.txt_unembed.weight.requires_grad = False
        # Print param counts
        text_params, audio_params = [], []
        is_audio_param = lambda name: any(keyword in name.lower() for keyword in ['audio', 'layer_weights', 'layer_selector'])
        for name, param in self.named_parameters():
            if is_audio_param(name):
                audio_params.append(param)
            else:
                text_params.append(param)
        num_text_params = sum(p.numel() for p in text_params)
        num_audio_params = sum(p.numel() for p in audio_params)
        print(f"num of text params:  {num_text_params:,}")
        print(f"num of audio params: {num_audio_params:,}")
        # initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None
        self.losses = {'lm': 0., 'coarse': 0., 'fine': 0.}
        if models_txt:
            self.losses.update({'txt': 0})
        if params.layer_selwa_audio:
            self.losses.update({'selwa_entropy': 0})

    def expand_pretrained_embs(self, emb_table, num_new_embeddings):
        embed_dim = emb_table.size(1)
        new_embs = torch.randn(
            num_new_embeddings, embed_dim,
            dtype=emb_table.dtype,
            device=emb_table.device
        )
        updated_emb_table = torch.cat([emb_table, new_embs], dim=0)
        return updated_emb_table

    def _init_weights(self, module):
        # regular inits    
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, dadapt=False, log_interval=None, audio_learning_rate=None):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # define groups, separate audio and non-audio modules
        audio_module_names = ['audio_in_layers', 'audio_out_layers', 'audio_embed', 'audio_unembed']
        # create optim groups. Any parameters that is 2D will be weight decayed, with the exception of the
        # state matrix A in Mamba models.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        audio_param_names = [pn for pn in param_dict if any(pn.startswith(mn) for mn in audio_module_names)]
        rest_param_names = [pn for pn in param_dict if pn not in audio_param_names]
        audio_decay_params = [p for n, p in param_dict.items() if n in audio_param_names and p.dim() >= 2 and 'mixer.A_log' not in n]
        audio_nodecay_params = [p for n, p in param_dict.items() if n in audio_param_names and (p.dim() < 2 or 'mixer.A_log' in n)]
        rest_decay_params = [p for n, p in param_dict.items() if n in rest_param_names and p.dim() >= 2 and 'mixer.A_log' not in n]
        rest_nodecay_params = [p for n, p in param_dict.items() if n in rest_param_names and (p.dim() < 2 or 'mixer.A_log' in n)]
        optim_groups = []
        if audio_decay_params:
            optim_groups.append({'params': audio_decay_params, 'weight_decay': weight_decay, 'lr': audio_learning_rate, 'is_audio': True})
        if audio_nodecay_params:
            optim_groups.append({'params': audio_nodecay_params, 'weight_decay': 0.0, 'lr': audio_learning_rate, 'is_audio': True})
        if rest_decay_params:
            optim_groups.append({'params': rest_decay_params, 'weight_decay': weight_decay, 'lr': learning_rate, 'is_audio': False})
        if rest_nodecay_params:
            optim_groups.append({'params': rest_nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate, 'is_audio': False})
        num_decay_params = sum(p.numel() for p in audio_decay_params + rest_decay_params)
        num_nodecay_params = sum(p.numel() for p in audio_nodecay_params + rest_nodecay_params)
        print(f"num decayed parameter tensors: {len(audio_decay_params) + len(rest_decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(audio_nodecay_params) + len(rest_nodecay_params)}, with {num_nodecay_params:,} parameters")
        if dadapt:
            # Using AdamW with adaptive learning rate https://arxiv.org/abs/2301.07733
            from dadapt_adam import DAdaptAdam
            optimizer = DAdaptAdam(optim_groups, lr=learning_rate, betas=betas, decouple=True, log_every=log_interval)
        else:
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        L, H, Q, T = self.n_layers, self.n_heads, self.dim//self.n_heads, self.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def embed_multimodal_input(self, audio_tokens, txt_tokens, audio_input_masks):
        # for the audio input, replace the text tokens for pad
        audio_tokens[~audio_input_masks.unsqueeze(1).expand(-1, self.n_codebooks, -1)] = self.audio_pad_token
        txt_tokens[audio_input_masks] = self.txt_pad_token # for the text input, replace the audio tokens for pad
        # apply codebook pattern
        input_audio_tokens = self.apply_delay_pattern(audio_tokens)
        # compute the audio embeddings as the sum of codebook embeddings
        h_audio = sum([self.audio_embed[k](input_audio_tokens[:, k]) for k in range(self.n_codebooks)])
        # make the non-audio audio embeddings zero
        h_audio[~audio_input_masks] = 0
        # compute text embeddings
        h_txt = self.txt_embed(txt_tokens)
        h_txt[audio_input_masks] = 0
        # adding the embeddings yields the multi-modal input embeddings
        return h_txt + h_audio

    def sample_from_logits(self, logits, temperature=1.0, top_k=None):
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[..., [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        return torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)

    def forward_audio_only(self, h, audio_input_masks, is_output, multimodal_input=False):
        """Forward pass for audio-only processing.
        
        Args:
            h (torch.Tensor): Input embeddings
            audio_input_masks (torch.Tensor): Boolean mask indicating audio tokens (1 for audio, 0 for text)
            is_output (bool): Whether this is an output adapter layer
            multimodal_input (bool): Whether the input is multimodal (i.e. text + audio)
            
        Returns:
            torch.Tensor: Upsampled audio embeddings
        """
        audio_layers = self.audio_out_layers if is_output else self.audio_in_layers
        bsize, seq_len, d = h.size()
        device = h.device
        if multimodal_input and not is_output:
            # we need to do some fancy masking to prevent modifying and attenting to text tokens
            original_h = h.clone()
            attention_masks = torch.full((bsize, seq_len, seq_len), False, device=device, dtype=bool)
            for i in range(bsize):
                audio_mask = audio_input_masks[i]
                attn_mask = torch.full((seq_len, seq_len), False, device=device, dtype=bool)
                # Identify contiguous segments of audio tokens
                segments = []
                start = None
                for j in range(seq_len):
                    if audio_mask[j] and start is None:
                        start = j
                    elif (not audio_mask[j]) and start is not None:
                        segments.append((start, j))
                        start = None
                if start is not None:
                    segments.append((start, seq_len))
                # For each audio segment, allow each token to attend to previous tokens within the same segment.
                for seg_start, seg_end in segments:
                    length = seg_end - seg_start
                    # Create a lower triangular mask for this segment
                    tri_mask = torch.tril(torch.ones((length, length), device=device, dtype=torch.bool))
                    attn_mask[seg_start:seg_end, seg_start:seg_end] = tri_mask
                # For non-audio tokens (text tokens), allow only self-attention.
                non_audio_positions = (~audio_mask).nonzero(as_tuple=False).squeeze(-1)
                # If there is only one non-audio token, non_audio_positions might not be 1D so ensure it's iterable:
                if non_audio_positions.dim() == 0:
                    non_audio_positions = non_audio_positions.unsqueeze(0)
                for pos in non_audio_positions:
                    attn_mask[pos, pos] = True
                attention_masks[i] = attn_mask
            # Expand attention masks to match the expected input shape for transformer layers
            # The attention mask should be broadcastable to [batch_size, num_heads, seq_len, seq_len]
            attention_masks = attention_masks.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            hidden_states = h
            position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0)
            for layer in audio_layers:
                hidden_states = layer(hidden_states, position_ids=position_ids, attention_mask=attention_masks)[0]
            # After processing, restore the original embeddings at non-audio positions
            # Ensure non-audio tokens are exactly as they were before
            h = torch.where(audio_input_masks.unsqueeze(-1), hidden_states, original_h)
        else: # otherwise just forward through the layers
            position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0)
            for layer in audio_layers:
                h = layer(h, position_ids=position_ids)[0]
        return h
    
    def forward(self,
                input_tokens: torch.Tensor,
                target_tokens: torch.Tensor,
                audio_input_masks: torch.Tensor,
                audio_preds_masks: torch.Tensor,
                txt_preds_masks: torch.Tensor,
                fine_weight: float = 1.) -> torch.Tensor:
        """Forward pass for multimodal processing.

        Args:
            input_tokens (torch.Tensor): Input tokens (B, 1, T)
            target_tokens (torch.Tensor): Target tokens (B, 1, T)
            audio_input_masks (torch.Tensor): Boolean mask indicating audio tokens (1 for audio, 0 for text) (B, T)
            audio_preds_masks (torch.Tensor): Boolean mask indicating audio predictions (1 for audio, 0 for text) (B, T)
            txt_preds_masks (torch.Tensor): Boolean mask indicating text predictions (1 for text, 0 for audio) (B, T)
            fine_weight (float, optional): Weight for fine audio tokens. Defaults to 1.
        
        Returns:
            torch.Tensor: Loss value
        """
        # sanity check
        bsize, ncodebooks, seqlen = input_tokens.size()
        assert seqlen <= self.params.block_size, f"Sequence beyond maximum length of {self.params.block_size}"
        assert ncodebooks == self.n_codebooks, "Sequence shape must match the specified number of codebooks"
        # determine if multimodal
        is_all_audio = audio_input_masks.all()
        has_audio = audio_input_masks.any()
        logits_audio = None
        logits_txt = None
        if is_all_audio:
            target_audio_tokens = target_tokens
            input_audio_tokens = self.apply_delay_pattern(input_tokens)
            # compute the frame audio embeddings as the sum of codebook embeddings
            h_raw = sum([self.audio_embed[k](input_audio_tokens[:, k]) for k in range(ncodebooks)])
            h = self.forward_audio_only(h_raw, audio_input_masks, is_output=False) if self.audio_in_layers is not None else h_raw
        elif not has_audio: # is all text
            target_txt_tokens = target_tokens[:, 0, :]
            input_txt_tokens = input_tokens[:, 0, :]
            # compute text embeddings
            h = self.txt_embed(input_txt_tokens)
        else: # multimodal input
            input_txt_tokens = input_tokens[:, 0, :]
            target_txt_tokens = target_tokens[:, 0, :]
            input_audio_tokens = input_tokens.clone()
            target_audio_tokens = target_tokens.clone()
            h_raw = self.embed_multimodal_input(input_audio_tokens, input_txt_tokens, audio_input_masks)
            h = self.forward_audio_only(h_raw, audio_input_masks, is_output=False, multimodal_input=True) if self.audio_in_layers is not None else h_raw
        # obtain contextual embeddings
        ctx_out = self.context_model(
            inputs_embeds=h,
            use_cache=False,
            output_hidden_states=self.params.layer_wa_audio or self.params.layer_selwa_audio
        )
        h_ctx = ctx_out['last_hidden_state']
        # compute loss
        txt_loss = 0.
        audio_loss = 0.
        selwa_entropy = 0.
        n_txt_preds = txt_preds_masks.sum()
        n_audio_preds = 0
        if audio_preds_masks.any():
            h_audio = h_ctx
            if self.params.layer_wa_audio or self.params.layer_selwa_audio:
                stacked_ctx_h = torch.stack(ctx_out['hidden_states'])[1:]
                if self.params.layer_wa_audio:
                    # compute context as weighted average of contextual representations in all layers
                    h_audio = (stacked_ctx_h * F.softmax(self.layer_weights, dim=0)).sum(0)
                elif self.params.selwa_in_layer is not None:
                    h_audio = ctx_out['hidden_states'][self.params.selwa_in_layer]
                if self.params.layer_selwa_audio:
                    # compute context as an input-dependent weighted average of contextual representations in all layers
                    selected_layer_weights = F.softmax(self.layer_selector(h_audio).permute(2, 0, 1).unsqueeze(-1), dim=0)
                    selwa_entropy = -(selected_layer_weights * (selected_layer_weights + 1e-12).log()).sum(dim=0).mean()
                    self.losses['selwa_entropy'] = selwa_entropy
                    h_audio = (stacked_ctx_h * selected_layer_weights).sum(0)
            if self.params.raw_speech_residual:
                h_audio = h_audio + h_raw
            if self.audio_out_layers is not None:
                h_audio = self.forward_audio_only(h_audio, audio_input_masks, is_output=True,
                                            multimodal_input=not is_all_audio)
            h_audio = h_audio if is_all_audio else h_audio[audio_preds_masks]
            # apply each audio prediction head to obtain logits per codebook
            logits_audio = torch.stack([self.audio_unembed[k](h_audio) for k in range(self.n_codebooks)], dim=1)
            if is_all_audio:
                logits_audio, logits_audio_mask= self.undelay_logits(logits_audio)
            else:
                # we put the logits back to the full sequence tensor shape filling with
                # nan (to make sure if there is any masking error we would easily detect it) the non-audio elements
                logits_padded = torch.full(
                    (bsize, self.params.block_size, ncodebooks, logits_audio.size(-1)),
                    float('nan'), dtype=logits_audio.dtype, device=logits_audio.device
                )
                logits_padded[audio_preds_masks] = logits_audio
                logits_padded = logits_padded.permute(0, 2, 1, 3)
                # we obtain the logits fitting the code pattern, while also masking the ncodebooks previous tokens to a modality change
                logits_audio, logits_audio_mask = self.undelay_logits(logits_padded, audio_preds_masks)                
                # compute the audio loss only on audio targets
                logits_audio_mask = logits_audio_mask * audio_preds_masks.unsqueeze(1).expand(-1, ncodebooks, -1)
            if logits_audio_mask.any():
                n_audio_preds = logits_audio_mask.sum()
                # calculate the loss on coarse audio tokens
                lm_coarse_loss = F.cross_entropy(logits_audio[:, 0][logits_audio_mask[:, 0]],
                                                 target_audio_tokens[:, 0][logits_audio_mask[:, 0]],
                                                 ignore_index=-1, reduction='sum')
                self.losses['coarse'] = lm_coarse_loss / (logits_audio_mask[:, 0]).sum()
                if self.n_codebooks > 1:
                    # calculate the loss on fine audio tokens
                    lm_fine_loss = F.cross_entropy(logits_audio[:, 1:][logits_audio_mask[:, 1:]],
                                                   target_audio_tokens[:, 1:][logits_audio_mask[:, 1:]], 
                                                   ignore_index=-1, reduction='sum')
                    audio_loss += ((lm_coarse_loss + fine_weight * lm_fine_loss)) / n_audio_preds
                    self.losses['fine'] = lm_fine_loss / (logits_audio_mask[:, 1:]).sum()
                else:
                    self.losses['fine'] = 0.
                    audio_loss += self.losses['coarse']
        if txt_preds_masks.any():
            logits_txt = self.txt_unembed(h_ctx[txt_preds_masks])
            # forbid predicting special tokens, not too important, but avoids high initial losses with pre-trained models
            if self.params.bop_token is not None: # is multimodal and we want to forbid the special tokens
                logits_txt[..., self.params.bop_token:self.params.bop_token + 3] = -float('inf')
            # calculate the loss on text tokens
            txt_loss = F.cross_entropy(logits_txt.view(-1, logits_txt.size(-1)), target_txt_tokens[txt_preds_masks], ignore_index=-1)
            self.losses['txt'] = txt_loss
        total_preds = n_txt_preds + n_audio_preds
        lm_loss = (txt_loss * n_txt_preds + audio_loss * n_audio_preds) / total_preds
        self.losses['lm'] = lm_loss
        self.last_loss = lm_loss - self.params.entropy_reg * selwa_entropy

    def get_modality_masks(self, prompt, audio_input_masks):
        audio_preds_masks = audio_input_masks.clone()
        txt_preds_masks = ~audio_input_masks
        if self.params.bop_token is not None: # is a multimodal model
            # prefix masks
            txt_tokens = prompt[:, 0, :]
            bop_idxs = torch.argwhere(txt_tokens == self.params.bop_token)
            eop_idxs = torch.argwhere(txt_tokens == self.params.eop_token)
            prefix_mask = torch.ones_like(txt_tokens, dtype=torch.bool)
            # Iterate through each pair of bop_idxs and eop_idxs and set the mask to zero for the tokens between bop_idx and eop_idx
            for bop, eop in zip(bop_idxs, eop_idxs):
                batch_idx = bop[0].item()
                bop_idx = bop[1].item()
                eop_idx = eop[1].item()
                prefix_mask[batch_idx, bop_idx:eop_idx + 1] = False  # inclusive of eop_idx
            # we don't want to predict the swt token
            swt_mask = txt_tokens != self.params.swt_token
            txt_preds_masks = txt_preds_masks * prefix_mask * swt_mask
            audio_preds_masks = audio_preds_masks * prefix_mask * swt_mask
            for bop, eop in zip(bop_idxs, eop_idxs):
                batch_idx = bop[0].item()
                bop_idx = bop[1].item()
                eop_idx = eop[1].item()
                is_asr = audio_input_masks[batch_idx, eop_idx - 1]
                if is_asr:
                    txt_preds_masks[batch_idx, eop_idx] = True
                else:
                    audio_preds_masks[batch_idx, eop_idx] = True
            # if there is any swt token
            if not swt_mask.all():
                swt_ixs = torch.argwhere(~swt_mask)
                # if before was text, then predict audio
                audio_preds_masks[swt_ixs[:, 0], swt_ixs[:, 1]] = ~audio_input_masks[swt_ixs[:, 0], swt_ixs[:, 1] - 1]
                # if before was audio, then predict text
                txt_preds_masks[swt_ixs[:, 0], swt_ixs[:, 1]] = audio_input_masks[swt_ixs[:, 0], swt_ixs[:, 1] - 1]
        return audio_preds_masks, txt_preds_masks

    def apply_delay_pattern(self, tokens):
        B, K, T = tokens.shape
        result = torch.full((B, K, T), self.audio_pad_token, dtype=tokens.dtype, device=tokens.device)
        for i in range(K):
            result[:, i, i: T] = tokens[:, i, :T - i]
        return result

    def undelay_logits(self, logits, modality_mask=None):
        B, K, T, D = logits.shape
        unpadded_length = T - K
        # Create an empty tensor to store the reconstructed sequence
        undelayed_logits = torch.full((B, K, T, D), float('nan'), dtype=logits.dtype, device=logits.device)
        undelayed_logits_mask = torch.ones((B, K, T), dtype=bool, device=logits.device)
        undelayed_logits_mask[..., -K:] = False
        # Reconstruct the original sequence by removing the delays
        for i in range(K):
            undelayed_logits[:, i, :-K] = logits[:, i, i:i+unpadded_length, :]
        # we want to skip the logits that are K steps before a modality change
        if modality_mask is not None:
            modality_changes = torch.argwhere((modality_mask & ~modality_mask.roll(-1))[:, :-1])
            for change_ix in modality_changes:
                undelayed_logits_mask[change_ix[0], :, max(0, change_ix[1] - logits.size(1) + 1):change_ix[1] + 1] = False
        return undelayed_logits, undelayed_logits_mask


    def generate(
        self,
        prompt: torch.Tensor,
        audio_input_masks: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        audio_eos_id: int = 500,
        txt_eos_id: int = 0,
    ) -> torch.Tensor:
        device = prompt.device
        bsize, n_codebooks, seqlen = prompt.size()

        # Determine modality masks.
        if self.params.txt_vocabsize:
            audio_preds_masks, txt_preds_masks = self.get_modality_masks(prompt, audio_input_masks)
        else:
            audio_preds_masks = torch.ones((bsize, seqlen), dtype=bool, device=device)
            txt_preds_masks = ~audio_input_masks
        
        # Audio-only branch.
        generated = prompt.clone()
        for _ in range(max_new_tokens):        
            if audio_input_masks.all():
                # Audio-only branch
                input_audio_tokens = generated.clone()
                h_raw = sum([self.audio_embed[k](input_audio_tokens[:, k])
                                for k in range(n_codebooks)])
                h = self.forward_audio_only(h_raw, audio_input_masks, is_output=False) if self.audio_in_layers is not None else h_raw
            elif not audio_input_masks.any():
                # Text-only branch.
                input_txt_tokens = generated[:, 0, :].clone()
                h = self.txt_embed(input_txt_tokens)
            else:
                input_txt_tokens = generated[:, 0, :].clone()
                input_audio_tokens = generated.clone()
                # Multimodal branch.
                h_raw = self.embed_multimodal_input(input_audio_tokens, input_txt_tokens, audio_input_masks)
                h = self.forward_audio_only(h_raw, audio_input_masks, is_output=False, multimodal_input=True) if self.audio_in_layers is not None else h_raw

            h = h.to(torch.bfloat16)
            ctx_out = self.context_model(
                inputs_embeds=h,
                use_cache=False,
                output_hidden_states=self.params.layer_wa_audio or self.params.layer_selwa_audio,
                past_key_values=None
            )
            h_ctx = ctx_out['last_hidden_state']
            
            is_audio_target = audio_preds_masks[..., -1].all()  # if all batches predict audio
            if is_audio_target:
                h_audio = h_ctx
                if (self.params.layer_wa_audio or self.params.layer_selwa_audio):
                    stacked_ctx_h = torch.stack(ctx_out['hidden_states'])[1:]
                    if self.params.layer_wa_audio:
                        h_audio = (stacked_ctx_h * F.softmax(self.layer_weights, dim=0)).sum(0)
                    elif self.params.selwa_in_layer is not None:
                        h_audio = ctx_out['hidden_states'][self.params.selwa_in_layer]
                    if self.params.layer_selwa_audio:
                        selected_layer_weights = F.softmax(
                            self.layer_selector(h_audio).permute(2, 0, 1).unsqueeze(-1), dim=0
                        )
                        h_audio = (stacked_ctx_h * selected_layer_weights).sum(0)
                if self.params.raw_speech_residual and audio_input_masks.any():
                    h_audio = h_audio + h_raw
                    
                if self.audio_out_layers is not None:
                    h_audio = self.forward_audio_only(
                        h_audio,
                        audio_input_masks,
                        is_output=True,
                        multimodal_input=not audio_input_masks.all()
                    )
                
                h_audio = h_audio[audio_preds_masks].view(1, -1, h_audio.shape[-1])
            
                logits = torch.stack(
                        [self.audio_unembed[k](h_audio) for k in range(n_codebooks)],
                        dim=1  # (B, n_codebooks, T, vocab_size)
                )
                logits = logits[:, :, -1]
                logits[..., self.audio_pad_token] = -float('inf')  # forbid pad token
                
                next_token = self.sample_from_logits(logits, temperature, top_k)
                next_token = next_token.reshape(bsize, n_codebooks, 1)
                generated = torch.cat([generated, next_token], dim=-1)
                # Update modality masks.
                audio_input_masks = torch.cat((audio_input_masks, torch.ones((bsize, 1), device=device, dtype=torch.bool)), dim=1)
                audio_preds_masks = torch.cat((audio_preds_masks, torch.ones((bsize, 1), device=device, dtype=torch.bool)), dim=1)
                txt_preds_masks = torch.cat((txt_preds_masks, torch.zeros((bsize, 1), device=device, dtype=torch.bool)), dim=1)

                if next_token == audio_eos_id:
                    break
            else:
                h_txt = h_ctx[txt_preds_masks].view(1, -1, h_ctx.shape[-1])
                # For text: compute logits for the last token.
                logits = self.txt_unembed(h_txt[:, -1])
                logits[..., -3:] = -float('Inf')
                next_token = self.sample_from_logits(logits, temperature, top_k)
                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=-1)
                # Update modality masks.
                audio_input_masks = torch.cat((audio_input_masks, torch.zeros((bsize, 1), device=device, dtype=torch.bool)), dim=1)
                audio_preds_masks = torch.cat((audio_preds_masks, torch.zeros((bsize, 1), device=device, dtype=torch.bool)), dim=1)
                txt_preds_masks = torch.cat((txt_preds_masks, torch.ones((bsize, 1), device=device, dtype=torch.bool)), dim=1)

                if next_token == txt_eos_id:
                    break
            
        return generated, (audio_input_masks, audio_preds_masks, txt_preds_masks)


def load_ckpt(model_id, ckpt_id, device, should_compile=True):
    ckpt = torch.load(f"out/{model_id}/{ckpt_id}", map_location=device, weights_only=False)
    model_args = ckpt["model_args"]
    state_dict = ckpt["model"]
    modelconf = ModelArgs(**model_args)
    model = TextAudioLMFromPretrained(modelconf)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if should_compile:
        model = torch.compile(model)
    return model