use candle_core::{IndexOp, Module, Result, Tensor, D, DType};
use candle_nn::{
    layer_norm, linear, linear_no_bias, Conv2d, Conv2dConfig, Embedding, LayerNorm,
    LayerNormConfig, Linear, VarBuilder,
};

#[derive(Debug, Clone)]
struct QwenRMSNorm {
    weight: Tensor,
    eps: f64,
}

impl QwenRMSNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_float = x.to_dtype(DType::F32)?;
        let variance = x_float.powf(2.0)?.mean_keepdim(D::Minus1)?;
        let x_normed = x_float.broadcast_mul(&(variance + self.eps)?.sqrt()?.recip()?)?;
        let w = self.weight.to_dtype(DType::F32)?;
        let output = x_normed.broadcast_mul(&(w + 1.0)?)?;
        output.to_dtype(x.dtype())
    }
}

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f64,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub delta_num_heads: usize,
    pub delta_head_dim: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
}

impl Default for Qwen35Config {
    fn default() -> Self {
        Self {
            vocab_size: 248320,
            hidden_size: 1024,
            intermediate_size: 3584,
            num_hidden_layers: 24,
            rms_norm_eps: 1e-6,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 256,
            delta_num_heads: 16,
            delta_head_dim: 128,
            patch_size: 16,
            temporal_patch_size: 2,
            in_channels: 3,
        }
    }
}

#[derive(Clone)]
pub enum LayerState {
    Linear(Tensor, Tensor),
    Full(Tensor, Tensor),
    None,
}

fn repeat_interleave(t: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    if repeats == 1 { return Ok(t.clone()); }
    let (b, h, s, d) = t.dims4()?;
    if dim == 1 {
        t.unsqueeze(2)?.expand((b, h, repeats, s, d))?.reshape((b, h * repeats, s, d))
    } else {
        todo!("repeat_interleave only implemented for dim 1")
    }
}

pub struct VisionEncoder {
    patch_embed: Conv2d,
    blocks: Vec<VisionBlock>,
    merger: VisionMerger,
}

struct VisionBlock {
    norm1: LayerNorm,
    attn_qkv: Linear,
    attn_proj: Linear,
    norm2: LayerNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
}

struct VisionMerger {
    linear_fc1: Linear,
    linear_fc2: Linear,
    norm: LayerNorm,
}

const VIS_HIDDEN: usize = 768;
const VIS_HEADS: usize = 16;
const VIS_HEAD_DIM: usize = VIS_HIDDEN / VIS_HEADS;
const VIS_INTERMEDIATE: usize = VIS_HIDDEN * 4;
const VIS_MERGER_OUT: usize = 1024;
const VIS_MERGER_IN: usize = VIS_HIDDEN * 4;

impl VisionEncoder {
    pub fn load(vb: VarBuilder, cfg: &Qwen35Config) -> Result<Self> {
        let w = vb.get((VIS_HIDDEN, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size), "patch_embed.proj.weight")?;
        let w = w.reshape((VIS_HIDDEN, cfg.in_channels * cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size))?;
        let bias = vb.get(VIS_HIDDEN, "patch_embed.proj.bias").ok();
        let patch_embed = Conv2d::new(w, bias, Conv2dConfig { stride: cfg.patch_size, ..Default::default() });
        let mut blocks = Vec::with_capacity(12);
        for i in 0..12 {
            let b_vb = vb.pp(&format!("blocks.{}", i));
            blocks.push(VisionBlock {
                norm1: layer_norm(VIS_HIDDEN, LayerNormConfig { eps: 1e-6, ..Default::default() }, b_vb.pp("norm1"))?,
                attn_qkv: linear(VIS_HIDDEN, VIS_HIDDEN * 3, b_vb.pp("attn.qkv"))?,
                attn_proj: linear(VIS_HIDDEN, VIS_HIDDEN, b_vb.pp("attn.proj"))?,
                norm2: layer_norm(VIS_HIDDEN, LayerNormConfig { eps: 1e-6, ..Default::default() }, b_vb.pp("norm2"))?,
                mlp_fc1: linear(VIS_HIDDEN, VIS_INTERMEDIATE, b_vb.pp("mlp.linear_fc1"))?,
                mlp_fc2: linear(VIS_INTERMEDIATE, VIS_HIDDEN, b_vb.pp("mlp.linear_fc2"))?,
            });
        }
        let merger = VisionMerger {
            norm: layer_norm(VIS_HIDDEN, LayerNormConfig { eps: 1e-6, ..Default::default() }, vb.pp("merger.norm"))?,
            linear_fc1: linear(VIS_MERGER_IN, VIS_MERGER_IN, vb.pp("merger.linear_fc1"))?,
            linear_fc2: linear(VIS_MERGER_IN, VIS_MERGER_OUT, vb.pp("merger.linear_fc2"))?,
        };
        Ok(Self { patch_embed, blocks, merger })
    }

    pub fn forward(&self, pixels: &Tensor) -> Result<Tensor> {
        let x = self.patch_embed.forward(pixels)?;
        let (b, c, h, w) = x.dims4()?;
        let mut x = x.reshape((b, c, h * w))?.transpose(1, 2)?;
        for block in &self.blocks {
            let residual = x.clone();
            let normed = block.norm1.forward(&x)?;
            let (bs, seq, _) = normed.dims3()?;
            let qkv = block.attn_qkv.forward(&normed)?;
            
            let mut q = qkv.i((.., .., ..VIS_HIDDEN))?.reshape((bs, seq, VIS_HEADS, VIS_HEAD_DIM))?.transpose(1, 2)?;
            let mut k = qkv.i((.., .., VIS_HIDDEN..VIS_HIDDEN * 2))?.reshape((bs, seq, VIS_HEADS, VIS_HEAD_DIM))?.transpose(1, 2)?;
            let v = qkv.i((.., .., VIS_HIDDEN * 2..))?.reshape((bs, seq, VIS_HEADS, VIS_HEAD_DIM))?.transpose(1, 2)?;

            // Vision RoPE
            let dev = x.device();
            let dtype = x.dtype();
            let v_rot_dim = VIS_HEAD_DIM / 2;
            let inv_freq: Vec<f32> = (0..v_rot_dim/2).map(|i| 1.0 / 10000.0f32.powf((i as f32 * 2.0) / v_rot_dim as f32)).collect();
            let inv_freq = Tensor::new(inv_freq, &dev)?;
            let t = Tensor::arange(0u32, seq as u32, &dev)?.to_dtype(DType::F32)?;
            let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
            let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
            let cos = emb.cos()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(dtype)?;
            let sin = emb.sin()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(dtype)?;
            
            // Vision RoPE is applied to each coordinate (2D)
            // Python: rotary_pos_emb = self.rot_pos_emb(grid_thw) -> (total_tokens, 24)
            // Then it's Cat'd with itself -> 48.
            // My simplified version: just use 1D RoPE over the sequence for now.
            q = apply_rope(&q, &cos, &sin)?;
            k = apply_rope(&k, &cos, &sin)?;

            let scale = 1.0 / (VIS_HEAD_DIM as f64).sqrt();
            let attn_w = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            let attn_w = candle_nn::ops::softmax(&attn_w, D::Minus1)?;
            let attn_out = attn_w.matmul(&v)?.transpose(1, 2)?.reshape((bs, seq, VIS_HIDDEN))?;
            let attn_out = block.attn_proj.forward(&attn_out)?;
            x = (attn_out + residual)?;
            let residual = x.clone();
            let normed = block.norm2.forward(&x)?;
            let mlp_out = block.mlp_fc1.forward(&normed)?.gelu()?;
            let mlp_out = block.mlp_fc2.forward(&mlp_out)?;
            x = (mlp_out + residual)?;
        }
        x = self.merger.norm.forward(&x)?;
        let (b, seq, _) = x.dims3()?;
        let x = x.reshape((b, seq / 4, VIS_MERGER_IN))?;
        let x = self.merger.linear_fc1.forward(&x)?;
        let x = x.gelu()?;
        self.merger.linear_fc2.forward(&x)
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, last_dim / 2)?;
    let x2 = x.narrow(D::Minus1, last_dim / 2, last_dim / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rope(t: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    t.broadcast_mul(cos)? + rotate_half(t)?.broadcast_mul(sin)?
}

struct FullAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: QwenRMSNorm,
    k_norm: QwenRMSNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl FullAttention {
    fn load(vb: VarBuilder, cfg: &Qwen35Config) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        let q_proj = linear_no_bias(hidden, cfg.num_attention_heads * head_dim * 2, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden, cfg.num_key_value_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden, cfg.num_key_value_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(cfg.num_attention_heads * head_dim, hidden, vb.pp("o_proj"))?;
        let q_norm = QwenRMSNorm::load(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = QwenRMSNorm::load(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self { q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, num_heads: cfg.num_attention_heads, num_kv_heads: cfg.num_key_value_heads, head_dim })
    }

    fn forward(&self, x: &Tensor, state: &mut LayerState) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let q_gate = self.q_proj.forward(x)?;
        let q_gate = q_gate.reshape((b, seq_len, self.num_heads, self.head_dim * 2))?;
        let q_raw = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate.narrow(D::Minus1, self.head_dim, self.head_dim)?.reshape((b, seq_len, self.num_heads * self.head_dim))?;
        let k_raw = self.k_proj.forward(x)?.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;
        let v_raw = self.v_proj.forward(x)?.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;
        
        let q_normed = self.q_norm.forward(&q_raw)?.transpose(1, 2)?;
        let k_normed = self.k_norm.forward(&k_raw)?.transpose(1, 2)?;
        let v = v_raw.transpose(1, 2)?;

        let (k_cache, v_cache) = match state {
            LayerState::Full(k, v) => (k.clone(), v.clone()),
            _ => {
                let dev = q_normed.device();
                (Tensor::zeros((b, self.num_kv_heads, 0, self.head_dim), q_normed.dtype(), dev)?,
                 Tensor::zeros((b, self.num_kv_heads, 0, self.head_dim), q_normed.dtype(), dev)?)
            }
        };

        let start_pos = k_cache.dim(2)?;
        let device = q_normed.device();
        let dtype = q_normed.dtype();
        let (_, _, s, _) = q_normed.dims4()?;
        
        // Qwen 3.5 RoPE: partial_rotary_factor=0.25, rope_theta=1e7
        let rotary_dim = (self.head_dim as f64 * 0.25) as usize;
        let inv_freq: Vec<f32> = (0..rotary_dim/2).map(|i| 1.0 / 10000000.0f32.powf((i as f32 * 2.0) / rotary_dim as f32)).collect();
        let inv_freq = Tensor::new(inv_freq, device)?;
        let t = (Tensor::arange(0u32, s as u32, device)?.to_dtype(DType::F32)? + start_pos as f64)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = emb.cos()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(dtype)?;
        let sin = emb.sin()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(dtype)?;

        // Apply RoPE to partial dimension
        let q_rot = q_normed.narrow(D::Minus1, 0, rotary_dim)?;
        let q_pass = q_normed.narrow(D::Minus1, rotary_dim, self.head_dim - rotary_dim)?;
        let q = Tensor::cat(&[&apply_rope(&q_rot, &cos, &sin)?, &q_pass], D::Minus1)?;

        let k_rot = k_normed.narrow(D::Minus1, 0, rotary_dim)?;
        let k_pass = k_normed.narrow(D::Minus1, rotary_dim, self.head_dim - rotary_dim)?;
        let k = Tensor::cat(&[&apply_rope(&k_rot, &cos, &sin)?, &k_pass], D::Minus1)?;

        let k = Tensor::cat(&[&k_cache, &k], 2)?;
        let v = Tensor::cat(&[&v_cache, &v], 2)?;
        *state = LayerState::Full(k.clone(), v.clone());

        let k_rep = if self.num_kv_heads != self.num_heads { repeat_interleave(&k, self.num_heads / self.num_kv_heads, 1)? } else { k };
        let v_rep = if self.num_kv_heads != self.num_heads { repeat_interleave(&v, self.num_heads / self.num_kv_heads, 1)? } else { v };

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k_rep.transpose(2, 3)?)? * scale)?;

        if seq_len > 1 {
            let mask_row = Tensor::arange(0u32, seq_len as u32, device)?.unsqueeze(0)?.broadcast_as((seq_len, seq_len))?;
            let mask_col = Tensor::arange(0u32, seq_len as u32, device)?.unsqueeze(1)?.broadcast_as((seq_len, seq_len))?;
            let causal_mask = mask_row.ge(&mask_col)?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(attn_weights.shape())?;
            attn_weights = causal_mask.where_cond(&attn_weights, &Tensor::full(f32::NEG_INFINITY, attn_weights.shape(), device)?)?;
        }

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let out = attn_weights.matmul(&v_rep)?.transpose(1, 2)?.reshape((b, seq_len, self.num_heads * self.head_dim))?;
        let out = (out * candle_nn::ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&out)
    }
}

fn l2norm(x: &Tensor) -> Result<Tensor> {
    let sum_sq = x.powf(2.0)?.sum_keepdim(D::Minus1)?;
    let inv_norm = (sum_sq + 1e-6)?.sqrt()?.recip()?;
    x.broadcast_mul(&inv_norm)
}

struct LinearAttention {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_a: Linear,
    in_proj_b: Linear,
    dt_bias: Tensor,
    a_log: Tensor,
    conv1d: candle_nn::Conv1d,
    out_proj: Linear,
    norm_weight: Tensor,
    norm_eps: f64,
    num_heads: usize,
    head_dim: usize,
}

impl LinearAttention {
    fn load(vb: VarBuilder, cfg: &Qwen35Config) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let num_heads = cfg.delta_num_heads;
        let head_dim = cfg.delta_head_dim;
        let conv_dim = head_dim * num_heads * 3;
        let in_proj_qkv = linear_no_bias(hidden, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_no_bias(hidden, num_heads * head_dim, vb.pp("in_proj_z"))?;
        let in_proj_a = linear_no_bias(hidden, num_heads, vb.pp("in_proj_a"))?;
        let in_proj_b = linear_no_bias(hidden, num_heads, vb.pp("in_proj_b"))?;
        let dt_bias = vb.get(num_heads, "dt_bias")?;
        let a_log = vb.get(num_heads, "A_log")?;
        let conv_w = vb.get((conv_dim, 1, 4), "conv1d.weight")?;
        let conv1d = candle_nn::Conv1d::new(conv_w, None, candle_nn::Conv1dConfig { groups: conv_dim, padding: 0, ..Default::default() });
        let out_proj = linear_no_bias(num_heads * head_dim, hidden, vb.pp("out_proj"))?;
        let norm_weight = vb.get(head_dim, "norm.weight")?;
        Ok(Self { in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, dt_bias, a_log, conv1d, out_proj, norm_weight, norm_eps: cfg.rms_norm_eps, num_heads, head_dim })
    }

    fn forward(&self, x: &Tensor, state: &mut LayerState) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let conv_dim = self.num_heads * self.head_dim * 3;
        
        if matches!(state, LayerState::None) || (seq > 1 && !matches!(state, LayerState::Linear(_, _))) {
            *state = LayerState::Linear(
                Tensor::zeros((b, self.num_heads, self.head_dim, self.head_dim), x.dtype(), x.device())?,
                Tensor::zeros((b, conv_dim, 3), x.dtype(), x.device())?,
            );
        }

        let (mut current_recurrent, mut current_conv) = match state {
            LayerState::Linear(r, c) => (r.clone(), c.clone()),
            _ => unreachable!(),
        };

        let mixed_qkv_raw = self.in_proj_qkv.forward(x)?;
        let mut mixed_qkv_t = mixed_qkv_raw.transpose(1, 2)?;
        
        if seq == 1 {
            let combined = Tensor::cat(&[&current_conv, &mixed_qkv_t], 2)?;
            current_conv = combined.narrow(2, 1, 3)?;
            mixed_qkv_t = self.conv1d.forward(&combined)?;
        } else {
            current_conv = mixed_qkv_t.narrow(2, seq - 3, 3)?;
            let zeros = Tensor::zeros((b, conv_dim, 3), mixed_qkv_t.dtype(), mixed_qkv_t.device())?;
            let padded = Tensor::cat(&[&zeros, &mixed_qkv_t], 2)?;
            mixed_qkv_t = self.conv1d.forward(&padded)?;
        }

        let mixed_qkv = candle_nn::ops::silu(&mixed_qkv_t)?.transpose(1, 2)?;
        let total_dim = self.num_heads * self.head_dim;
        let query = l2norm(&mixed_qkv.i((.., .., ..total_dim))?.reshape((b, (), self.num_heads, self.head_dim))?.transpose(1, 2)?)?;
        let key = l2norm(&mixed_qkv.i((.., .., total_dim..total_dim * 2))?.reshape((b, (), self.num_heads, self.head_dim))?.transpose(1, 2)?)?;
        let value = mixed_qkv.i((.., .., total_dim * 2..))?.reshape((b, (), self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let z = self.in_proj_z.forward(x)?;
        let b_gate = candle_nn::ops::sigmoid(&self.in_proj_b.forward(x)?)?;
        let a_gate = self.in_proj_a.forward(x)?;
        
        let dt = a_gate.broadcast_add(&self.dt_bias)?;
        let softplus_dt = (dt.exp()? + 1.0)?.log()?;
        let g = self.a_log.exp()?.neg()?.broadcast_mul(&softplus_dt.to_dtype(DType::F32)?)?.transpose(1, 2)?;
        
        let mut out_list = Vec::with_capacity(seq);
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q_scaled = (query * scale)?;

        for i in 0..seq {
            let q_t = q_scaled.i((.., .., i, ..))?;
            let k_t = key.i((.., .., i, ..))?;
            let v_t = value.i((.., .., i, ..))?;
            let g_t = g.i((.., .., i))?.exp()?.unsqueeze(2)?.unsqueeze(3)?;
            let beta_t = b_gate.i((.., i, ..))?.unsqueeze(2)?;

            current_recurrent = current_recurrent.broadcast_mul(&g_t)?;
            let kv_mem = k_t.unsqueeze(D::Minus2)?.matmul(&current_recurrent)?.squeeze(D::Minus2)?;
            let delta = (v_t - kv_mem)?.broadcast_mul(&beta_t)?;
            current_recurrent = (current_recurrent + k_t.unsqueeze(D::Minus1)?.matmul(&delta.unsqueeze(D::Minus2)?)?)?;
            out_list.push(q_t.unsqueeze(D::Minus2)?.matmul(&current_recurrent)?.squeeze(D::Minus2)?.unsqueeze(2)?);
        }

        *state = LayerState::Linear(current_recurrent, current_conv);
        let out = Tensor::cat(&out_list, 2)?.transpose(1, 2)?.reshape((b, seq, self.num_heads, self.head_dim))?;
        
        let out_f32 = out.to_dtype(DType::F32)?;
        let var = out_f32.powf(2.0)?.mean_keepdim(D::Minus1)?;
        let normed = out_f32.broadcast_mul(&(var + self.norm_eps)?.sqrt()?.recip()?)?;
        
        let out = normed.broadcast_mul(&self.norm_weight.to_dtype(DType::F32)?)?
            .broadcast_mul(&candle_nn::ops::silu(&z.reshape((b, seq, self.num_heads, self.head_dim))?)?.to_dtype(DType::F32)?)?;
        
        self.out_proj.forward(&out.to_dtype(x.dtype())?.reshape((b, seq, total_dim))?)
    }
}

enum AttentionLayer { Full(FullAttention), Linear(LinearAttention) }
struct MLP { gate_proj: Linear, up_proj: Linear, down_proj: Linear }
impl MLP {
    fn load(vb: VarBuilder, cfg: &Qwen35Config) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        Ok(Self { gate_proj: linear_no_bias(hidden, inter, vb.pp("gate_proj"))?, up_proj: linear_no_bias(hidden, inter, vb.pp("up_proj"))?, down_proj: linear_no_bias(inter, hidden, vb.pp("down_proj"))? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = (candle_nn::ops::silu(&self.gate_proj.forward(x)?)? * self.up_proj.forward(x)?)?;
        self.down_proj.forward(&out)
    }
}

struct Qwen35DecoderLayer { attn: AttentionLayer, mlp: MLP, input_layernorm: QwenRMSNorm, post_attention_layernorm: QwenRMSNorm }
impl Qwen35DecoderLayer {
    fn load(vb: VarBuilder, cfg: &Qwen35Config, layer_idx: usize) -> Result<Self> {
        let attn = if (layer_idx + 1) % 4 == 0 { AttentionLayer::Full(FullAttention::load(vb.pp("self_attn"), cfg)?) } else { AttentionLayer::Linear(LinearAttention::load(vb.pp("linear_attn"), cfg)?) };
        Ok(Self { attn, mlp: MLP::load(vb.pp("mlp"), cfg)?, input_layernorm: QwenRMSNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?, post_attention_layernorm: QwenRMSNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))? })
    }
    fn forward(&self, x: &Tensor, state: &mut LayerState) -> Result<Tensor> {
        let residual = x.clone();
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = match &self.attn { AttentionLayer::Full(a) => a.forward(&normed, state)?, AttentionLayer::Linear(a) => a.forward(&normed, state)? };
        let x = (attn_out + residual)?;
        let residual = x.clone();
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        mlp_out + residual
    }
}

pub struct Qwen35Model { embed_tokens: Embedding, vision_encoder: VisionEncoder, layers: Vec<Qwen35DecoderLayer>, norm: QwenRMSNorm, lm_head: Linear }
impl Qwen35Model {
    pub fn load(vb: VarBuilder, cfg: &Qwen35Config) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.language_model.embed_tokens"))?;
        let vision_encoder = VisionEncoder::load(vb.pp("model.visual"), cfg)?;
        let layers = (0..cfg.num_hidden_layers).map(|i| Qwen35DecoderLayer::load(vb.pp(&format!("model.language_model.layers.{}", i)), cfg, i)).collect::<Result<Vec<_>>>()?;
        let norm = QwenRMSNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.language_model.norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("model.language_model.embed_tokens"))?;
        Ok(Self { embed_tokens, vision_encoder, layers, norm, lm_head })
    }
    pub fn forward(&self, input_ids: Option<&Tensor>, pixel_values: Option<&Tensor>, states: &mut Vec<LayerState>) -> Result<Tensor> {
        let mut x = match (input_ids, pixel_values) {
            (Some(ids), Some(pixels)) => {
                let vis = self.vision_encoder.forward(pixels)?;
                let lang = self.embed_tokens.forward(ids)?;
                Tensor::cat(&[&vis, &lang], 1)?
            },
            (Some(ids), None) => self.embed_tokens.forward(ids)?,
            (None, Some(pixels)) => self.vision_encoder.forward(pixels)?,
            _ => candle_core::bail!("Need input"),
        };
        for (i, layer) in self.layers.iter().enumerate() { x = layer.forward(&x, &mut states[i])?; }
        x = self.norm.forward(&x)?;
        let last = x.dim(1)? - 1;
        self.lm_head.forward(&x.i((.., last, ..))?)
    }
}
