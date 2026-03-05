use candle_core::{Module, Result, Tensor, D, DType};
use candle_nn::{
    linear, linear_no_bias, Conv2d, Conv2dConfig, Embedding,
    Linear, VarBuilder,
};
fn manual_sigmoid(x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let output = (x_f32.neg()?.exp()? + 1.0)?.recip()?;
    output.to_dtype(dtype)
}

fn manual_silu(x: &Tensor) -> Result<Tensor> {
    let sig = manual_sigmoid(x)?;
    x.broadcast_mul(&sig)
}

fn manual_softmax(x: &Tensor, dim: D) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let max = x_f32.max_keepdim(dim)?;
    let exp = x_f32.broadcast_sub(&max)?.exp()?;
    let sum = exp.sum_keepdim(dim)?;
    let output = exp.broadcast_div(&sum)?;
    output.to_dtype(dtype)
}

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
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let norm_x = x_f32.powf(2.0)?.mean_keepdim(D::Minus1)?;
        let x_normed = x_f32.broadcast_mul(&(norm_x + self.eps)?.sqrt()?.recip()?)?;
        let w = self.weight.to_dtype(DType::F32)?;
        let output = x_normed.broadcast_mul(&(w + 1.0)?)?;
        output.to_dtype(dtype)
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
    let dims = t.dims();
    if dims.len() == 4 && dim == 1 {
        let (b, h, s, d) = t.dims4()?;
        t.unsqueeze(2)?.expand((b, h, repeats, s, d))?.reshape((b, h * repeats, s, d))
    } else {
        candle_core::bail!("repeat_interleave only implemented for 4D tensor and dim 1")
    }
}

#[derive(Debug, Clone)]
struct QwenLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl QwenLayerNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let bias = vb.get(dim, "bias")?;
        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let mean = x_f32.mean_keepdim(D::Minus1)?;
        let x_centered = x_f32.broadcast_sub(&mean)?;
        let var = x_centered.powf(2.0)?.mean_keepdim(D::Minus1)?;
        let x_norm = x_centered.broadcast_mul(&(var + self.eps)?.sqrt()?.recip()?)?;
        let weight_f32 = self.weight.to_dtype(DType::F32)?;
        let bias_f32 = self.bias.to_dtype(DType::F32)?;
        let output = x_norm.broadcast_mul(&weight_f32)?.broadcast_add(&bias_f32)?;
        output.to_dtype(dtype)
    }
}

pub struct VisionEncoder {
    patch_embed: Conv2d,
    blocks: Vec<VisionBlock>,
    merger: VisionMerger,
}

struct VisionBlock {
    norm1: QwenLayerNorm,
    attn_qkv: Linear,
    attn_proj: Linear,
    norm2: QwenLayerNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
}

struct VisionMerger {
    linear_fc1: Linear,
    linear_fc2: Linear,
    norm: QwenLayerNorm,
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
                norm1: QwenLayerNorm::load(VIS_HIDDEN, 1e-6, b_vb.pp("norm1"))?,
                attn_qkv: linear(VIS_HIDDEN, VIS_HIDDEN * 3, b_vb.pp("attn.qkv"))?,
                attn_proj: linear(VIS_HIDDEN, VIS_HIDDEN, b_vb.pp("attn.proj"))?,
                norm2: QwenLayerNorm::load(VIS_HIDDEN, 1e-6, b_vb.pp("norm2"))?,
                mlp_fc1: linear(VIS_HIDDEN, VIS_INTERMEDIATE, b_vb.pp("mlp.linear_fc1"))?,
                mlp_fc2: linear(VIS_INTERMEDIATE, VIS_HIDDEN, b_vb.pp("mlp.linear_fc2"))?,
            });
        }
        let merger = VisionMerger {
            norm: QwenLayerNorm::load(VIS_HIDDEN, 1e-6, vb.pp("merger.norm"))?,
            linear_fc1: linear(VIS_MERGER_IN, VIS_MERGER_IN, vb.pp("merger.linear_fc1"))?,
            linear_fc2: linear(VIS_MERGER_IN, VIS_MERGER_OUT, vb.pp("merger.linear_fc2"))?,
        };
        Ok(Self { patch_embed, blocks, merger })
    }

    pub fn forward(&self, pixels: &Tensor) -> Result<Tensor> {
        let x = self.patch_embed.forward(pixels)?;
        let (b, c, h, w) = x.dims4()?;
        let mut x = x.reshape((b, c, h * w))?.transpose(1, 2)?;
        let dev = x.device();
        let dtype = x.dtype();
        let (_, seq, _) = x.dims3()?;

        let inv_freq: Vec<f32> = (0..VIS_HEAD_DIM/2).map(|i| 1.0 / 10000.0f32.powf((i as f32 * 2.0) / VIS_HEAD_DIM as f32)).collect();
        let inv_freq = Tensor::new(inv_freq, &dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, seq as u32, &dev)?.to_dtype(dtype)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = emb.cos()?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = emb.sin()?.unsqueeze(0)?.unsqueeze(0)?;

        for block in &self.blocks {
            let residual = x.clone();
            let normed = block.norm1.forward(&x)?;
            let (bs, _, _) = normed.dims3()?;
            let qkv = block.attn_qkv.forward(&normed)?;
            
            let mut q = qkv.narrow(D::Minus1, 0, VIS_HIDDEN)?.reshape((bs, seq, VIS_HEADS, VIS_HEAD_DIM))?.transpose(1, 2)?;
            let mut k = qkv.narrow(D::Minus1, VIS_HIDDEN, VIS_HIDDEN)?.reshape((bs, seq, VIS_HEADS, VIS_HEAD_DIM))?.transpose(1, 2)?;
            let v = qkv.narrow(D::Minus1, VIS_HIDDEN * 2, VIS_HIDDEN)?.reshape((bs, seq, VIS_HEADS, VIS_HEAD_DIM))?.transpose(1, 2)?;

            q = apply_rope(&q, &cos, &sin)?;
            k = apply_rope(&k, &cos, &sin)?;

            let scale = 1.0 / (VIS_HEAD_DIM as f64).sqrt();
            let attn_w = q.matmul(&k.transpose(2, 3)?.contiguous()?)?.affine(scale, 0.0)?;
            let attn_w = manual_softmax(&attn_w, D::Minus1)?;
            let attn_out = attn_w.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((bs, seq, VIS_HIDDEN))?;
            let attn_out = block.attn_proj.forward(&attn_out)?;
            x = (attn_out + residual)?;
            
            let residual = x.clone();
            let normed = block.norm2.forward(&x)?;
            let mlp_out = manual_silu(&block.mlp_fc1.forward(&normed)?)?;
            let mlp_out = block.mlp_fc2.forward(&mlp_out)?;
            x = (mlp_out + residual)?;
        }
        x = self.merger.norm.forward(&x)?;
        let (b, seq, _) = x.dims3()?;
        let x = x.reshape((b, seq / 4, VIS_MERGER_IN))?;
        let x = self.merger.linear_fc1.forward(&x)?;
        let x = manual_silu(&x)?;
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
        
        let q_normed = self.q_norm.forward(&q_raw)?.transpose(1, 2)?.contiguous()?;
        let k_normed = self.k_norm.forward(&k_raw)?.transpose(1, 2)?.contiguous()?;
        let v = v_raw.transpose(1, 2)?.contiguous()?;

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
        
        let rotary_dim = (self.head_dim as f64 * 0.25) as usize;
        let inv_freq: Vec<f32> = (0..rotary_dim/2).map(|i| 1.0 / 10000000.0f32.powf((i as f32 * 2.0) / rotary_dim as f32)).collect();
        let inv_freq = Tensor::new(inv_freq, device)?.to_dtype(dtype)?;
        let t = (Tensor::arange(0u32, s as u32, device)?.to_dtype(DType::F32)? + start_pos as f64)?.to_dtype(dtype)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = emb.cos()?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = emb.sin()?.unsqueeze(0)?.unsqueeze(0)?;

        let q_rot = q_normed.narrow(D::Minus1, 0, rotary_dim)?;
        let q_pass = q_normed.narrow(D::Minus1, rotary_dim, self.head_dim - rotary_dim)?;
        let q = Tensor::cat(&[&apply_rope(&q_rot, &cos, &sin)?, &q_pass], D::Minus1)?.contiguous()?;

        let k_rot = k_normed.narrow(D::Minus1, 0, rotary_dim)?;
        let k_pass = k_normed.narrow(D::Minus1, rotary_dim, self.head_dim - rotary_dim)?;
        let k = Tensor::cat(&[&apply_rope(&k_rot, &cos, &sin)?, &k_pass], D::Minus1)?.contiguous()?;

        let k = Tensor::cat(&[&k_cache, &k], 2)?;
        let v = Tensor::cat(&[&v_cache, &v], 2)?;
        *state = LayerState::Full(k.clone(), v.clone());

        let k_rep = if self.num_kv_heads != self.num_heads { repeat_interleave(&k, self.num_heads / self.num_kv_heads, 1)? } else { k };
        let v_rep = if self.num_kv_heads != self.num_heads { repeat_interleave(&v, self.num_heads / self.num_kv_heads, 1)? } else { v };

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = q.matmul(&k_rep.transpose(2, 3)?.contiguous()?)?.affine(scale, 0.0)?;

        if seq_len > 1 {
            let mask_row = Tensor::arange(0u32, seq_len as u32, device)?.unsqueeze(0)?.broadcast_as((seq_len, seq_len))?;
            let mask_col = Tensor::arange(0u32, seq_len as u32, device)?.unsqueeze(1)?.broadcast_as((seq_len, seq_len))?;
            let causal_mask = mask_row.ge(&mask_col)?.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(attn_weights.shape())?;
            let neg_inf = Tensor::full(f32::NEG_INFINITY, attn_weights.shape(), device)?.to_dtype(dtype)?;
            attn_weights = causal_mask.where_cond(&attn_weights, &neg_inf)?;
        }

        let attn_weights = manual_softmax(&attn_weights, D::Minus1)?;
        let out = attn_weights.matmul(&v_rep.contiguous()?)?.transpose(1, 2)?.reshape((b, seq_len, self.num_heads * self.head_dim))?;
        let gate_val = manual_sigmoid(&gate)?;
        let out = out.broadcast_mul(&gate_val)?;
        self.o_proj.forward(&out)
    }
}

fn l2norm(x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let sum_sq = x_f32.powf(2.0)?.sum_keepdim(D::Minus1)?;
    let inv_norm = (sum_sq + 1e-6)?.sqrt()?.recip()?;
    x_f32.broadcast_mul(&inv_norm)?.to_dtype(dtype)
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
        let dt_bias = vb.get(num_heads, "dt_bias")?.to_dtype(vb.dtype())?;
        let a_log = vb.get(num_heads, "A_log")?.to_dtype(vb.dtype())?;
        let conv_w = vb.get((conv_dim, 1, 4), "conv1d.weight")?.to_dtype(vb.dtype())?;
        let conv1d = candle_nn::Conv1d::new(conv_w, None, candle_nn::Conv1dConfig { groups: conv_dim, padding: 0, ..Default::default() });
        let out_proj = linear_no_bias(num_heads * head_dim, hidden, vb.pp("out_proj"))?;
        let norm_weight = vb.get(head_dim, "norm.weight")?.to_dtype(vb.dtype())?;
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

        let mixed_qkv = manual_silu(&mixed_qkv_t)?.transpose(1, 2)?;
        let total_dim = self.num_heads * self.head_dim;
        let query = l2norm(&mixed_qkv.narrow(D::Minus1, 0, total_dim)?.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?)?;
        let key = l2norm(&mixed_qkv.narrow(D::Minus1, total_dim, total_dim)?.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?)?;
        let value = mixed_qkv.narrow(D::Minus1, total_dim * 2, total_dim)?.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let z = self.in_proj_z.forward(x)?;
        let b_gate = manual_sigmoid(&self.in_proj_b.forward(x)?)?;
        let a_gate = self.in_proj_a.forward(x)?;
        
        let dt = a_gate.broadcast_add(&self.dt_bias)?;
        let softplus_dt = (dt.exp()? + 1.0)?.log()?;
        let g = self.a_log.to_dtype(DType::F32)?.exp()?.neg()?.broadcast_mul(&softplus_dt.to_dtype(DType::F32)?)?.transpose(1, 2)?.to_dtype(x.dtype())?;
        
        let mut out_list = Vec::with_capacity(seq);
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q_scaled = query.affine(scale, 0.0)?;

        for i in 0..seq {
            let q_t = q_scaled.narrow(2, i, 1)?.squeeze(2)?;
            let k_t = key.narrow(2, i, 1)?.squeeze(2)?;
            let v_t = value.narrow(2, i, 1)?.squeeze(2)?;
            let g_t = g.narrow(2, i, 1)?.exp()?.unsqueeze(3)?;
            let beta_t = b_gate.narrow(1, i, 1)?.squeeze(1)?.unsqueeze(2)?;

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
            .broadcast_mul(&manual_silu(&z.reshape((b, seq, self.num_heads, self.head_dim))?)?.to_dtype(DType::F32)?)?;
        
        self.out_proj.forward(&out.to_dtype(x.dtype())?.reshape((b, seq, total_dim))?)
    }
}

enum AttentionLayer { Full(FullAttention), Linear(LinearAttention) }
struct MLP { gate_proj: Linear, up_proj: Linear, down_proj: Linear }
impl MLP {
    fn load(vb: VarBuilder, cfg: &Qwen35Config) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        Ok(Self { 
            gate_proj: linear_no_bias(hidden, inter, vb.pp("gate_proj"))?, 
            up_proj: linear_no_bias(hidden, inter, vb.pp("up_proj"))?, 
            down_proj: linear_no_bias(inter, hidden, vb.pp("down_proj"))? 
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = manual_silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let out = gate.broadcast_mul(&up)?;
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
                let dev = ids.device();
                let vis = self.vision_encoder.forward(pixels)?;
                let lang = self.embed_tokens.forward(ids)?;
                let v_start = self.embed_tokens.forward(&Tensor::new(&[248053u32], dev)?.unsqueeze(0)?)?;
                let v_end = self.embed_tokens.forward(&Tensor::new(&[248054u32], dev)?.unsqueeze(0)?)?;
                Tensor::cat(&[&v_start, &vis, &v_end, &lang], 1)?
            },
            (Some(ids), None) => self.embed_tokens.forward(ids)?,
            (None, Some(pixels)) => {
                let dev = pixels.device();
                let vis = self.vision_encoder.forward(pixels)?;
                let v_start = self.embed_tokens.forward(&Tensor::new(&[248053u32], dev)?.unsqueeze(0)?)?;
                let v_end = self.embed_tokens.forward(&Tensor::new(&[248054u32], dev)?.unsqueeze(0)?)?;
                Tensor::cat(&[&v_start, &vis, &v_end], 1)?
            },
            _ => candle_core::bail!("Need input"),
        };
        for (i, layer) in self.layers.iter().enumerate() { x = layer.forward(&x, &mut states[i])?; }
        x = self.norm.forward(&x)?;
        let last = x.dim(1)? - 1;
        self.lm_head.forward(&x.narrow(1, last, 1)?.squeeze(1)?)
    }
}
