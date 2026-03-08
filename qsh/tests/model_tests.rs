use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use qsh::model::Config;
use qsh::model::Qwen3_5RmsNorm;

#[test]
fn test_model_config_default() {
    let config = Config::default();
    assert_eq!(config.text_config.hidden_size, 1024);
    assert_eq!(config.text_config.num_hidden_layers, 24);
    assert_eq!(config.head_dim(), 256);
}

#[test]
fn test_rms_norm() {
    let device = Device::Cpu;
    let dim = 1024;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // VarBuilder::get(dim, "weight") expects a tensor of size dim
    // Since it's from VarMap, it will be initialized if not present
    let rms_norm = Qwen3_5RmsNorm::new(dim, 1e-6, vb).expect("Failed to create RMSNorm");

    let input = Tensor::randn(0f32, 1f32, (1, 10, dim), &device).unwrap();
    let output = rms_norm.forward(&input).expect("Failed forward pass");

    assert_eq!(output.dims(), input.dims());
}
