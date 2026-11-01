use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod python;

pub use python::PythonBridge;

#[derive(Serialize)]
pub struct InferenceRequest {
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history: Option<Vec<(String, String)>>,
}

#[derive(Deserialize, Debug)]
pub struct InferenceResponse {
    pub result: Option<bool>,
    pub text: Option<String>,
    pub error: Option<String>,
    pub info: Option<String>,
    pub chunk: Option<String>,
}

pub trait Bridge {
    fn query_bool(&mut self, request: &InferenceRequest) -> Result<bool>;
    fn query_text(&mut self, request: &InferenceRequest, stream: bool) -> Result<String>;
}
