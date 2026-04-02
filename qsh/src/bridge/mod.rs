use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod llamacpp;
pub mod python;
pub mod rust;

pub use llamacpp::LlamaCppBridge;
pub use python::PythonBridge;
pub use rust::RustBridge;

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

/// Strip `<think>...</think>` reasoning blocks from model output.
/// If the closing tag is missing (model hit token limit mid-think),
/// returns whatever text came before `<think>` rather than an empty string.
pub fn strip_think_tags(text: &str) -> String {
    if let Some(start) = text.find("<think>") {
        if let Some(end) = text.find("</think>") {
            text[end + 8..].trim().to_string()
        } else {
            // Model hit token limit mid-think; return whatever came before <think>
            text[..start].trim().to_string()
        }
    } else {
        text.to_string()
    }
}

/// Strip markdown code fences from model output.
/// Handles: ```lang\n...\n```, ```\n...\n```, and lone `backtick` wrapping.
pub fn strip_code_fence(text: &str) -> String {
    let text = text.trim();
    if text.starts_with("```") {
        let after_open = text.trim_start_matches('`');
        let body = if let Some(nl) = after_open.find('\n') {
            &after_open[nl + 1..]
        } else {
            after_open
        };
        let body = if let Some(close) = body.rfind("```") {
            body[..close].trim_end()
        } else {
            body.trim_end()
        };
        return body.to_string();
    }
    if text.starts_with('`') && text.ends_with('`') && text.len() > 2 {
        return text[1..text.len() - 1].to_string();
    }
    text.to_string()
}
