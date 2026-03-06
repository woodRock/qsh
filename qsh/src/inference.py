import sys
import torch
import json
import os
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load model and processor
model_name = os.getenv("QSH_MODEL", "Qwen/Qwen3.5-0.8B")
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

print(json.dumps({"info": f"Loading model {model_name} onto {device}..."}))
sys.stdout.flush()

# Load model to device
try:
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print(json.dumps({"info": "Model loaded successfully!"}))
    sys.stdout.flush()
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.stdout.flush()
    sys.exit(1)

def run_vision(image_path, query):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Question: {query} Answer YES or NO."},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Resize to prevent MPS memory overflow
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        min_pixels=256*28*28,
        max_pixels=512*28*28, 
    ).to(device)

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=10,
        pad_token_id=processor.tokenizer.eos_token_id,
        do_sample=False
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Clean up thinking tags
    if "<think>" in output_text:
        output_text = output_text.split("</think>")[-1]
    
    print(json.dumps({"info": f"Model response: {output_text.strip()}"}))
    sys.stdout.flush()
    
    return "YES" in output_text.upper()

def run_filter(line, query):
    messages = [
        {"role": "system", "content": "You are a text filter. Answer YES or NO."},
        {"role": "user", "content": f"Is the following line related to '{query}'?\nLine: {line}"}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=5,
        pad_token_id=processor.tokenizer.eos_token_id,
        do_sample=False
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    if "<think>" in output_text:
        output_text = output_text.split("</think>")[-1]
        
    print(json.dumps({"info": f"Model response: {output_text.strip()}"}))
    sys.stdout.flush()
    
    return "YES" in output_text.upper()

def run_bash(prompt):
    messages = [
        {"role": "system", "content": "You are a Unix shell expert. Provide the valid Bash command for the user's request. Output ONLY the command, no reasoning, no explanation."},
        {"role": "user", "content": prompt}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=128,
        pad_token_id=processor.tokenizer.eos_token_id,
        do_sample=False
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    if "<think>" in output_text:
        output_text = output_text.split("</think>")[-1]
        
    output_text = output_text.strip()
    if "```" in output_text:
        import re
        match = re.search(r"```(?:bash)?\n?(.*?)\n?```", output_text, re.DOTALL)
        if match:
            output_text = match.group(1).strip()
    elif "`" in output_text:
        output_text = output_text.replace("`", "").strip()
        
    return output_text

def run_explain(command):
    messages = [
        {"role": "system", "content": "You are a Unix shell expert."},
        {"role": "user", "content": f"Explain this Bash command briefly: {command}"}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=200,
        pad_token_id=processor.tokenizer.eos_token_id
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    if "<think>" in output_text:
        output_text = output_text.split("</think>")[-1]
        
    return output_text.strip()

if __name__ == "__main__":
    for line in sys.stdin:
        try:
            data = json.loads(line)
            mode = data.get("mode")
            
            if mode == "vision":
                query = data.get("query")
                path = data.get("path")
                result = run_vision(path, query)
                print(json.dumps({"result": result}))
            elif mode == "filter":
                query = data.get("query")
                text = data.get("text")
                result = run_filter(text, query)
                print(json.dumps({"result": result}))
            elif mode == "bash":
                prompt = data.get("prompt")
                result_text = run_bash(prompt)
                print(json.dumps({"text": result_text}))
            elif mode == "explain":
                cmd = data.get("command")
                result_text = run_explain(cmd)
                print(json.dumps({"text": result_text}))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()
