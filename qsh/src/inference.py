import sys
import torch
import json
import os
import warnings
import logging
import sqlite3

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, AutoProcessor, TrainingArguments, Trainer, TrainerCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset

# Load model and processor
model_name = os.getenv("QSH_MODEL", "Qwen/Qwen3.5-0.8B")
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

# LoRA Path
LORA_PATH = os.path.expanduser("~/.local/share/qsh/lora_weights")

print(json.dumps({"info": f"Loading model {model_name} onto {device}..."}))
sys.stdout.flush()

# Load model to device
try:
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
    )
    if os.path.exists(LORA_PATH) and os.path.exists(os.path.join(LORA_PATH, "adapter_config.json")):
        print(json.dumps({"info": "Loading LoRA weights..."}))
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model = model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print(json.dumps({"info": "Model loaded successfully!"}))
    sys.stdout.flush()
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.stdout.flush()
    sys.exit(1)

def run_lora(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Fetch successful interactions
        cursor.execute("""
            SELECT h1.content as prompt, h2.content as command
            FROM history h1
            JOIN history h2 ON h1.id = h2.id - 1
            WHERE h1.role = 'user' AND h2.role = 'assistant' AND h2.outcome = 'execute'
        """)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No successful history found to train on."

        print(json.dumps({"info": f"Found {len(rows)} successful interactions for training."}))
        
        dataset_data = {"prompt": [], "command": []}
        for prompt, command in rows:
            dataset_data["prompt"].append(prompt)
            dataset_data["command"].append(command)
        
        dataset = Dataset.from_dict(dataset_data)

        def tokenize_function(examples):
            prompts = examples["prompt"]
            commands = examples["command"]
            
            input_ids_list = []
            labels_list = []
            
            for p, c in zip(prompts, commands):
                messages = [
                    {"role": "system", "content": "You are a Unix shell expert. Provide the valid Bash command for the user's request. Output ONLY the command, no reasoning, no explanation."},
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": c}
                ]
                
                # Use the processor's chat template to stay consistent with inference
                # We tokenize the prompt part and the full text to find the split point
                prompt_messages = messages[:-1]
                prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                if not full_text.endswith(processor.tokenizer.eos_token):
                    full_text += processor.tokenizer.eos_token
                
                # Tokenize
                tokenized_prompt = processor.tokenizer(prompt_text, truncation=True, max_length=128, padding=False, add_special_tokens=False)
                tokenized_full = processor.tokenizer(full_text, truncation=True, max_length=128, padding=False, add_special_tokens=False)
                
                input_ids = tokenized_full["input_ids"]
                prompt_len = len(tokenized_prompt["input_ids"])
                
                # Mask out the prompt in labels
                labels = [-100] * len(input_ids)
                if len(input_ids) > prompt_len:
                    labels[prompt_len:] = input_ids[prompt_len:]
                
                input_ids_list.append(input_ids)
                labels_list.append(labels)
            
            # Manually pad because we have varying lengths
            max_len = max(len(ids) for ids in input_ids_list)
            padded_input_ids = []
            padded_labels = []
            padded_attention_mask = []
            
            for ids, labs in zip(input_ids_list, labels_list):
                padding_len = max_len - len(ids)
                padded_input_ids.append(ids + [processor.tokenizer.pad_token_id] * padding_len)
                padded_labels.append(labs + [-100] * padding_len)
                padded_attention_mask.append([1] * len(ids) + [0] * padding_len)

            return {
                "input_ids": padded_input_ids,
                "labels": padded_labels,
                "attention_mask": padded_attention_mask
            }

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

        lora_config = LoraConfig(
            r=8, 
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        global model
        # If it's already a PeftModel, start fresh for a new training session
        # Use get_base_model() to avoid merging old adapters into the base model (which causes drift)
        if isinstance(model, PeftModel):
            print(json.dumps({"info": "Discarding existing LoRA for fresh training..."}))
            model = model.get_base_model()

        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir="./lora_tmp",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            bf16=False, # Use float32 for stability on MPS if needed
            fp16=False,
            max_grad_norm=0.3,
        )

        class ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if state.max_steps > 0:
                    progress = (state.global_step / state.max_steps) * 100
                    print(json.dumps({"chunk": f"\rProgress: {progress:.1f}%"}))
                    sys.stdout.flush()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            callbacks=[ProgressCallback()]
        )

        trainer.train()
        model.save_pretrained(LORA_PATH)
        
        return "LoRA weights updated successfully!"
    except Exception as e:
        return f"Error during LoRA training: {str(e)}"

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

from transformers import TextIteratorStreamer
from threading import Thread

def run_bash(prompt, history=None):
    messages = [
        {"role": "system", "content": "You are a Unix shell expert. Provide the valid Bash command for the user's request. Output ONLY the command, no reasoning, no explanation."}
    ]
    if history:
        for role, content in history:
            messages.append({"role": role, "content": content})
    
    messages.append({"role": "user", "content": prompt})
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=1024,
        pad_token_id=processor.tokenizer.eos_token_id,
        do_sample=False
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    full_response = ""
    for new_text in streamer:
        full_response += new_text
        print(json.dumps({"chunk": new_text}))
        sys.stdout.flush()
    
    # Final cleanup after streaming
    if "<think>" in full_response:
        full_response = full_response.split("</think>")[-1]
        
    output_text = full_response.strip()
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
                history = data.get("history")
                result_text = run_bash(prompt, history)
                print(json.dumps({"text": result_text}))
            elif mode == "explain":
                cmd = data.get("command")
                result_text = run_explain(cmd)
                print(json.dumps({"text": result_text}))
            elif mode == "lora":
                db_path = data.get("path")
                result_text = run_lora(db_path)
                print(json.dumps({"text": result_text}))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()
