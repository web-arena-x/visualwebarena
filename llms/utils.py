import argparse
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor, AutoProcessor
from peft import PeftModel
from PIL import Image
import ast

try:
    from vertexai.preview.generative_models import Image
    from llms import generate_from_gemini_completion
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image and llms.generate_from_gemini_completion')

from llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)

APIInput = str | list[Any] | dict[str, Any]

## Message template for yadong_model
SYSTEM_MESSAGE = '''You are an expert at completing instructions on Webpage screens. 
               You will be presented with a screenshot image with some numeric tags.
               If you decide to click somewhere, you should choose the numeric idx that is the closest to the location you want to click.  
               You should decide the action to continue this instruction.
               Here are the available actions:
{"action": "click", "action_natural_language": str, "idx": <element_idx chosen from the second screen>}
{"action": "hover", "action_natural_language": str, "idx": <element_idx chosen from the second screen>}
{"action": "enter", "action_natural_language": str, "idx": <element_idx chosen from the second screen>}
{"action": "type", "action_natural_language": str, "idx": <element_idx chosen from the second screen>, "value": <the text to enter>}
{"action": "select", "action_natural_language": str, "idx": <element_idx chosen from the second screen>, "value": <the option to select>}
Your final answer must be in the above format.
'''

def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
) -> str:
    response: str
    if lm_config.provider == "openai":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_openai_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
            )
        elif lm_config.mode == "completion":
            assert isinstance(prompt, str)
            response = generate_from_openai_completion(
                prompt=prompt,
                engine=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                max_tokens=lm_config.gen_config["max_tokens"],
                top_p=lm_config.gen_config["top_p"],
                stop_token=lm_config.gen_config["stop_token"],
            )
        else:
            raise ValueError(
                f"OpenAI models do not support mode {lm_config.mode}"
            )
    elif lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        response = generate_from_huggingface_completion(
            prompt=prompt,
            model_endpoint=lm_config.gen_config["model_endpoint"],
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            stop_sequences=lm_config.gen_config["stop_sequences"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
        )
    elif lm_config.provider == "google":
        assert isinstance(prompt, list)
        assert all(
            [isinstance(p, str) or isinstance(p, Image) for p in prompt]
        )
        response = generate_from_gemini_completion(
            prompt=prompt,
            engine=lm_config.model,
            temperature=lm_config.gen_config["temperature"],
            max_tokens=lm_config.gen_config["max_tokens"],
            top_p=lm_config.gen_config["top_p"],
        )
    elif lm_config.provider == "yadong_model":
        system_message = {
            'role': 'system',
            'content': SYSTEM_MESSAGE,
        }
        prompt_message = prompt
        model_id = "/home/gbassman/LAM/m2w_train_visible_ckpts_phi3.5_stage2_freeze"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            _attn_implementation='flash_attention_2'
        ).to('cuda')
        
        image_path = "/home/gbassman/LAM/webarena/webarena_image/current_image.png"
        image = Image.open(image_path)
        
        prompt = processor.tokenizer.apply_chat_template(
            [system_message, prompt_message], tokenize=False, add_generation_prompt=True
        )
        
        # Create a batch combining text and image inputs
        batch = processor(prompt, images=[image], return_tensors="pt").to('cuda')
        
        # Run inference
        with torch.no_grad():
            generated_ids = model.generate(**batch, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=64)
        
        generated_texts = processor.batch_decode(
            generated_ids[:, batch['input_ids'].size(1) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generated_dict = ast.literal_eval(generated_texts[0].strip().strip('.'))
        
        # Reformat to WebArena format
        response = generated_dict["action_natural_language"] + " ```"
        if generated_dict["action"] == "click":
            response += "click [" + str(generated_dict["idx"]) + "]"
        elif generated_dict["action"] == "hover":
            response += "hover [" + str(generated_dict["idx"]) + "]"
        elif generated_dict["action"] == "type":
            response += "type [" + str(generated_dict["idx"]) + "] [" + generated_dict["value"] + "] [1]"
        elif generated_dict["action"] == "stop":
            response += "stop [" + generated_dict["value"] + "]"
        
        response += "```"

    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )

    print("RESPONSE\n\n" + response + "\n\nEND RESPONSE")
    return response
