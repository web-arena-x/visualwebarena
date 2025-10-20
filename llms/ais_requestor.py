import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
from tqdm import tqdm
from PIL import Image
import pandas as pd
import json
import io
import requests
import base64
from io import BytesIO
from openai import OpenAI

def get_git():
    current_file = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file)
    return current_folder + '/'

def pil_image_to_base64(img, format='png'):
    import PIL, PIL.JpegImagePlugin
    if isinstance(img, PIL.JpegImagePlugin.JpegImageFile): format = 'jpg'
    if format == 'jpg': format = 'JPEG'
    elif format == 'png': format = 'PNG'
    output_buffer = BytesIO()
    img.save(output_buffer, format=format)
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str

class AISRequestor():
    def __init__(self, config):
        if 'models' in config.keys():
            self.model = config['model']
        else:
            self.model = config['model']
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']
        self.client = OpenAI(
            api_key = config['api_key'],
            base_url = config['base_url'],
            timeout = 600000,
        )

    def infer(self, system, prompt, pil_img):
        messages = \
        [
            {
                'role': 'system', 
                'content': system
            }, 
            {
                'role': 'user', 
                'content': [
                    {
                        'type': 'text', 
                        'text': prompt
                    }, 
                ]
            }
        ]

        if pil_img:
            image_msg = {
                'type': 'image_url', 
                'image_url': { 
                    'url': 'data:image/png;base64,' 
                }
            }
            messages[1]['content'].append(image_msg)
            image_base64 = pil_image_to_base64(pil_img)
            messages[1]['content'][1]['image_url']['url'] += image_base64

        return self.infer_messages(messages)

    def infer_with_input_img(self, system, prompt, input_img, screenshot):
        messages = \
        [
            {
                'role': 'system', 
                'content': system
            }, 
            {
                'role': 'user', 
                'content': [
                    {
                        'type': 'text', 
                        'text': prompt
                    }, 
                    {
                        "type": "text",
                        "text": "IMAGES: (0) user input image"
                    },
                    {
                        'type': 'image_url', 
                        'image_url': { 
                            'url': 'data:image/png;base64,' 
                        }
                    },
                    {
                        "type": "text",
                        "text": "(1) current page screenshot"
                    },
                    {
                        'type': 'image_url', 
                        'image_url': { 
                            'url': 'data:image/png;base64,' 
                        }
                    }
                ]
            }
        ]
        input_image_base64 = pil_image_to_base64(input_img)
        messages[1]['content'][2]['image_url']['url'] += input_image_base64
        ss_image_base64 = pil_image_to_base64(screenshot)
        messages[1]['content'][4]['image_url']['url'] += ss_image_base64
        return self.infer_messages(messages)

    def infer_messages(self, messages) -> str:
        response = self.client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = self.temperature,
            max_tokens = self.max_tokens,
        )
        response = response.choices[0].message.content
        return response

def check_models(models):
    kq_config = {
        'api_key': os.getenv("AGI_API_KEY"),
        'model': '',
        'base_url': "https://agi-pre.alipay.com/api",
        'temperature': 0.0,
        'max_tokens': 4096,
    }
    for model in models:
        kq_config['model'] = model
        kq = AISRequestor(kq_config)
        prompt = ""
        image_file = f'{get_git()}/asset/unireco_bird_example.jpg'
        img = Image.open(image_file)
        try:
            response = kq.infer('', prompt, img)
            if isinstance(response, str): INFO(f'model {model} good')
        except Exception as e:
            DEBUG(f'model {model} {e}')
            return False
    return True

def get_lm_requestor(model):
    kq_config = {}
    if 'gpt' in model:
        kq_config = {
            'api_key': os.getenv("OPENROUTER_KEY"),
            'model': f'openai/{model}',
            'base_url': os.getenv("BASE_URL"),
            'temperature': 0.0,
            'max_tokens': 4096,
        }
    else:
        kq_config = {
            'api_key': os.getenv('AGI_API_KEY'),
            'model': model,
            'base_url': os.getenv("AGI_URL"),
            'temperature': 0.0,
            'max_tokens': 4096,
        }
    kq = AISRequestor(kq_config)
    return kq

if __name__ == "__main__":
    models = [
        'KevinBlip',
        'KevinBlip2'
    ]

    check_models(models)