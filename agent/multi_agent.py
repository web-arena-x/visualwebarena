import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
from typing import Any, Optional
from beartype import beartype
from PIL import Image
from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_vision_action,
    create_mas_action,
)
from browser_env.utils import Observation, StateInfo
from llms import lm_config
from llms.tokenizers import Tokenizer
from prompts.prompt_constructor import MultimodalCoTPromptConstructor, MultimodalCoTPromptConstructor
from base_agent import Agent
from AWorld.examples.visualwebarena.action_team import ActionTeam

class MultiAgent(Agent):
    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor | 
        MultimodalCoTPromptConstructor | 
        MultimodalCoTPromptConstructor,
        captioning_fn = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn
        self.multimodal_inputs = True
        self.bu_agent = ActionTeam(self.lm_config.model)

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def next_action(
        self, 
        trajectory: Trajectory, 
        intent: str, 
        meta_data: dict[str, Any], 
        images: Optional[list[Image.Image]] = None,
        output_response: bool = False
    ) -> Action:
        # Create page screenshot image for multimodal models.
        ori_page_screenshot_arr = trajectory[-1]["observation"]["ori_image"]
        ori_page_screenshot_img = Image.fromarray(ori_page_screenshot_arr)
        som_page_screenshot_arr = trajectory[-1]["observation"]["image"]
        som_page_screenshot_img = Image.fromarray(som_page_screenshot_arr)

        last_som_img = None
        last_ori_img = None
        for i in range(-2, -len(trajectory), -1):
            if 'observation' in trajectory[i].keys():
                last_som_img = trajectory[i]['observation']['image']
                last_som_img = Image.fromarray(last_som_img)
                last_ori_img = trajectory[i]['observation']['ori_image']
                last_ori_img = Image.fromarray(last_ori_img)
                break

        # Caption the input image, if provided.
        image_input_caption = ''
        input_img = None
        try:
            if images is not None and len(images) > 0:
                if self.captioning_fn is not None:
                    for image_i, image in enumerate(images):
                        if image_i == 0:
                            image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                        else:
                            image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                        if len(images) > 1:
                            image_input_caption += ", "
                    # Update intent to include captions of input images.
                    # intent = f"{image_input_caption}\nTask: {intent}"
                elif not self.multimodal_inputs:
                    print("WARNING: Input image provided but no image captioner available.")
                input_img = images[0]
        except Exception as e:
            ERROR(f'caption function {self.lm_config.caption_model} {e}')
        
        page_text = trajectory[-1]['observation']['text']
        page_texts = page_text.split('\n')
        page_texts = [a.replace('  ', '') for a in page_texts if not a.startswith('[]')]
        page_text = '\n'.join(page_texts)

        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs = state_info["observation"]['text']
        page = meta_data["page"]
        url = page.url
        action_history = meta_data["action_history"]
        tabs = meta_data['tabs']
        site_name = self.lm_config.domain

        action_info, response = self.bu_agent.next_action(output_response,
            intent, action_history, site_name, url, obs, tabs, input_img, som_page_screenshot_img, ori_page_screenshot_img, page)
        force_prefix = self.prompt_constructor.instruction["meta_data"].get("force_prefix", "")
        response = f"{force_prefix}{response}"

        try:
            parsed_response = self.prompt_constructor.extract_action(response)
            action = create_mas_action(parsed_response, obs)
            action["raw_prediction"] = response
        except Exception as e:
            action = create_none_action()
            action["raw_prediction"] = response
        
        action['action_info'] = action_info
        action['domain'] = self.lm_config.domain
        return action

    def reset(self, test_config_file: str) -> None:
        self.called_closing_agents = []