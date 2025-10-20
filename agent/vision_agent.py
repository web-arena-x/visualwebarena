import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
import json
import importlib
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
from llms import (
    # call_llm,
    # generate_from_huggingface_completion,
    # generate_from_openai_chat_completion,
    # generate_from_openai_completion,
    lm_config,
)
from llms.tokenizers import Tokenizer
from prompts.prompt_constructor import MultimodalCoTPromptConstructor, MultimodalCoTPromptConstructor
from llms.ais_requestor import get_lm_requestor
from base_agent import Agent

class VisionAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor | MultimodalCoTPromptConstructor | MultimodalCoTPromptConstructor,
        captioning_fn = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn
        self.multimodal_inputs = True
        self.kq = get_lm_requestor(self.lm_config.model)

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def map_to_vwa_actions(self, action_info):
        pred_action_history = action_info['pred_action_history']
        pred_action_description = action_info['pred_action_description']
        pred_action = action_info['pred_action']
        pred_action_type = action_info['pred_action_type']
        pred_bbox = action_info['pred_bbox']
        pred_type_value = action_info['pred_type_value']
        pred_click_point = action_info['pred_click_point']
        parse_error_msg = action_info['parse_error_msg']
        return pred_action

    def filter_bboxes(self, bboxes, w, h):
        to_be_del = []
        for idx, box in bboxes.items():
            if box['x'] < 0 or box['x'] > w or box['y'] < 0 or box['y'] > h:
                to_be_del.append(idx)
        for idx in to_be_del:
            del bboxes[idx]
        return bboxes

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
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["ori_image"]
            page_screenshot_img = Image.fromarray(page_screenshot_arr)  # size = (viewport_width, viewport_width)

        bboxes = meta_data['bbox']

        n = 0
        while True:
            # task, action history, memo, tabs, hint
            prompt = prompt_f.format(intent, meta_data["action_history"], '', '', meta_data['hint'])
            if images:
                input_img = images[0]
                response = self.kq.infer_with_input_img(
                    sys_prompt, 
                    prompt,
                    input_img, 
                    page_screenshot_img)
            else:
                response = self.kq.infer(
                    sys_prompt,
                    prompt,
                    page_screenshot_img)

            action_info = parse_response(response, 'bbox')
            force_prefix = self.prompt_constructor.instruction["meta_data"].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            if output_response: print(f'{response}', flush=True)

            n += 1
            try:
                action = create_vision_action(action_info, bboxes)
                action["raw_prediction"] = response
                action['action_info'] = action_info
                break
            except ActionParsingError as e:
                INFO(e)
                if n >= self.lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break

        action['prompt'] = prompt
        action['action_info'] = action_info
        return action

    def reset(self, test_config_file: str) -> None:
        pass

