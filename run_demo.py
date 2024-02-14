"""Script to run end-to-end evaluation on the benchmark.

Modified from https://github.com/web-arena-x/webarena/blob/main/run.py.
"""
import argparse
import json
import logging
import os
import random
import time
import tempfile
from pathlib import Path

import openai
import requests
import torch
from beartype import beartype
from PIL import Image

from agent import (
    PromptAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import image_utils

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )

    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="som", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
        ],
        default="image_som",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agent/prompts/jsons/p_som_cot_id_actree_3s.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failures exceed this threshold, the agent will terminate early.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeated actions exceed this threshold, the agent will terminate early.",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--eval_captioning_model_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run eval captioning model on. By default, runs it on CPU.",
    )
    parser.add_argument(
        "--eval_captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl"],
        help="Captioning backbone for VQA-type evals.",
    )
    parser.add_argument(
        "--captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf"],
        help="Captioning backbone for accessibility tree alt text.",
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4-vision-preview")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=3840,
    )


    # example config
    parser.add_argument("--start_url", type=str, default="https://google.com")
    parser.add_argument("--intent", type=str, required=True)
    parser.add_argument("--image", type=str, default="", help="url of images, seperated by |AND|")

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type
        not in [
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "image_som",
        ]
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


@beartype
def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to stop early"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


@beartype
def test(
    args: argparse.Namespace,
    config_file: str
) -> None:
    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    caption_image_fn = None  # Don't use captioning for the demo, due to extra resources required to run BLIP-2.


    agent = construct_agent(
        args,
        captioning_fn=caption_image_fn
        if args.observation_type == "accessibility_tree_with_captioner"
        else None,
    )  # NOTE: captioning_fn here is used for captioning input images.

    assert args.render, "Rendering is required for end-to-end evaluation"

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        # NOTE: captioning_fn here is used for LLM + captioning baselines.
        # This can be different from the captioning model used for evals.
        captioning_fn=caption_image_fn,
    )

    try:
        render_helper = RenderHelper(
            config_file, args.result_dir, args.action_set_tag
        )

        # Load task.
        with open(config_file, 'r') as f:
            _c = json.load(f)
            intent = _c["intent"]
            image_paths = _c.get("image", None)
            images = []

            # Load input images for the task, if any.
            if image_paths is not None:
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                for image_path in image_paths:
                    # Load image either from the web or from a local path.
                    if image_path.startswith("http"):
                        input_image = Image.open(requests.get(image_path, stream=True).raw)
                    else:
                        input_image = Image.open(image_path)
                    
                    images.append(input_image)

        logger.info(f"[Config file]: {config_file}")
        logger.info(f"[Intent]: {intent}")

        agent.reset(config_file)
        trajectory: Trajectory = []
        obs, info = env.reset(options={"config_file": config_file})
        state_info: StateInfo = {"observation": obs, "info": info}
        trajectory.append(state_info)

        meta_data = {"action_history": ["None"]}
        while True:
            early_stop_flag, stop_info = early_stop(
                trajectory, max_steps, early_stop_thresholds
            )

            if early_stop_flag:
                action = create_stop_action(f"Early stop: {stop_info}")
            else:
                try:
                    print('=' * 30)
                    print('Agent: Thinking...')
                    action = agent.next_action(
                        trajectory,
                        intent,
                        images=images,
                        meta_data=meta_data,
                        output_response=True
                    )
                except ValueError as e:
                    # get the error message
                    action = create_stop_action(f"ERROR: {str(e)}")

            trajectory.append(action)

            action_str = get_action_description(
                action,
                state_info["info"]["observation_metadata"],
                action_set_tag=args.action_set_tag,
                prompt_constructor=agent.prompt_constructor
                if isinstance(agent, PromptAgent)
                else None,
            )
            render_helper.render(
                action, state_info, meta_data, args.render_screenshot
            )
            meta_data["action_history"].append(action_str)

            if action["action_type"] == ActionTypes.STOP:
                break

            obs, _, terminated, _, info = env.step(action)
            state_info = {"observation": obs, "info": info}
            trajectory.append(state_info)

            if terminated:
                # add a action place holder
                trajectory.append(create_stop_action(""))
                break

        if args.save_trace_enabled:
            env.save_trace(
                Path(args.result_dir) / "trace.zip"
            )
    except openai.OpenAIError as e:
        logger.info(f"[OpenAI Error] {repr(e)}")
    except Exception as e:
        logger.info(f"[Unhandled Error] {repr(e)}]")
        import traceback

        # write to error file
        with open(Path(args.result_dir) / "error.txt", "a") as f:
            f.write(f"[Config file]: {config_file}\n")
            f.write(f"[Unhandled Error] {repr(e)}\n")
            f.write(traceback.format_exc())  # write stack trace to file

    render_helper.close()

    env.close()


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


@beartype
def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    args.sleep_after_execution = 2.5
    prepare(args)

    _, tmp_config_file = tempfile.mkstemp(text=True)
    images_url = None
    if args.image:
        images_url = args.image.split('|AND|')
    with open(tmp_config_file, 'w') as f:
        json.dump({
            "task_id": 0,
          "start_url": args.start_url,
          "intent": args.intent,
          "image": images_url
        }, f)

    args.render_screenshot = True
    args.save_trace_enabled = True

    args.current_viewport_only = True
    dump_config(args)

    test(args, tmp_config_file)

    os.remove(tmp_config_file)
