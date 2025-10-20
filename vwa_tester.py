import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
import argparse
import requests
import threading
import torch
import subprocess
import multiprocessing
import queue
import time
from tqdm import tqdm
from PIL import Image, ImageChops
from pathlib import Path
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
    create_none_action,
)
from browser_env.utils import DetachedPage
from agent.agents import (
    PromptAgent,
    construct_agent,
)
from evaluation_harness.evaluators import evaluator_router
from evaluation_harness import image_utils
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
import test_utils as tu
from typing import Optional, Literal

class VWAConfig:
    # Environment
    render: bool = False
    render_screenshot: bool = True
    render_fail_only: bool = True
    slow_mo: int = 0
    action_set_tag: str = "id_accessibility_tree"
    observation_type: Literal[
        "accessibility_tree",
        "accessibility_tree_with_captioner",
        "html",
        "image",
        "image_som",
    ] = "accessibility_tree"
    current_viewport_only: bool = True
    viewport_width: int = 1280
    viewport_height: int = 2100
    sleep_after_execution: float = 0.0
    output_response: bool = False
    save_trace_enabled: bool = False

    # Task
    max_steps: int = 20
    single_site_mode = False
    flush: bool = False

    # Agent
    instruction_path: str = ""
    parsing_failure_th: int = 3
    repeating_action_failure_th: int = 4
    test_config_base_dir: Optional[str] = None

    # Captioning
    caption_model: str = ''

    # Language Model
    provider: str = "openai"
    eval_provider: str = ''
    model: str = "qwen25vl72b"
    mode: str = "som"
    temperature: float = 1.0
    top_p: float = 0.9
    context_length: int = 0
    max_tokens: int = 32768
    stop_token: Optional[str] = None
    vwa_code_path: Optional[str] = None
    vwa_data_path: Optional[str] = None
    domain: Optional[str] = None
    print_time: bool = False
    max_retry: int = 30
    max_obs_length: int = 3840

    # Example range
    test_start_idx: int = 0
    test_end_idx: int = 910

    test_config_files = []

class VWAPathHandler():
    args: VWAConfig
    dataset_result_path : str
    out_model_path : str
    output_path_model : str
    result_path : str
    traj_path : str
    render_dir : str
    cache_dir : str
    auth_dir: str
    config_dir : str
    metrics_path : str
    domain: str

    def __init__(self, args):
        self.args = args
        self.dataset_result_path = f'{args.vwa_data_path}/results/'
        u.mkdir(self.dataset_result_path)
        self.out_model_path = f'{args.model}_{args.mode}'
        self.output_path_model = f'{self.dataset_result_path}/{self.out_model_path}'
        u.mkdir(self.output_path_model)
        self.result_path = f'{self.output_path_model}/{args.domain}/'
        u.mkdir(self.result_path)
        self.traj_path = f'{self.result_path}/traj/'
        u.mkdir(self.traj_path)
        self.render_dir = f'{self.result_path}/render/'
        u.mkdir(self.render_dir)
        self.cache_dir = f'{self.result_path}/cache/'
        u.mkdir(self.cache_dir)
        self.auth_dir = f'{self.result_path}/auth/'
        u.mkdir(self.auth_dir)
        self.config_dir = f'{self.result_path}/config/'
        u.mkdir(self.config_dir)
        self.metrics_path = f'{self.result_path}/results/'
        u.mkdir(self.metrics_path)

    def reset_output_folder(self, folder_name, result_folder = None):
        if result_folder:
            self.dataset_result_path = f'{self.args.vwa_data_path}/{result_folder}/'
        self.output_path_model = f'{self.dataset_result_path}/{folder_name}'
        self.result_path = f'{self.output_path_model}/{self.args.domain}/'
        self.traj_path = f'{self.result_path}/traj/'
        self.render_dir = f'{self.result_path}/render/'
        self.cache_dir = f'{self.result_path}/cache/'
        self.auth_dir = f'{self.result_path}/auth/'
        self.config_dir = f'{self.result_path}/config/'
        self.metrics_path = f'{self.result_path}/results/'

class VWATester():
    def __init__(self, args: VWAConfig):
        self.args = args
        self.max_steps = args.max_steps
        self.print_time = self.args.print_time
        # self.domains = ['reddit', 'classifieds', 'shopping']
        # end_idxs = [210, 234, 466]
        if self.args.domain == None or self.args.domain == 'None':
            exit()

        self.kq_config = {
            'api_key': os.getenv("OPENROUTER_KEY"),
            'model': self.args.eval_provider,
            'base_url': os.getenv("BASE_URL"),
            'temperature': 0.0,
            'max_tokens': 4096,
        }

        self.early_stop_thresholds = {
            "parsing_failure": args.parsing_failure_th,
            "repeating_action": args.repeating_action_failure_th,
        }

        caption_image_fn = image_utils.get_captioning_fn(self.args.caption_model)
        self.eval_caption_image_fn = caption_image_fn

        self.agent = construct_agent(
            args,
            captioning_fn=caption_image_fn
        )  # NOTE: captioning_fn here is used for captioning input images.

        self.env = ScriptBrowserEnv(
            headless=True,
            slow_mo=args.slow_mo,
            observation_type=args.observation_type,
            current_viewport_only=args.current_viewport_only,
            viewport_size={
                "width": args.viewport_width,
                "height": args.viewport_height,
            },
            save_trace_enabled=True,
            sleep_after_execution=args.sleep_after_execution,
            # NOTE: captioning_fn here is used for LLM + captioning baselines.
            # This can be different from the captioning model used for evals.
            captioning_fn=caption_image_fn,
        )

        self.ph = VWAPathHandler(args)

    def handle_meta_bf(self, meta_data, trajectory):
        if self.args.mode == 'vision':
            try:
                bboxes = self.env.get_bboxes()
            except Exception as e:
                ERROR(e)
                meta_data['bbox'] = {}
                return meta_data

            meta_data['bbox'] = bboxes
            if len(trajectory) > 3:
                last_img_str = trajectory[-3]["observation"]["ori_image"]
                last_img = Image.fromarray(last_img_str)  # size = (viewport_width, viewport_width)
                curr_img_str = trajectory[-1]["observation"]["ori_image"]
                curr_img = Image.fromarray(curr_img_str)  # size = (viewport_width, viewport_width)
                f_same = tu.is_images_same(curr_img, last_img)
                if f_same and trajectory[-2]['action_info']['pred_action_type'] == 'SCROLL':
                    meta_data['hint'] += 'You have scrolled to the end of this page.'
            
            return meta_data
        elif self.args.mode == 'som':
            return meta_data
        else:
            return meta_data

    def handle_meta_af(self, meta_data, action, action_str):
        if self.args.mode == 'vision':
            last_action = action['action_info']['pred_action_description'] + ' ' + action['action_info']['pred_action']
            if action_str == "None" or action_str == 'none': 
                meta_data['hint'] = 'Last step you clicked an uninteractable area, you should try to change the element you click this time.'
            else:
                meta_data['hint'] = ''
            meta_data["action_history"].append(last_action)
        elif self.args.mode == 'som':
            raw_response = action['raw_prediction']
            try:
                if 'Let\'s think step-by-step. ' in raw_response:
                    key_content = u.extract_text(raw_response, 'Let\'s think step-by-step. ', ' In summary, the next action I will perform is')[0]
                else:
                    key_content = u.extract_text(raw_response, None, ' In summary, the next action I will perform is')[0]
                meta_data["action_history"].append(key_content)
            except Exception as e: 
                # ERROR(f'{e}, response format error')
                meta_data["action_history"].append(raw_response)
        else:
            last_action = action['action_info']['pred_action_description']
            meta_data["action_history"].append(last_action)
        return meta_data

    def auto_loging(self, _c, config_file):
        # automatically login
        if _c["storage_state"]:
            cookie_file_name = os.path.basename(_c["storage_state"])
            comb = get_site_comb_from_filepath(cookie_file_name)
            # temp_dir = tempfile.mkdtemp()
            # subprocess to renew the cookie
            print(f'auto login for {comb} ...')
            subprocess.run(
                [
                    "python",
                    f"{self.args.vwa_code_path}browser_env/auto_login.py",
                    "--auth_folder",
                    self.ph.auth_dir,
                    "--site_list",
                    *comb,
                ]
            )
            print('auto login done')
            _c["storage_state"] = f"{self.ph.auth_dir}/{cookie_file_name}"
            assert os.path.exists(_c["storage_state"])
            # update the config file
            config_file = f"{self.ph.config_dir}/{os.path.basename(config_file)}"
            u.write_json(config_file, _c)
            return config_file

    def load_input_image(self, image_paths):
        # Load input images for the task, if any.
        images = []
        if image_paths is not None:
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            for image_path in image_paths:
                # Load image either from the web or from a local path.
                if image_path.startswith("http"):
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                    input_image = Image.open(requests.get(image_path, stream=True, headers = headers).raw)
                else:
                    input_image = Image.open(image_path)
                input_image = tu.resize_image_proportional(input_image, 500, 500) # TODO
                images.append(input_image)
        return images

    def rollout(self, config_file, intent, images, render_helper = RenderHelper):
        self.agent.reset(config_file)
        trajectory: Trajectory = []
        try:
            obs, info = self.env.reset(options={"config_file": config_file})
        except Exception as e:
            ERROR(f'reset error {e}, waiting for re-setup')
            self.__reset()
            obs, info = self.env.reset(options={"config_file": config_file})

        state_info: StateInfo = {"observation": obs, "info": info}

        trajectory.append(state_info)
        meta_data = {"action_history": [], 'hint': '', 'tabs': obs['tabs']}

        # for step in range(self.max_steps):
        i_step = -1
        while 1:
            step_start_time = time.time()
            i_step += 1
            early_stop_flag, stop_info = tu.early_stop(
                trajectory, self.max_steps, self.early_stop_thresholds
            )

            if early_stop_flag:
                WARN('early stop')
                action = create_stop_action(f"Early stop: {stop_info}")
            else:
                meta_data = self.handle_meta_bf(meta_data, trajectory)
                meta_data["page"] = self.env.page
                # try:
                start_time = time.time()
                action = self.agent.next_action(
                    trajectory,
                    intent,
                    images=images,
                    meta_data=meta_data,
                    output_response=self.args.output_response)
                end_time = time.time()
                if self.print_time:
                    INFO(f'step {i_step} model infer time = {round(end_time - start_time, 2)}')
                # except Exception as e:
                #     # get the error message
                #     ERROR(e)
                #     action = create_stop_action(f"ERROR: {str(e)}")

            trajectory.append(action)
            observation_metadata = state_info["info"]["observation_metadata"]

            start_time = time.time()
            action_str = get_action_description(
                action,
                observation_metadata,
                action_set_tag = self.args.action_set_tag,
                prompt_constructor=self.agent.prompt_constructor
                if isinstance(self.agent, PromptAgent)
                else None,
            )
            end_time = time.time()
            if self.print_time:
                INFO(f'step {i_step} get action str time = {round(end_time - start_time, 2)}')

            meta_data = self.handle_meta_af(meta_data, action, action_str)

            if render_helper:
                start_time = time.time()
                render_helper.render(action, state_info, meta_data, self.args.render_screenshot)
                end_time = time.time()
                if self.print_time:
                    INFO(f'step {i_step} render time = {round(end_time - start_time, 2)}')

            if action["action_type"] == ActionTypes.STOP: break

            start_time = time.time()
            try:
                obs, _, terminated, _, info = self.env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)
                u.wait(1)
            except Exception as e:
                ERROR(f'execution error {e}')
                trajectory.append(create_stop_action(""))
                break

            end_time = time.time()
            if self.print_time:
                INFO(f'step {i_step} action exe time = {round(end_time - start_time, 2)}')

            if terminated:
                DEBUG(terminated)
                # add a action place holder
                trajectory.append(create_stop_action(""))
                break

            step_end_time = time.time()
            if self.print_time:
                INFO(f'step {i_step} total time = {round(step_end_time - step_start_time, 2)}')
        
        return trajectory

    def __find_last_ob(self, lst):
        result = None
        i = len(lst)
        for d in reversed(lst):
            i -= 1
            for key in d.keys():
                if "observation" in key:
                    result = d
                    break
            if result is not None:
                break
        return i

    def __reset(self):
        if self.args.domain == 'reddit':
            u.execute(f'bash {u.get_git()}/dataset/visualwebarena/scripts/reset_reddit.sh')
        elif self.args.domain == 'classifieds':
            u.execute(f'curl -X POST http://localhost:9980/index.php?page=reset -d "token=4b61655535e7ed388f0d40a93600254c"')
        elif self.args.domain == 'shopping':
            u.execute(f'bash {u.get_git()}/dataset/visualwebarena/scripts/reset_shopping.sh')

    def test(self):
        for config_file in tqdm(self.args.test_config_files, multiprocessing.current_process().name):
            _c = u.read_json(config_file)
            sites = _c['sites']
            if self.args.single_site_mode and len(sites) != 1: 
                WARN(f'{u.get_name(config_file)} is multi sites task: {sites}')
                continue
            intent = _c["intent"]
            task_id = _c["task_id"]
            result_file = f'{self.ph.metrics_path}/{task_id}.json'

            if u.is_file_exist(result_file) and not self.args.flush: 
                INFO('skip')
                continue
            if task_id < self.args.test_start_idx or task_id > self.args.test_end_idx: continue

            require_reset = _c['require_reset']
            if require_reset: self.__reset()

            u.write_json(f'{self.ph.config_dir}/{task_id}.json', vars(self.args))
            render_file = f'{self.ph.render_dir}/{task_id}.html'
            cache_config_file = self.auto_loging(_c, config_file)
            image_paths = _c.get("image", None)
            images = self.load_input_image(image_paths)

            traj_file = f'{self.ph.traj_path}/{task_id}.json'
            trajectory = []

            render_helper = None
            if self.args.render:
                render_helper = RenderHelper(_c, render_file, self.args.action_set_tag, images)
            
            i_retry = 0
            while i_retry < self.args.max_retry:
                trajectory = self.rollout(cache_config_file, intent, images, render_helper)
                if trajectory: break
                i_retry += 1
            if i_retry == self.args.max_retry: 
                return

            last_ob_idx = self.__find_last_ob(trajectory)
            for i in range(len(trajectory)):
                step = trajectory[i]
                if i == last_ob_idx:
                    last_page = trajectory[i]['info']['page']
                    trajectory[i]['info']['page'] = {'url': last_page.url, 'content': last_page.content}
                if 'observation' in step.keys() and i != last_ob_idx:
                    del trajectory[i]['info']['page']
                if 'observation' in step.keys():
                    del trajectory[i]['observation']
                if 'coords' in step.keys():
                    trajectory[i]['coords'] = trajectory[i]['coords'].tolist()
            u.write_json(traj_file, trajectory)

            last_page = self.env.page
            if render_helper: render_helper.close()

            eval_types = _c["eval"]["eval_types"]
            evaluator = evaluator_router(
                eval_types, 
                self.kq_config,
                captioning_fn = self.eval_caption_image_fn
            )
            start_time = time.time()
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=last_page
            )
            end_time = time.time()

            if self.args.save_trace_enabled:
                traj_file = f'{self.ph.traj_path}/{task_id}.zip'
                self.env.save_trace(traj_file)

            if self.args.render_fail_only and score: u.execute(f'rm {render_file}')
            if self.print_time:
                INFO(f'eval time = {round(end_time - start_time, 2)}, result = {score}')

            u.write_json(result_file, {task_id: score})

        self.env.close()
