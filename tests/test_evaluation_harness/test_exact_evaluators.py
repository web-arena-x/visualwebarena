import json
import os
import random
from glob import glob
from pathlib import Path
from typing import Any

import pytest
import requests
from PIL import Image
from py import test

from agent import Agent, TeacherForcingAgent
from browser_env import ActionTypes, ScriptBrowserEnv
from browser_env.env_config import *
from evaluation_harness import (
    HTMLContentExactEvaluator,
    PageImageEvaluator,
    StringEvaluator,
    URLExactEvaluator,
    image_utils,
)
from evaluation_harness.evaluators import EvaluatorComb

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
HEADLESS = True
config_file_folder = "tests/test_evaluation_harness/configs"


def tf_roll_out(
    agent: Agent, env: ScriptBrowserEnv, config_file: str
) -> list[Any]:
    """Roll out the agent using teacher forcing actions"""
    obs, state_info = env.reset(options={"config_file": config_file})

    trajectory: list[Any] = [{"observation": obs, "info": state_info}]
    while True:
        action = agent.next_action(
            trajectory=trajectory, intent="", meta_data={}
        )
        trajectory.append(action)
        if action["action_type"] == ActionTypes.STOP:
            break

        # preceed to next action
        obs, reward, terminated, truncated, info = env.step(action)
        state_info = {"observation": obs, "info": info}
        trajectory.append(state_info)

    return trajectory


def test_string_match_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/string_match.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = """page.stop("The date is 1985/04/18")"""
    agent.set_actions(action_seq)

    env = script_browser_env
    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = StringEvaluator()
    score = evalutor(
        trajectory, config_file, env.page
    )

    assert score == 1.0


def test_string_match_fail(script_browser_env: ScriptBrowserEnv) -> None:
    config_file = f"{config_file_folder}/string_match.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = """page.stop("The date is 1936/04/18")"""
    agent.set_actions(action_seq)

    env = script_browser_env
    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = StringEvaluator()
    score = evalutor(
        trajectory, config_file, env.page
    )

    assert score == 0.0


def test_url_exact_match_success(script_browser_env: ScriptBrowserEnv) -> None:
    config_file = f"{config_file_folder}/url_exact_match.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.goto("https://www.google.com/")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = URLExactEvaluator()
    score = evalutor(
        trajectory, config_file, env.page
    )
    assert score == 1.0


def test_url_exact_match_fail(script_browser_env: ScriptBrowserEnv) -> None:
    config_file = f"{config_file_folder}/url_exact_match.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.goto("https://github.com/web-arena-x")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = URLExactEvaluator()
    score = evalutor(
        trajectory, config_file, env.page
    )
    print(env.page.url)
    assert score == 0.0


def test_html_content_match_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/html_content_exact_match.json"

    # randomly sample a string
    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.goto("https://russmaxdesign.github.io/exercise")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = HTMLContentExactEvaluator()
    score = evalutor(
        trajectory, config_file, env.page
    )
    assert score == 1.0


def test_html_content_match_fail(script_browser_env: ScriptBrowserEnv) -> None:
    config_file = f"{config_file_folder}/html_content_exact_match.json"

    # randomly sample a string
    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = """page.goto("https://www.google.com/")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = HTMLContentExactEvaluator()
    score = evalutor(
        trajectory, config_file, env.page
    )
    assert score == 0.0


def test_html_content_element_match_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/html_content_element_exact_match.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.goto("https://russmaxdesign.github.io/exercise/")
    page.get_by_label("Full name").fill("Hello World")
    page.get_by_label("Email").click()
    page.get_by_label("Email").fill("alexisxy@hotmail.com")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = HTMLContentExactEvaluator()
    score = evalutor(
        trajectory, config_file, env.page
    )
    assert score == 1.0


def test_html_content_element_match_fail(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/html_content_element_exact_match.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.goto("https://russmaxdesign.github.io/exercise/")
    page.get_by_label("Full name").fill("Hello")
    page.get_by_label("Email").click()
    page.get_by_label("Email").fill("alexisxy@hotmail.com")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = HTMLContentExactEvaluator()
    score = evalutor(
        trajectory, config_file, env.page
    )
    assert score == 0.0


def test_html_content_url_comb_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/html_content_url_comb.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.goto("https://russmaxdesign.github.io/exercise/")
    page.get_by_label("Full name").fill("Hello World")
    page.get_by_label("Email").click()
    page.get_by_label("Email").fill("alexisxy@hotmail.com")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, config_file)

    evaluators = EvaluatorComb(
        [URLExactEvaluator(), HTMLContentExactEvaluator()]
    )
    score = evaluators(
        trajectory, config_file, env.page
    )
    assert score == 1.0


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Won't work using the demo sites"
)
def test_func_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/func_eval_success.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env
    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = HTMLContentExactEvaluator()
    score = evalutor(
        trajectory, config_file, env.page
    )
    assert score == 1.0


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Won't work using the demo sites"
)
def test_func_fail(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/func_eval_fail.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env
    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = HTMLContentExactEvaluator()
    score = evalutor(
        trajectory, config_file, env.page
    )
    assert score == 0.0


def test_func_url_func_last_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/func_url_func_1.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.goto("{REDDIT}/f/wallstreetbets/50431/-/comment/676875")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env
    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = HTMLContentExactEvaluator()
    score = evalutor(
        trajectory, config_file, env.page
    )
    assert score == 1.0


def test_html_required_values_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    for config_file in glob(
        f"{config_file_folder}/html_required_values_success_*.json"
    ):
        # change the URL placeholder with the concrete URL
        with open(config_file, "r") as f:
            configs = json.load(f)
            configs["eval"]["reference_url"] = configs["eval"][
                "reference_url"
            ].replace("__SHOPPING__", SHOPPING)
        tmp_config = config_file.replace(".json", ".json.tmp")
        with open(tmp_config, "w+") as f:
            json.dump(configs, f, indent=4)

        # randomly sample a string
        agent = TeacherForcingAgent()
        agent.set_action_set_tag(tag="playwright")
        gt_url = configs["eval"]["reference_url"]
        action_seq = f"""page.goto("{gt_url}")
        page.stop()"""
        agent.set_actions(action_seq)

        env = script_browser_env

        trajectory = tf_roll_out(agent, env, tmp_config)

        evalutor = HTMLContentExactEvaluator()
        score = evalutor(
            trajectory, tmp_config, env.page
        )
        os.remove(tmp_config)
        assert score == 1.0


def test_page_image_evaluator(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    for config_file in [
        f"{config_file_folder}/image_evaluator_yes.json",
        f"{config_file_folder}/image_evaluator_yes_direct_img.json",
    ]:
        # change the URL placeholder with the concrete URL
        with open(config_file, "r") as f:
            configs = json.load(f)
            configs["eval"]["reference_url"] = configs["eval"][
                "reference_url"
            ].replace("__SHOPPING__", SHOPPING)
            configs["start_url"] = configs["start_url"].replace(
                "__SHOPPING__", SHOPPING
            )
            for e in configs["eval"]["page_image_query"]:
                e["eval_image_url"] = e["eval_image_url"].replace("__SHOPPING__", SHOPPING)
        tmp_config = config_file.replace(".json", ".json.tmp")
        with open(tmp_config, "w+") as f:
            json.dump(configs, f, indent=4)

        # randomly sample a string
        agent = TeacherForcingAgent()
        agent.set_action_set_tag(tag="playwright")
        gt_url = configs["eval"]["page_image_query"][0]["eval_image_url"]
        action_seq = f"""page.goto("{gt_url}")
        page.stop()"""
        agent.set_actions(action_seq)

        env = script_browser_env

        trajectory = tf_roll_out(agent, env, tmp_config)

        # Create a dummy captioning function that always returns "yes"
        captioning_fn = lambda images, *args, **kwargs: ["yes"] * len(images)
        evalutor = PageImageEvaluator(captioning_fn)
        score = evalutor(
            trajectory, tmp_config, env.page
        )

        # Create a dummy captioning function that always returns "yes"
        captioning_fn = lambda images, *args, **kwargs: ["no"] * len(images)
        evalutor = PageImageEvaluator(captioning_fn)
        score_no = evalutor(
            trajectory, tmp_config, env.page
        )
        os.remove(tmp_config)
        assert score == 1.0
        assert score_no == 0.0


def test_page_image_evaluator_yes_no(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/image_evaluator_yes_no.json"
    # change the URL placeholder with the concrete URL
    with open(config_file, "r") as f:
        configs = json.load(f)
        configs["eval"]["reference_url"] = configs["eval"][
            "reference_url"
        ].replace("__SHOPPING__", SHOPPING)
        configs["start_url"] = configs["start_url"].replace(
            "__SHOPPING__", SHOPPING
        )
        for e in configs["eval"]["page_image_query"]:
            e["eval_image_url"] = e["eval_image_url"].replace("__SHOPPING__", SHOPPING)
    tmp_config = config_file.replace(".json", ".json.tmp")
    with open(tmp_config, "w+") as f:
        json.dump(configs, f, indent=4)

    # randomly sample a string
    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    gt_url = configs["eval"]["page_image_query"][0]["eval_image_url"]
    action_seq = f"""page.goto("{gt_url}")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, tmp_config)

    # Create a dummy captioning function that always returns "yes"
    captioning_fn = lambda images, *args, **kwargs: ["yes"] * len(images)
    evalutor = PageImageEvaluator(captioning_fn)
    score = evalutor(
        trajectory, tmp_config, env.page
    )
    assert score == 0.0


def test_html_required_values_failure(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    for config_file in glob(
        f"{config_file_folder}/html_required_values_failure_*.json"
    ):
        # change the URL placeholder with the concrete URL
        with open(config_file, "r") as f:
            configs = json.load(f)
            configs["eval"]["reference_url"] = configs["eval"][
                "reference_url"
            ].replace("__SHOPPING__", SHOPPING)
        tmp_config = config_file.replace(".json", ".json.tmp")
        with open(tmp_config, "w+") as f:
            json.dump(configs, f, indent=4)

        # randomly sample a string
        agent = TeacherForcingAgent()
        agent.set_action_set_tag(tag="playwright")
        gt_url = configs["eval"]["reference_url"]
        action_seq = f"""page.goto("{gt_url}")
        page.stop()"""
        agent.set_actions(action_seq)

        env = script_browser_env

        trajectory = tf_roll_out(agent, env, tmp_config)

        evalutor = HTMLContentExactEvaluator()
        score = evalutor(
            trajectory, tmp_config, env.page
        )
        os.remove(tmp_config)
        assert score == 0.0


def test_exact_image(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    for config_file, expected_score in zip(
        [
            f"{config_file_folder}/exact_image_success.json",
            f"{config_file_folder}/exact_image_failure.json",
            f"{config_file_folder}/exact_image_resize_failure_1.json",
            f"{config_file_folder}/exact_image_resize_failure_2.json",
        ],
        [1.0, 0.0, 0.0, 0.0],
    ):
        # change the URL placeholder with the concrete URL
        with open(config_file, "r") as f:
            configs = json.load(f)
            configs["start_url"] = configs["start_url"].replace(
                "__REDDIT__", REDDIT
            )
            configs["eval"]["reference_url"] = configs["eval"][
                "reference_url"
            ].replace("__REDDIT__", REDDIT)

            for e in configs["eval"]["page_image_query"]:
                e["eval_fuzzy_image_match"] = e["eval_fuzzy_image_match"].replace("__REDDIT__", REDDIT)
            configs["image"] = configs["image"].replace("__REDDIT__", REDDIT)
            configs["intent"] = configs["intent"].replace("__REDDIT__", REDDIT)
        tmp_config = config_file.replace(".json", ".tmp.json")
        with open(tmp_config, "w+") as f:
            json.dump(configs, f, indent=4)

        gt_url = configs["eval"]["reference_url"]
        agent = TeacherForcingAgent()
        agent.set_action_set_tag(tag="playwright")
        action_seq = f"""page.goto("{gt_url}")
        page.stop()"""
        agent.set_actions(action_seq)

        env = script_browser_env
        trajectory = tf_roll_out(agent, env, tmp_config)

        captioning_fn = lambda images, *args, **kwargs: ["yes"] * len(
            images
        )  # Unused for this task
        evalutor = PageImageEvaluator(captioning_fn)
        score = evalutor(
            trajectory, tmp_config, env.page
        )
        assert score == expected_score, config_file
        os.remove(tmp_config)


def test_exact_image_resize_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/exact_image_resize_template.json"
    resized_img_path = "resized_img.tmp.png"
    for downscale, expected_score in zip(
        [2, 8], [1.0, 0.0]
    ):  # 2x (should pass) and 8x (should fail) downscale
        # change the URL placeholder with the concrete URL
        with open(config_file, "r") as f:
            configs = json.load(f)
            configs["start_url"] = configs["start_url"].replace(
                "__REDDIT__", REDDIT
            )
            configs["eval"]["reference_url"] = configs["eval"][
                "reference_url"
            ].replace("__REDDIT__", REDDIT)
            for e in configs["eval"]["page_image_query"]:
                e["eval_fuzzy_image_match"] = e["eval_fuzzy_image_match"].replace("__REDDIT__", REDDIT)
            configs["image"] = configs["image"].replace("__REDDIT__", REDDIT)
            configs["intent"] = configs["intent"].replace("__REDDIT__", REDDIT)

            # Download an image and resize
            img_url = configs["instantiation_dict"]["image_url"].replace(
                "__REDDIT__", REDDIT
            )
            img = Image.open(requests.get(img_url, stream=True).raw)
            # Resize image to half its size
            resized_img = img.resize(
                (img.width // downscale, img.height // downscale)
            )
            with open(resized_img_path, "wb") as wf:
                resized_img.save(wf, format="png")

            configs["eval"]["page_image_query"][0]["eval_fuzzy_image_match"] = resized_img_path

        tmp_config = config_file.replace(".json", ".tmp.json")
        with open(tmp_config, "w+") as f:
            json.dump(configs, f, indent=4)

        gt_url = configs["eval"]["reference_url"]
        agent = TeacherForcingAgent()
        agent.set_action_set_tag(tag="playwright")
        action_seq = f"""page.goto("{gt_url}")
        page.stop()"""
        agent.set_actions(action_seq)

        env = script_browser_env
        trajectory = tf_roll_out(agent, env, tmp_config)

        captioning_fn = lambda images, *args, **kwargs: ["yes"] * len(
            images
        )  # Unused for this task
        evalutor = PageImageEvaluator(captioning_fn)
        score = evalutor(
            trajectory, tmp_config, env.page
        )
        assert score == expected_score, config_file
        os.remove(tmp_config)
        os.remove(resized_img_path)