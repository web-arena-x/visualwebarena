import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ObservationMetadata,
    StateInfo,
    action2str,
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<head>
    <style>
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<html>
    <body>
     {body}
    </body>
</html>
"""


import re
from PIL import Image, ImageDraw, ImageFont

def add_text_top(
    img,
    text,
    font_path=None,
    font_size=36,
    text_color=(0, 0, 0),
    background_color="white",  # None 表示在 RGBA 模式下使用透明背景
    padding=(24, 24, 24, 24),  # (left, top, right, bottom)
    align="center",            # "left" | "center" | "right"
    line_spacing=1.3           # 行距系数
):
    # 统一模式，便于透明和非透明处理
    base_mode = "RGBA" if img.mode in ("RGBA", "LA") else "RGB"
    img_conv = img.convert(base_mode)
    W = img_conv.width

    # 字体
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        # 默认字体不一定支持中文，建议传入中文字体路径，如 NotoSansSC、SourceHanSans、微软雅黑等
        font = ImageFont.load_default()

    # 用于测量文字宽度
    draw_tmp = ImageDraw.Draw(img_conv)
    def measure(s):
        try:
            return draw_tmp.textlength(s, font=font)
        except Exception:
            bbox = font.getbbox(s)
            return bbox[2] - bbox[0]

    # 最大文本宽度
    left, top, right, bottom = padding
    max_text_width = W - left - right
    if max_text_width <= 0:
        raise ValueError("Padding 太大，已超过图片宽度。")

    # 自动换行：兼顾中英文，优先按词（空白分隔）换行，不可行时按字符切分
    def wrap(text):
        lines = []
        for para in text.split('\n'):
            if para == '':
                lines.append('')
                continue
            tokens = re.findall(r'\S+|\s+', para)  # 保留空白作为可能的断点
            current = ''
            for tok in tokens:
                test = current + tok
                if measure(test) <= max_text_width:
                    current = test
                else:
                    if current.strip() != '' or current != '':
                        lines.append(current.rstrip())
                        current = tok
                        # 若单个 token 仍超宽，按字符拆分
                        if measure(current) > max_text_width:
                            buf = ''
                            for ch in current:
                                t2 = buf + ch
                                if measure(t2) <= max_text_width:
                                    buf = t2
                                else:
                                    lines.append(buf.rstrip())
                                    buf = ch
                            current = buf
                    else:
                        # 极端情况：一开始就超宽，按字符拆分
                        buf = ''
                        for ch in tok:
                            t2 = buf + ch
                            if measure(t2) <= max_text_width:
                                buf = t2
                            else:
                                lines.append(buf.rstrip())
                                buf = ch
                        current = buf
            if current != '':
                lines.append(current.rstrip())
        return lines

    lines = wrap(text)

    # 行高计算
    try:
        ascent, descent = font.getmetrics()
        base_line_height = ascent + descent
    except Exception:
        b = font.getbbox("Ay")
        base_line_height = b[3] - b[1]
    line_h = int(base_line_height * line_spacing)

    # 计算需要的顶部空白高度
    text_block_height = top + bottom + (line_h * len(lines) if lines else 0)

    # 背景色处理
    if background_color is None:
        bg = (255, 255, 255, 0) if base_mode == "RGBA" else (255, 255, 255)
    else:
        bg = background_color

    # 生成新图（上方空白 + 原图）
    new_img = Image.new(base_mode, (W, img_conv.height + text_block_height), bg)
    new_img.paste(img_conv, (0, text_block_height))

    # 绘制文字
    draw = ImageDraw.Draw(new_img)
    y = top
    for line in lines:
        lw = measure(line)
        if align == "center":
            x = (W - lw) // 2
        elif align == "right":
            x = W - right - lw
        else:
            x = left
        draw.text((x, y), line, font=font, fill=text_color)
        y += line_h

    return new_img


def get_render_action(
    action: Action,
    observation_metadata: dict[str, ObservationMetadata],
    action_set_tag: str,
) -> str:
    """Parse the predicted actions for rendering purpose. More comprehensive information"""
    match action_set_tag:
        case "id_accessibility_tree":
            text_meta_data = observation_metadata["text"]
            if action["element_id"] in text_meta_data["obs_nodes_info"]:
                node_content = text_meta_data["obs_nodes_info"][
                    action["element_id"]
                ]["text"]
            else:
                node_content = "No match found"

            action_str = f"<div class='raw_parsed_prediction' style='background-color:grey'><pre>{action['raw_prediction']}</pre></div>"
            action_str += f"<div class='action_object' style='background-color:grey'><pre>{repr(action)}</pre></div>"
            action_str += f"<div class='parsed_action' style='background-color:yellow'><pre>{action2str(action, action_set_tag, node_content)}</pre></div>"

        case "som":
            text_meta_data = observation_metadata["text"]
            if action["element_id"] in text_meta_data["obs_nodes_info"]:
                node_content = text_meta_data["obs_nodes_info"][
                    action["element_id"]
                ]
            else:
                node_content = "No match found"

            action_str = f"<div class='raw_parsed_prediction' style='background-color:grey'><pre>{action['raw_prediction']}</pre></div>"
            action_str += f"<div class='action_object' style='background-color:grey'><pre>{repr(action)}</pre></div>"
            action_str += f"<div class='parsed_action' style='background-color:yellow'><pre>{action2str(action, action_set_tag, node_content)}</pre></div>"

        case "playwright":
            action_str = action["pw_code"]
        case _:
            raise ValueError(
                f"Unknown action type {action['action_type'], action_set_tag}"
            )
    return action_str


def get_action_description(
    action: Action,
    observation_metadata: dict[str, ObservationMetadata],
    action_set_tag: str,
    prompt_constructor: PromptConstructor | None,
) -> str:
    """Generate the text version of the predicted actions to store in action history for prompt use.
    May contain hint information to recover from the failures"""

    match action_set_tag:
        case "id_accessibility_tree":
            text_meta_data = observation_metadata["text"]
            if action["action_type"] in [
                ActionTypes.CLICK,
                ActionTypes.HOVER,
                ActionTypes.TYPE,
            ]:
                action_name = str(action["action_type"]).split(".")[1].lower()
                print(f'get_action_description in helper_functions.py: {action["element_id"]}')
                print(f'get_action_description in helper_functions.py: {text_meta_data["obs_nodes_info"]}')
                if action["element_id"] in text_meta_data["obs_nodes_info"]:
                    node_content = text_meta_data["obs_nodes_info"][
                        action["element_id"]
                    ]["text"]
                    node_content = " ".join(node_content.split()[1:])
                    action_str = action2str(
                        action, action_set_tag, node_content
                    )
                else:
                    action_str = f"Attempt to perfom \"{action_name}\" on element \"[{action['element_id']}]\" but no matching element found. Please check the observation more carefully."
            else:
                if (
                    action["action_type"] == ActionTypes.NONE
                    and prompt_constructor is not None
                ):
                    action_splitter = prompt_constructor.instruction[
                        "meta_data"
                    ]["action_splitter"]
                    action_str = f'The previous prediction you issued was "{action["raw_prediction"]}". However, the format was incorrect. Ensure that the action is wrapped inside a pair of {action_splitter} and enclose arguments within [] as follows: {action_splitter}action [arg] ...{action_splitter}.'
                else:
                    action_str = action2str(action, action_set_tag, "")

        case "som":
            text_meta_data = observation_metadata["image"]
            if action["action_type"] in [
                ActionTypes.CLICK,
                ActionTypes.HOVER,
                ActionTypes.TYPE,
            ]:
                action_name = str(action["action_type"]).split(".")[1].lower()
                if action["element_id"] in text_meta_data["obs_nodes_info"]:
                    action_str = action2str(action, action_set_tag, "")
                else:
                    print(
                        'action["element_id"], text_meta_data["obs_nodes_info"]',
                        action["element_id"],
                        text_meta_data["obs_nodes_info"],
                    )
                    action_str = f"Attempt to perfom \"{action_name}\" on element \"[{action['element_id']}]\" but no matching element found. Please check the observation more carefully."
            else:
                if (
                    action["action_type"] == ActionTypes.NONE
                    and prompt_constructor is not None
                ):
                    action_splitter = prompt_constructor.instruction[
                        "meta_data"
                    ]["action_splitter"]
                    action_str = f'The previous prediction you issued was "{action["raw_prediction"]}". However, the format was incorrect. Ensure that the action is wrapped inside a pair of {action_splitter} and enclose arguments within [] as follows: {action_splitter}action [arg] ...{action_splitter}.'
                else:
                    action_str = action2str(action, action_set_tag, "")

        case "playwright":
            action_str = action["pw_code"]

        case _:
            raise ValueError(f"Unknown action type {action['action_type']}")

    return action_str


class RenderHelper(object):
    """Helper class to render text and image observations and meta data in the trajectory"""

    def __init__(
        self, 
        _config,
        result_file: str,
        action_set_tag: str,
        input_images = [],
    ) -> None:
        _config_str = ""
        for k, v in _config.items():
            _config_str += f"{k}: {v}\n"
        _config_str = f"<pre>{_config_str}</pre>\n"
        if input_images:
            for input_image in input_images:
                byte_io = io.BytesIO()
                input_image.save(byte_io, format="PNG")
                byte_io.seek(0)
                image_bytes = base64.b64encode(byte_io.read())
                image_str = image_bytes.decode("utf-8")
                _config_str += f"<img src='data:image/png;base64,{image_str}' style='width:auto; height:auto;'/>\n"

        self.action_set_tag = action_set_tag
        self.render_file = open(result_file, "a+", encoding="utf-8")
        self.render_file.truncate(0)
        # write init template
        self.render_file.write(HTML_TEMPLATE.format(body=f"{_config_str}"))
        self.render_file.read()
        self.render_file.flush()

    def render(
        self,
        action: Action,
        state_info: StateInfo,
        meta_data: dict[str, Any],
        render_screenshot: bool = False,
    ) -> None:
        """Render the trajectory"""
        # text observation
        observation = state_info["observation"]
        text_obs = observation["text"]
        info = state_info["info"]
        new_content = f"<h2>New Page</h2>\n"
        new_content += f"<h3 class='url'><a href={state_info['info']['page'].url}>URL: {state_info['info']['page'].url}</a></h3>\n"
        # new_content += f"<div class='state_obv'><pre>{text_obs}</pre><div>\n"

        if render_screenshot:
            # image observation
            img_ori = observation['ori_image']
            image = Image.fromarray(img_ori)
            byte_io = io.BytesIO()
            image.save(byte_io, format="PNG")
            byte_io.seek(0)
            image_bytes = base64.b64encode(byte_io.read())
            image_str = image_bytes.decode("utf-8")
            new_content += f"<img src='data:image/png;base64,{image_str}' style='width:30vw; height:auto;'/>\n"

            img_obs = observation["image"]
            image = Image.fromarray(img_obs)
            if 'action_info' in action.keys():
                action_info = action['action_info']
                pred_action_history = action_info['pred_action_history']
                pred_action_description = action_info['pred_action_description']
                pred_action = action_info['pred_action']
                pred_action_type = action_info['pred_action_type']
                pred_bbox = action_info['pred_bbox']
                pred_type_value = action_info['pred_type_value']
                pred_click_point = action_info['pred_click_point']
                parse_error_msg = action_info['parse_error_msg']

            byte_io = io.BytesIO()
            image.save(byte_io, format="PNG")
            byte_io.seek(0)
            image_bytes = base64.b64encode(byte_io.read())
            image_str = image_bytes.decode("utf-8")
            new_content += f"<img src='data:image/png;base64,{image_str}' style='width:30vw; height:auto;'/>\n"

        if 'prompt' in action.keys():
            new_content += f"<div class='prompt' style='background-color:white'>{action['prompt']}</div>\n"

        # meta data
        if meta_data['action_history']:
            new_content += f"<div class='prev_action' style='background-color:pink'>{meta_data['action_history']}</div>\n"

        if state_info["observation"]:
            new_content += f"<div class='obs' style='background-color:white'>{state_info['observation']['text']}</div>\n"

        if 'tabs' in meta_data.keys():
            new_content += f"<div class='tabs' style='background-color:pink'>{meta_data['tabs']}</div>\n"

        # action
        action_str = get_render_action(
            action,
            info["observation_metadata"],
            action_set_tag=self.action_set_tag,
        )
        # with yellow background
        action_str = f"<div class='predict_action'>{action_str}</div>"
        new_content += f"{action_str}\n"

        # add new content
        self.render_file.seek(0)
        html = self.render_file.read()
        html_body = re.findall(r"<body>(.*?)</body>", html, re.DOTALL)[0]
        html_body += new_content

        html = HTML_TEMPLATE.format(body=html_body)
        self.render_file.seek(0)
        self.render_file.truncate()
        self.render_file.write(html)
        self.render_file.flush()

    def close(self) -> None:
        self.render_file.close()


class URLRenderHelper(object):
    """Helper class to render text and image observations and meta data in the trajectory"""

    def __init__(
        self, 
        result_file: str,
        action_set_tag: str,
        img_folder: str = '',
    ) -> None:
        _config_str = ""
        self.action_set_tag = action_set_tag
        self.render_file = open(result_file, "a+", encoding="utf-8")
        self.render_file.truncate(0)
        # write init template
        self.render_file.write(HTML_TEMPLATE.format(body=f"{_config_str}"))
        self.render_file.read()
        self.render_file.flush()
        self.img_folder = img_folder

    def render(
        self,
        action: Action,
        state_info: StateInfo,
        meta_data: dict[str, Any],
        render_screenshot: bool = False,
    ) -> None:
        """Render the trajectory"""
        # text observation
        observation = state_info["observation"]
        text_obs = observation["text"]
        info = state_info["info"]
        new_content = f"<h2>New Page</h2>\n"
        new_content += f"<h3 class='url'><a href={state_info['info']['page'].url}>URL: {state_info['info']['page'].url}</a></h3>\n"
        # new_content += f"<div class='state_obv'><pre>{text_obs}</pre><div>\n"

        if render_screenshot:
            # image observation
            img_ori = observation['ori_image']
            image = Image.fromarray(img_ori)
            byte_io = io.BytesIO()
            image.save(byte_io, format="PNG")
            byte_io.seek(0)
            image_bytes = base64.b64encode(byte_io.read())
            image_str = image_bytes.decode("utf-8")
            new_content += f"<img src='data:image/png;base64,{image_str}' style='width:30vw; height:auto;'/>\n"

            img_obs = observation["image"]
            image = Image.fromarray(img_obs)
            if 'action_info' in action.keys():
                action_info = action['action_info']
                pred_action_history = action_info['pred_action_history']
                pred_action_description = action_info['pred_action_description']
                pred_action = action_info['pred_action']
                pred_action_type = action_info['pred_action_type']
                pred_bbox = action_info['pred_bbox']
                pred_type_value = action_info['pred_type_value']
                pred_click_point = action_info['pred_click_point']
                parse_error_msg = action_info['parse_error_msg']

            byte_io = io.BytesIO()
            image.save(byte_io, format="PNG")
            byte_io.seek(0)
            image_bytes = base64.b64encode(byte_io.read())
            image_str = image_bytes.decode("utf-8")
            new_content += f"<img src='data:image/png;base64,{image_str}' style='width:30vw; height:auto;'/>\n"

        if 'prompt' in action.keys():
            new_content += f"<div class='prompt' style='background-color:white'>{action['prompt']}</div>\n"

        # meta data
        if meta_data['action_history']:
            new_content += f"<div class='prev_action' style='background-color:pink'>{meta_data['action_history']}</div>\n"

        if state_info["observation"]:
            new_content += f"<div class='obs' style='background-color:white'>{state_info['observation']['text']}</div>\n"

        if 'tabs' in meta_data.keys():
            new_content += f"<div class='tabs' style='background-color:pink'>{meta_data['tabs']}</div>\n"

        # action
        action_str = get_render_action(
            action,
            info["observation_metadata"],
            action_set_tag=self.action_set_tag,
        )
        # with yellow background
        action_str = f"<div class='predict_action'>{action_str}</div>"
        new_content += f"{action_str}\n"

        # add new content
        self.render_file.seek(0)
        html = self.render_file.read()
        html_body = re.findall(r"<body>(.*?)</body>", html, re.DOTALL)[0]
        html_body += new_content

        html = HTML_TEMPLATE.format(body=html_body)
        self.render_file.seek(0)
        self.render_file.truncate()
        self.render_file.write(html)
        self.render_file.flush()

    def save_img(
        self,
        intent,
        state_info,
        i_step,
        task_id,
    ) -> None:
        if task_id and i_step >= 0:
            observation = state_info["observation"]
            img_ori = observation['ori_image']
            image = Image.fromarray(img_ori)

            image = add_text_top(
                image,
                intent,
                font_path=f"{os.path.expanduser('~')}/data/font/Arial_Unicode.ttf",
                font_size=42,
                text_color=(0, 0, 0),
                background_color="white",
                padding=(40, 40, 40, 40),
                align="center",
                line_spacing=1.3
            )

            task_folder = f'{self.img_folder}/{task_id}'
            os.makedirs(task_folder, exist_ok=True)
            img_file = f'{task_folder}/{i_step}.png'
            image.save(img_file)

    def close(self) -> None:
        self.render_file.close()
