import json
import re

import requests
from openai import AsyncOpenAI

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 4096 * 2 * 28 * 28


def get_upload_image_prompt(upload_img_paths: list | str) -> str:
    """Get the prompt for uploading an image."""
    if isinstance(upload_img_paths, str):
        upload_img_paths = [upload_img_paths]
    img_path_text = ""
    for i, img_path in enumerate(upload_img_paths):
        img_path_text += f'Picture {i} path: "{img_path}"\n'

    question_prefix = f"I have upload the following images:\n{img_path_text}\n\n"
    return question_prefix


def query_template(question: str, sandbox_image_path: list[str] | str, need_picture_text: bool = True):
    if isinstance(sandbox_image_path, str):
        sandbox_image_path = [sandbox_image_path]
    image_text = ''.join([f'Picture {i}: <image>\n' for i in range(len(sandbox_image_path))]) if need_picture_text else ""
    return f"""{get_upload_image_prompt(sandbox_image_path)}{image_text}
Now please answer the following question:
{question}
Please answer in the following format:
<think>...</think>
<answer>...</answer>""".strip()



def run_jupyter_code(cell_list, sandbox_url, upload_file_dict=None):
    response = requests.post(
        f"{sandbox_url}/run_jupyter",
        json={
            "cells": cell_list,
            "kernel": "python3",
            "files": upload_file_dict,
            "total_timeout": 22,
        },
    )
    output_cells = response.json().get("cells", [])
    if not output_cells:
        raise ValueError(f"No output cells returned from Jupyter execution. Cell List: {cell_list}")

    return output_cells



def parse_cell_output(cell_output: dict) -> dict:
    """Parse the output of a Jupyter cell."""
    if not cell_output:
        return {
            "text_output": "",
            "image_output": [],
            "has_error": False,
        }
    stdout = cell_output.get("stdout", "")

    errors = cell_output.get("error", "")
    error_message = ""
    for e in errors:
        # traceback = e.get("traceback", "")
        e_name = e.get("ename", "")
        e_value = e.get("evalue", "")
        error_message = f"[CODE RUN ERROR]: {e_name} - {e_value}\n\nPlease read the bug information and fix it to continue solve the question. Hint: All variables in this cell can not be used in the next cell."
        if len(error_message) > 3000:
            error_message = error_message[:1500] + "..." + error_message[-1500:]
    # show display output
    display_output = cell_output.get("display", [])
    display_text = ""
    display_image = []
    for cell_output_item in display_output:
        for key, value in cell_output_item.items():
            if key == "text/plain":
                display_text += value
            elif key == "image/png":
                display_image.append(f"data:image/png;base64,{value}")
            elif key == "image/jpeg":
                display_image.append(f"data:image/jpeg;base64,{value}")
            else:
                print(f"Unknown key: {key}")
    text_output = ""
    if stdout:
        text_output += f"stdout: {stdout}\n"
    if display_text:
        text_output += f"display text: {display_text}\n"
    if error_message:
        text_output += f"error: {error_message}\n"

    image_output = display_image
    return {
        "text_output": text_output.strip(),
        "image_output": image_output,
        "has_error": bool(error_message),
    }

def cell_output_to_content_list(cell_output: dict) -> list:
    text_output = cell_output["text_output"]
    image_output = cell_output["image_output"]
    content = [
        {"type": "text", "text": "<tool_response><interpreter>" + text_output},
    ]
    for image in image_output:
        content.append(
            {"type": "image_url", "image_url": {"url": image}}  # type: ignore
        )
    content.append({"type": "text", "text": "</interpreter></tool_response>"})
    return content

def extract_code_from_response(response_text: str) -> str:
    """Extract code from the response."""
    if "<tool_call>" not in response_text or "</tool_call>" not in response_text:
        raise ValueError("Response does not contain a valid tool call.")

    code_json_str = response_text.split("<tool_call>")[1].split("</tool_call>")[0]
    try:
        code = json.loads(code_json_str)["arguments"]["code"]
    except (json.JSONDecodeError, KeyError):
        raise ValueError("Failed to extract code from the tool call.")
    return code


def call_tool(
    resposne: str,
    code_list: list,
    upload_file_dict: dict,
    sandbox_url: str | None = None,
) -> dict:
    """Call the tool with the extracted code."""

    try:
        code = extract_code_from_response(resposne)
    except Exception as e:
        return {"status": "error", "message": "Extract Code Error: " + str(e)}
    code_list.append(code)

    try:
        code_output = run_jupyter_code(code_list, sandbox_url, upload_file_dict=upload_file_dict)
    except Exception as e:
        
        raise e
        return {"status": "error", "message": "Tool Call Jupyter Runing Error: " + str(e)}

    try:
        parsed_output = parse_cell_output(code_output[-1])
        code_output_content_list = cell_output_to_content_list(parsed_output)
    except Exception as e:
        return {"status": "error", "message": "Parse Cell Output Error: " + str(e)}

    return {
        "status": "success",
        "code_output_content_list": code_output_content_list,
        "code": code,
        "code_output": code_output,
        "code_list": code_list,
    }


async def solve_one_query(
    message,
    upload_image_dict: dict,
    client: AsyncOpenAI,
    model_name: str,
    sandbox_url: str,
    max_turn=10,
    gen_kwargs=None,
) -> list[dict]:
    """Solve one query with the given message."""
    if gen_kwargs is None:
        gen_kwargs = {}

    code_list = []
    response_num = 0

    while response_num < max_turn:
        params = {
            "model": model_name,
            "messages": message,
            **gen_kwargs,
        }
        response = await client.chat.completions.create(**params)
        response_text = response.choices[0].message.content

        if "<tool_call>" in response_text and "</tool_call>" in response_text:
            tool_out = call_tool(response_text, code_list, upload_image_dict, sandbox_url)
            if tool_out["status"] == "error":
                message.extend(
                    [
                        {"role": "assistant", "content": response_text},
                        {
                            "role": "user",
                            "content": f"Error: {tool_out['message']}",
                        },
                    ]
                )
            else:
                code_list = tool_out["code_list"]
                code_output_content_list = tool_out["code_output_content_list"]
                message.extend(
                    [
                        {"role": "assistant", "content": response_text},
                        {"role": "user", "content": code_output_content_list},
                    ]
                )

            if response_num == max_turn - 1:
                if isinstance(message[-1]["content"], list):
                    message[-1]["content"].append(
                        {
                            "type": "text",
                            "text": "You have reached the maximum number of tool calls. Please provide a final response.",
                        }
                    )
                elif isinstance(message[-1]["content"], str):
                    message[-1]["content"] += (
                        "\nYou have reached the maximum number of tool calls. Please provide a final response."
                    )
                else:
                    raise ValueError(f"Unknown message content type: {type(message[-1]['content'])}")
        else:
            message.append({"role": "assistant", "content": response_text})
            break
        response_num += 1

    return message


def check_is_done(response_text: str) -> bool:
    """Helper function to check if the LLM response is a final answer."""
    return "<tool_call>" not in response_text and "</tool_call>" not in response_text


def extract_response(messages: list[dict]) -> str:
    response = ""
    
    assistant_begin = False
    for m in messages:
        # find the first assistant message
        if not assistant_begin:
            continue
        if m['role'] == "assistant":
            assistant_begin = True

        content = m.get("content", "")
        if isinstance(content, str):
            response += content
        elif isinstance(content, list):
            for c in content:
                if c.get("type") == "text":
                    response += c.get("text", "")
                elif c.get("type") == "image_url":
                    response += "<IMAGE>"
    return response

def extract_answer(response: str) -> str:
    search_result = re.search(r"<answer>(.*)</answer>", response)
    if search_result:
        response = search_result.group(1)
    else:
        response = response.strip()
    return response

def build_user_message_for_gen(
    base64_images: str | list[str], 
    question_text: str
) -> list[dict]:
    # 统一成列表并复制，避免修改调用方的原始列表
    images = [base64_images] if isinstance(base64_images, str) else list(base64_images)

    parts = question_text.split("<image>")
    assert len(images) == len(parts) - 1, (
        "The number of images and the number of <image> tags in the question text must be the same"
        f"images: {len(images)}, parts: {len(parts)}"
    )

    content: list[dict] = []
    for i, part in enumerate(parts):
        if part:  # 保留空段行为：如果连续 <image>，就只插图，不插文本
            content.append({"type": "text", "text": part})
        if i < len(images):  # 只在段与段之间插入对应图片（不会误给最后一段再塞一张）
            img = images[i]
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            })
    return content
