import json
from typing import Union


def get_tool_description() -> list:
    tool_description = [
        {
            "type": "function",
            "function": {
                "name": "execute_python_code_in_jupyter",
                "description": "Execute Python code in a persistent Jupyter environment to solve a wide variety of problems. This powerful tool runs code and returns results and error information.\n\n**Persistent Environment**: This is a stateful Jupyter notebook environment where:\n- Variables and data structures persist between code executions\n- Previously imported libraries remain available for reuse\n- Functions and classes you define are remembered\n- You can build upon previous computations step by step\n- Commonly used packages such as matplotlib, scipy, pandas, and seaborn are already installed\n- You can get all output (including the image output) of jupyter cell. \n\nPython code is incredibly versatile and can help you solve numerous types of problems:\n1. **Mathematical & Scientific Computing**: Perform complex calculations, solve equations, statistical analysis, linear algebra operations using libraries like NumPy, SciPy, SymPy. If you are doing math question;\n2. **Data Analysis & Visualization**: Process datasets, create charts and graphs, analyze trends using Pandas, Matplotlib, Plotly, Seaborn.\n3. **Image Processing**: Load, manipulate, crop, rotate, enhance contrast, adjust brightness, apply filters, detect features using PIL. You can use img.show() to display results.",  # noqa: E501
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code for a single Jupyter cell that you need to run",
                        }
                    },
                    "required": ["code"],
                },
            },
        }
    ]
    return tool_description


def get_system_prompt() -> str:
    tool_description = get_tool_description()
    system_message = f"""You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{json.dumps(tool_description, ensure_ascii=False)}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""

    return system_message


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

