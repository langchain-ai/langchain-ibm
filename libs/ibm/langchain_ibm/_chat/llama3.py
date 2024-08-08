import json
import random
import string
from typing import List, Union
from langchain_core.messages import ToolCall
from langchain_core.prompts import PromptTemplate

from langchain_ibm._chat.chat_schema import ChatSchema

_alphanum = string.ascii_letters + string.digits

# Prompt template shared by all llama3 and llama3.1 models present in watsonx.ai.
#
# Supports tool calls for llama3.1 models.
# See: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling
_PROMPT = PromptTemplate.from_template("""<|begin_of_text|>
{%- set system_message = messages | selectattr('type', 'equalto', 'system') | list -%}
{%- if system_message|length == 0 and tools -%}
<|start_header_id|>system<|end_header_id|>

Environment: ipython<|eot_id|>
{%- endif -%}
{%- for message in messages -%}
{%-     if message.type == "system" -%}
<|start_header_id|>system<|end_header_id|>

{%          if tools -%}
Environment: ipython

{%         endif -%}
{{ message.content }}<|eot_id|>
{%-     elif message.type == "human" -%}
<|start_header_id|>user<|end_header_id|>

{%          if tools and loop.last -%}
Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{{ tools }}

{%          endif -%}
{{ message.content }}<|eot_id|>
{%-     elif message.type == "ai" -%}
<|start_header_id|>assistant<|end_header_id|>

{%          if message.tool_calls -%}
<|python_tag|>{{ message.additional_kwargs["tool_calls"] }}{% if not loop.last %}<|eom_id|>{% endif %}
{%-         else -%}
{{ message.content }}{% if not loop.last %}<|eot_id|>{% endif %}
{%-         endif -%}
{%-     elif message.type == "tool" -%}
<|start_header_id|>ipython<|end_header_id|>

[{"content": "{{ message.content }}", "id": "{{ message.tool_call_id }}"}]<|eot_id|>
{%-     endif -%}
{%-     if loop.last and message.type != "ai" -%}
<|start_header_id|>assistant<|end_header_id|>
{%-     endif -%}
{%- endfor -%}""", template_format="jinja2")


def parse_llama31_tool_call(text: str) -> Union[str, List[ToolCall]]:
    tool_calls = []
    if text.strip().startswith("<|python_tag|>"):
        text = text.strip()[len("<|python_tag|>"):]

        try:
            json_calls = json.loads(text)
            if not isinstance(json_calls, list):
                json_calls = [json_calls]

            for call in json_calls:
                # Follow mistral's tool call id generation. See `mistral.py`
                id = "".join(random.choice(_alphanum) for _ in range(9))
                tool_call = ToolCall(
                    name=call["name"],
                    args=call["parameters"],
                    id=id,
                )
                tool_calls.append(tool_call)

            return tool_calls
        except:
            return text
    else:
        return text


# ==== Llama 3.1 Models  with tool call support ====

LLAMA31_405B = ChatSchema(
    model_id="meta-llama/llama-3-405b-instruct",
    prompt_template=_PROMPT,
    tools=True,
    tools_parser=parse_llama31_tool_call,
    tools_stop_sequences=["<|eom_id|>"]
)

LLAMA31_70B = ChatSchema(
    model_id="meta-llama/llama-3-1-70b-instruct",
    prompt_template=_PROMPT,
    tools=True,
    tools_parser=parse_llama31_tool_call,
    tools_stop_sequences=["<|eom_id|>"]
)

LLAMA31_8B = ChatSchema(
    model_id="meta-llama/llama-3-1-8b-instruct",
    prompt_template=_PROMPT,
    tools=True,
    tools_parser=parse_llama31_tool_call,
    tools_stop_sequences=["<|eom_id|>"]
)

# ==== Llama 3 models ====

LLAMA3_70B = ChatSchema(
    model_id="meta-llama/llama-3-70b-instruct",
    prompt_template=_PROMPT,
    tools=False,
)

LLAMA3_8B = ChatSchema(
    model_id="meta-llama/llama-3-8b-instruct",
    prompt_template=_PROMPT,
    tools=False,
)
