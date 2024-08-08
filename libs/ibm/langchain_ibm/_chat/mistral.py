import json
import random
import string
from typing import List, Union
from langchain_core.messages import ToolCall
from langchain_core.prompts import PromptTemplate

from langchain_ibm._chat.chat_schema import ChatSchema

_alphanum = string.ascii_letters + string.digits

# Prompt template shared by all Mistral models present in watsonx.ai.
#
# Supports tool calls for tool-enabled Mistral models.
# See: https://github.com/mistralai/mistral-common/blob/main/examples/tokenizer.ipynb
# See: https://ollama.com/library/mistral-large/blobs/cd887e2923a9
_PROMPT = PromptTemplate.from_template("""<s>
{%- for message in messages -%}
    {%- if message.type == "system" -%}
        [INST] {{ message.content }} [/INST]</s>
    {%- elif message.type == "human" -%}
        {%- if (messages|length - loop.index == 0 or (messages|length - loop.index == 1 and messages[messages|length - 1].type != "tool")) and tools -%}
            [AVAILABLE_TOOLS] {{ tools }} [/AVAILABLE_TOOLS]
        {%- endif -%}
        [INST] {{ message.content }} [/INST]
    {%- elif message.type == "ai" -%}
        {%- if message.content -%}
            {{ message.content }}
            {%- if messages|length - loop.index != 0 -%}
                </s>
            {%- endif -%}
        {%- elif message.tool_calls -%}
            [TOOL_CALLS]{{ message.additional_kwargs["tool_calls"] }}</s>
        {%- endif -%}
    {%- elif message.type == "tool" -%}
        [TOOL_RESULTS] {"content": "{{ message.content }}", "id": "{{ message.tool_call_id }}"} [/TOOL_RESULTS]
    {%- endif -%}
{%- endfor -%}""", template_format="jinja2")


def parse_mistral_tool_call(text: str) -> Union[str, List[ToolCall]]:
    tool_calls = []
    if text.strip().startswith("[TOOL_CALLS]"):
        text = text.strip()[len("[TOOL_CALLS]"):]

        try:
            json_calls = json.loads(text)
            if not isinstance(json_calls, list):
                json_calls = [json_calls]

            for call in json_calls:
                # A Mistral tool call id is a random string of 9 characters in [a-zA-Z0-9].
                # https://github.com/mistralai/mistral-common/blob/00780973136e3dac4d541e0712174eb34016debf/src/mistral_common/protocol/instruct/validator.py#L307
                id = "".join(random.choice(_alphanum) for _ in range(9))
                tool_call = ToolCall(
                    name=call["name"],
                    args=call["arguments"],
                    id=id,
                )
                tool_calls.append(tool_call)

            return tool_calls
        except:
            return text
    else:
        return text


MISTRAL_LARGE = ChatSchema(
    model_id="mistralai/mistral-large",
    prompt_template=_PROMPT,
    tools=True,
    tools_parser=parse_mistral_tool_call,
)

MIXTRAL_8X7B_V01 = ChatSchema(
    model_id="mistralai/mixtral-8x7b-instruct-v01",
    prompt_template=_PROMPT,
    tools=False,
)
