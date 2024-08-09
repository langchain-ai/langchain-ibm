import json
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_ibm._chat.llama3 import LLAMA31_405B
from langchain_ibm._chat.llama2 import LLAMA2_70B_CHAT
from langchain_ibm._chat.mistral import MISTRAL_LARGE
from langchain_ibm._chat.granite import GRANITE_13B_CHAT_V2
from langchain_ibm.chat_models import _merge_chunk_message_runs


CONVERSATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a respectful AI assistant."),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
])

HISTORY = [
    ("user", "Hello, how are you?"),
    ("assistant", "Hello! I'm doing well, thank you for asking. I'm a large language model assistant."),
    ("user", "What's the capital of Italy?"),
    ("assistant", "Rome"),
]

INSTRUCT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting entities from text. Always extract entities using the following json output format:
{{
    "type": "entity_type",
    "value": "entity_value"
}}"""),
    ("user",
     "Here is the input:\n\"{input}\". Remember to always respond in the json format specified above."),
    ("assistant", "Here is the json output:\n")
])

TOOL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an helpful assistant."),
    ("user", "What's 3 + 2?")
])


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_llama3_conversation_no_history() -> None:
    expected = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a respectful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello, how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    assert LLAMA31_405B.template.render(messages=CONVERSATION_PROMPT.format_messages(
        chat_history=[], input="Hello, how are you?")) == expected


def test_llama3_conversation_history() -> None:
    expected = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a respectful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello, how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hello! I'm doing well, thank you for asking. I'm a large language model assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What's the capital of Italy?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Rome<|eot_id|><|start_header_id|>user<|end_header_id|>

Thanks!<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    assert LLAMA31_405B.template.render(messages=CONVERSATION_PROMPT.format_messages(
        chat_history=HISTORY, input="Thanks!")) == expected


def test_llama3_instruct_prompt() -> None:
    expected = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert at extracting entities from text. Always extract entities using the following json output format:
{
    "type": "entity_type",
    "value": "entity_value"
}<|eot_id|><|start_header_id|>user<|end_header_id|>

Here is the input:
"The capital of Italy is Rome.". Remember to always respond in the json format specified above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Here is the json output:
"""

    assert LLAMA31_405B.template.render(messages=INSTRUCT_PROMPT.format_messages(
        input="The capital of Italy is Rome.")) == expected


def test_llama3_tool_no_system_prompt():
    expected = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

tools

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    messages = ChatPromptTemplate.from_messages([
        ("human", "Hello")
    ])

    assert LLAMA31_405B.template.render(
        messages=messages.format_messages(), tools="tools") == expected


def test_llama3_tool_prompt():
    tools = [convert_to_openai_tool(add)]

    expected = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython

You are an helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}. Do not use variables.

{json.dumps(tools)}

What's 3 + 2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    assert LLAMA31_405B.template.render(
        messages=TOOL_PROMPT.format_messages(),
        input="What's 3 + 2?",
        tools=json.dumps(tools)
    ) == expected

    tool_call_message = AIMessage(
        content='<|python_tag|>[{"name": "add", "parameters": {"a": 3, "b": 2}}]',
    )

    assert LLAMA31_405B.tools_parser

    tool_calls = LLAMA31_405B.tools_parser(str(tool_call_message.content))
    assert isinstance(tool_calls, list) and tool_calls[0]["name"] == "add" and tool_calls[0]["args"] == {
        "a": 3, "b": 2}

    parsed_tool_call_message = AIMessage(
        content='',
        tool_calls=tool_calls,
        additional_kwargs={"tool_calls": json.dumps([
            {"name": "add", "parameters": {"a": 3, "b": 2}, "id": "123"}
        ])}
    )
    
    expected += f"""

<|python_tag|>[{{"name": "add", "parameters": {{"a": 3, "b": 2}}, "id": "{tool_calls[0]['id']}"}}]<|eom_id|><|start_header_id|>ipython<|end_header_id|>

[{{"content": "5", "id": "{tool_calls[0]['id']}"}}]<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    assert LLAMA31_405B.template.render(
        messages=(
            TOOL_PROMPT +
            parsed_tool_call_message +
            ToolMessage(content="5", tool_call_id=tool_calls[0]["id"])).format_messages(),
        input="What's 3 + 2?",
        tools=json.dumps(tools)
    ) == expected


def test_mistral_conversation_no_history() -> None:
    expected = """<s>[INST] You are a respectful AI assistant. [/INST]</s>\
[INST] Hello, how are you? [/INST]"""

    assert MISTRAL_LARGE.template.render(messages=CONVERSATION_PROMPT.format_messages(
        chat_history=[], input="Hello, how are you?")) == expected


def test_mistral_conversation_history() -> None:
    expected = """<s>[INST] You are a respectful AI assistant. [/INST]</s>\
[INST] Hello, how are you? [/INST]\
Hello! I'm doing well, thank you for asking. I'm a large language model assistant.</s>\
[INST] What's the capital of Italy? [/INST]\
Rome</s>\
[INST] Thanks! [/INST]"""

    assert MISTRAL_LARGE.template.render(messages=CONVERSATION_PROMPT.format_messages(
        chat_history=HISTORY, input="Thanks!")) == expected


def test_mistral_instruct_prompt() -> None:
    expected = """<s>[INST] You are an expert at extracting entities from text. Always extract entities using the following json output format:
{
    "type": "entity_type",
    "value": "entity_value"
} [/INST]</s>\
[INST] Here is the input:
"The capital of Italy is Rome.". Remember to always respond in the json format specified above. [/INST]\
Here is the json output:
"""

    assert MISTRAL_LARGE.template.render(messages=INSTRUCT_PROMPT.format_messages(
        input="The capital of Italy is Rome.")) == expected


def test_mistral_tool_prompt():
    tools = [convert_to_openai_tool(add)]

    expected = f"""<s>[INST] You are an helpful assistant. [/INST]</s>\
[AVAILABLE_TOOLS] {json.dumps(tools)} [/AVAILABLE_TOOLS]\
[INST] What's 3 + 2? [/INST]"""

    assert MISTRAL_LARGE.template.render(
        messages=TOOL_PROMPT.format_messages(),
        input="What's 3 + 2?",
        tools=json.dumps(tools)
    ) == expected

    tool_call_message = AIMessage(
        content='[TOOL_CALLS][{"name": "add", "arguments": {"a": 3, "b": 2}}]',
    )

    assert MISTRAL_LARGE.tools_parser

    tool_calls = MISTRAL_LARGE.tools_parser(str(tool_call_message.content))
    assert isinstance(tool_calls, list) and tool_calls[0]["name"] == "add" and tool_calls[0]["args"] == {
        "a": 3, "b": 2}

    parsed_tool_call_message = AIMessage(
        content='',
        id="123",
        tool_calls=tool_calls,
        additional_kwargs={"tool_calls": json.dumps([
            {"name": "add", "arguments": {"a": 3, "b": 2}, "id": "123"}
        ])}
    )

    expected += f"""[TOOL_CALLS] [{{"name": "add", "arguments": {{"a": 3, "b": 2}}, "id": "{tool_calls[0]['id']}"}}]</s>\
[TOOL_RESULTS] {{"content": "5", "id": "{tool_calls[0]['id']}"}} [/TOOL_RESULTS]"""

    assert MISTRAL_LARGE.template.render(
        messages=(
            TOOL_PROMPT +
            parsed_tool_call_message +
            ToolMessage(content="5", tool_call_id=tool_calls[0]["id"])).format_messages(),
        input="What's 3 + 2?",
        tools=json.dumps(tools)
    ) == expected


def test_granite_conversation_no_history() -> None:
    expected = """<|system|>
You are a respectful AI assistant.
<|user|>
Hello, how are you?
<|assistant|>"""

    assert GRANITE_13B_CHAT_V2.template.render(messages=CONVERSATION_PROMPT.format_messages(
        chat_history=[], input="Hello, how are you?")) == expected


def test_granite_conversation_history() -> None:
    expected = """<|system|>
You are a respectful AI assistant.
<|user|>
Hello, how are you?
<|assistant|>
Hello! I'm doing well, thank you for asking. I'm a large language model assistant.
<|user|>
What's the capital of Italy?
<|assistant|>
Rome
<|user|>
Thanks!
<|assistant|>"""

    assert GRANITE_13B_CHAT_V2.template.render(messages=CONVERSATION_PROMPT.format_messages(
        chat_history=HISTORY, input="Thanks!")) == expected


def test_granite_instruct_prompt() -> None:
    expected = """<|system|>
You are an expert at extracting entities from text. Always extract entities using the following json output format:
{
    "type": "entity_type",
    "value": "entity_value"
}
<|user|>
Here is the input:
"The capital of Italy is Rome.". Remember to always respond in the json format specified above.
<|assistant|>
Here is the json output:
"""

    assert GRANITE_13B_CHAT_V2.template.render(messages=INSTRUCT_PROMPT.format_messages(
        input="The capital of Italy is Rome.")) == expected


def test_llama2_conversation_no_history() -> None:
    expected = """[INST] <<SYS>>
You are a respectful AI assistant.
<</SYS>>

Hello, how are you?[/INST]"""

    assert LLAMA2_70B_CHAT.template.render(messages=CONVERSATION_PROMPT.format_messages(
        chat_history=[], input="Hello, how are you?")) == expected


def test_llama2_conversation_history() -> None:
    expected = """[INST] <<SYS>>
You are a respectful AI assistant.
<</SYS>>

Hello, how are you?[/INST]Hello! I'm doing well, thank you for asking. I'm a large language model assistant.[INST]
What's the capital of Italy?[/INST]Rome[INST]
Thanks![/INST]"""

    assert LLAMA2_70B_CHAT.template.render(messages=CONVERSATION_PROMPT.format_messages(
        chat_history=HISTORY, input="Thanks!")) == expected


def test_llama2_instruct_prompt() -> None:
    expected = """[INST] <<SYS>>
You are an expert at extracting entities from text. Always extract entities using the following json output format:
{
    "type": "entity_type",
    "value": "entity_value"
}
<</SYS>>

Here is the input:
"The capital of Italy is Rome.". Remember to always respond in the json format specified above.[/INST]Here is the json output:
"""

    assert LLAMA2_70B_CHAT.template.render(messages=INSTRUCT_PROMPT.format_messages(
        input="The capital of Italy is Rome.")) == expected


def test_merge_chunk_message_runs():
    UNMERGED_MESSAGES = [
        ("system", "You are a respectful AI assistant."),
        ("user", "What is the capital of Italy?"),
        AIMessage(content="The"),
        AIMessage(content="Capital"),
        AIMessageChunk(content=" of Italy"),
        AIMessageChunk(content=" is"),
        AIMessageChunk(content=" Rome"),
    ]

    merged = _merge_chunk_message_runs(UNMERGED_MESSAGES)

    assert merged == [
        SystemMessage(content="You are a respectful AI assistant."),
        HumanMessage(content="What is the capital of Italy?"),
        AIMessage(content="The"),
        AIMessage(content="Capital"),
        AIMessage(content=" of Italy is Rome"),

    ]
