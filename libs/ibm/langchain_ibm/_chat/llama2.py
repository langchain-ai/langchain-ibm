from langchain_core.prompts import PromptTemplate

from langchain_ibm._chat.chat_schema import ChatSchema

# Prompt template shared by all llama2 models present in watsonx.ai.
_PROMPT = PromptTemplate.from_template("""{%- for message in messages -%}
{%-     if message.type == "system" -%}
[INST] <<SYS>>
{{ message.content }}
<</SYS>>

{%     elif message.type == "human" -%}
{{ message.content }}[/INST]
{%-     elif message.type == "ai" -%}
{{ message.content }}
{%-         if not loop.last -%}
[INST]
{%          endif %}
{%-     endif -%}
{%- endfor -%}""", template_format="jinja2")

LLAMA2_70B_CHAT = ChatSchema(
    model_id="meta-llama/llama-2-70b-chat",
    prompt_template=_PROMPT,
    tools=False
)

LLAMA2_13B_CHAT = ChatSchema(
    model_id="meta-llama/llama-2-13b-chat",
    prompt_template=_PROMPT,
    tools=False
)
