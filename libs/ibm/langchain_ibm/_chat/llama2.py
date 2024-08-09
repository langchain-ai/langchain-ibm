from langchain_ibm._chat.chat_schema import ChatSchema, template_env

# Prompt template shared by all llama2 models present in watsonx.ai.
_TEMPLATE = template_env.from_string("""{%- for message in messages -%}
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
{%- endfor -%}""")

LLAMA2_70B_CHAT = ChatSchema(
    model_id="meta-llama/llama-2-70b-chat",
    template=_TEMPLATE,
    tools=False
)

LLAMA2_13B_CHAT = ChatSchema(
    model_id="meta-llama/llama-2-13b-chat",
    template=_TEMPLATE,
    tools=False
)
