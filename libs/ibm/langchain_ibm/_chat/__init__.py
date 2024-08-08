from langchain_ibm._chat.llama3 import LLAMA31_405B, LLAMA31_70B, LLAMA31_8B, LLAMA3_70B, LLAMA3_8B
from langchain_ibm._chat.llama2 import LLAMA2_70B_CHAT, LLAMA2_13B_CHAT
from langchain_ibm._chat.granite import GRANITE_13B_CHAT_V2
from langchain_ibm._chat.mistral import MISTRAL_LARGE, MIXTRAL_8X7B_V01


CHAT_SCHEMAS = {
    schema.model_id: schema for schema in (
        MISTRAL_LARGE,
        MIXTRAL_8X7B_V01,
        LLAMA31_405B,
        LLAMA31_70B,
        LLAMA31_8B,
        LLAMA3_70B,
        LLAMA3_8B,
        LLAMA2_70B_CHAT,
        LLAMA2_13B_CHAT,
        GRANITE_13B_CHAT_V2
    )
}

__all__ = ["CHAT_SCHEMAS"]
