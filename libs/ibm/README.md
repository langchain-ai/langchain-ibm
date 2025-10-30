# langchain-ibm

This package provides the integration between LangChain and IBM watsonx.ai through the `ibm-watsonx-ai` SDK.

## Installation

To use the `langchain-ibm` package, follow these installation steps:

```bash
pip install -U langchain-ibm
```

## Setting up

To use IBM's models, you must have an IBM Cloud user API key. Here's how to obtain and set up your API key:

1. **Obtain an API Key:** For more details on how to create and manage an API key, refer to IBM's [documentation](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui).
2. **Set the API Key as an Environment Variable:** For security reasons, it's recommended to not hard-code your API key directly in your scripts. Instead, set it up as an environment variable. You can use the following code to prompt for the API key and set it as an environment variable:

```python
import os
from getpass import getpass

watsonx_api_key = getpass()
os.environ["WATSONX_API_KEY"] = watsonx_api_key
```

In alternative, you can set the environment variable in your terminal.

- **Linux/macOS:** Open your terminal and execute the following command:

  ```bash
  export WATSONX_API_KEY='your_ibm_api_key'
  ```

  To make this environment variable persistent across terminal sessions, add the above line to your `~/.bashrc`, `~/.bash_profile`, or `~/.zshrc` file.

- **Windows:** For Command Prompt, use:
  ```cmd
  set WATSONX_API_KEY=your_ibm_api_key
  ```

## Setting parameters

You might need to adjust model parameters for different models or tasks. For more details on the parameters, refer to [Parameter Scheme](https://ibm.github.io/watsonx-ai-python-sdk/fm_schema.html#) IBM's documentation.

**Note:** You must use the correct parameter schema for the class you are initializing:

- `ChatWatsonx` (for chat) uses [TextChatParameters](https://ibm.github.io/watsonx-ai-python-sdk/fm_schema.html#chat-parameters)
- `WatsonxLLM` (for text generation) uses [TextGenParameters](https://ibm.github.io/watsonx-ai-python-sdk/fm_schema.html#generate-parameters). 
- `WatsonxRerank` (for reranking) uses [RerankParameters](https://ibm.github.io/watsonx-ai-python-sdk/fm_schema.html#rerank-parameters).

This example uses `ChatWatsonx`, so we import `TextChatParameters`.

```python
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

parameters = TextChatParameters(
    temperature=0.5,
    max_completion_tokens=1024,
    top_p=1,
)
```

You can also pass it as a dictionary object.

```python
parameters = {
    "temperature": 0.5,
    "max_completion_tokens": 1024,
    "top_p": 1,
}
```

## Chat Models

`ChatWatsonx` class exposes chat models from IBM.

Initialization the `ChatWatsonx` class with the previously set parameters.

```python
from langchain_ibm import ChatWatsonx

model = ChatWatsonx(
    model_id="PASTE THE CHOSEN MODEL_ID HERE",
    url="PASTE YOUR URL HERE",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)

model.invoke("Sing a ballad of LangChain.")
```

**Note:**

- You must provide a `project_id` or `space_id`. For more information refer to IBM's [documentation](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=projects).
- Depending on the region of your provisioned service instance, use one of the urls described [here](https://ibm.github.io/watsonx-ai-python-sdk/setup_cloud.html#authentication).
- You need to specify the model you want to use for inferencing through `model_id`. You can find the list of available models [here](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes).

Alternatively for all class you can use Cloud Pak for Data credentials. For more details, refer to IBM's [documentation](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).

```python
from langchain_ibm import ChatWatsonx

model = ChatWatsonx(
    model_id="ibm/granite-3-3-8b-instruct",
    url="PASTE YOUR URL HERE",
    username="PASTE YOUR USERNAME HERE",
    password="PASTE YOUR PASSWORD HERE",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)
```

## Embedding Models

`WatsonxEmbeddings` class exposes embeddings from IBM.

```python
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

embeddings = WatsonxEmbeddings(
    model_id="ibm/granite-embedding-107m-multilingual",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=embed_params,
)

embeddings.embed_query("What is the meaning of life?")
```

## LLMs

`WatsonxLLM` class exposes LLMs from IBM.

```python
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.foundation_models.schema import TextGenParameters, TextGenDecodingMethod

parameters = TextGenParameters(
    decoding_method=TextGenDecodingMethod.SAMPLE,
    temperature=0.5,
    top_k=50,
    top_p=1
)

llm = WatsonxLLM(
    model_id="ibm/granite-3-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)

llm.invoke("The meaning of life is")
```

## Reranker

`WatsonxRerank` class exposes reranker from IBM.

```python
from langchain_ibm import WatsonxRerank

rerank = WatsonxRerank(
    model_id="cross-encoder/ms-marco-minilm-l-12-v2",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
)
```

## Toolkit

`WatsonxToolkit` class exposes Toolkit from IBM.

```python
from langchain_ibm.agent_toolkits.utility import WatsonxToolkit

watsonx_toolkit = WatsonxToolkit(
    url="https://us-south.ml.cloud.ibm.com",
)
```
