# voice_assistant/response_generation.py

from openai import OpenAI
from groq import Groq
import ollama
import logging
from voice_assistant.config import Config
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from IPython.display import Markdown, display
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_parse import LlamaParse  # pip install llama-parse

import logging
import sys
from IPython.display import Markdown, display

import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
def generate_response(model, api_key, chat_history, local_model_path=None):
    
    # Generate a response using the specified model.

    # Args:
    # model (str): The model to use for response generation ('openai', 'groq', 'local').
    # api_key (str): The API key for the response generation service.
    # chat_history (list): The chat history as a list of messages.
    # local_model_path (str): The path to the local model (if applicable).

    # Returns:
    # str: The generated response text.
    
    try:
        if model == 'openai':
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=Config.OPENAI_LLM,
                messages=chat_history
            )
            return response.choices[0].message.content
        elif model == 'groq':
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=Config.GROQ_LLM,  # "llama3-8b-8192",
                messages=chat_history
            )
            return response.choices[0].message.content
        elif model == 'ollama':
            response = ollama.chat(
                model=Config.OLLAMA_LLM,
                messages=chat_history,
                # stream=True,
            )
            return response['message']['content']
        elif model == 'local':
            # Placeholder for local LLM response generation
            return "Generated response from local model"
        else:
            raise ValueError("Unsupported response generation model")
    except Exception as e:
        logging.error(f"Failed to generate response: {e}")
        return "Error in generating response"
"""


def rag_setup():

    chat_store = SimpleChatStore()

    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key="user1",
    )

    llm = Gemini(api_key=Config.GEMINI_KEY)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    parser = LlamaParse(
        # can also be set in your env as LLAMA_CLOUD_API_KEY
        api_key=Config.LLAMA_CLOUD_KEY,
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=4,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="en",  # Optionally you can define a language, default=en
    )

# sync
    documents = parser.load_data("data/sam.txt")
    # documents = pd.read_csv("data/flipkart_com-ecommerce_sample.csv")
    # documents = SimpleDirectoryReader("data").load_data()
    global chat_engine
    index = VectorStoreIndex.from_documents(documents)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=chat_memory,
    )
    # chat_engine = PandasQueryEngine(df=documents, verbose=True)


def generate_response(model, api_key, chat_history, local_model_path=None):
    response = chat_engine.chat(chat_history)
    return response.response
