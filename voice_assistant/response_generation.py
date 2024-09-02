# voice_assistant/response_generation.py
import logging
from config import Config
# from IPython.display import Markdown, display

import logging
import sys
# from IPython.display import Markdown, display
from tools_and_assisstant.assisstant import assisstant_graph
import uuid
from tools_and_assisstant.utils import _print_event
from voice_assistant.text_to_speech import text_to_speech


thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "user_id": "anmolagarwal403@gmail.com",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

_printed = set()


async def generate_response(query, local_model_path=None):
    response = assisstant_graph.invoke(
        {"messages": ("user", query)}, config, stream_mode="values"
    )
    print(response['messages'][-1].content)
    text_to_speech(Config.TTS_MODEL, Config.OPENAI_API_KEY,
                               response['messages'][-1].content, "model.mp3", Config.LOCAL_MODEL_PATH)
    return response['messages'][-1].content
