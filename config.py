# voice_assistant/config.py

import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


class Config:
    """
    Configuration class to hold the model selection and API keys.

    Attributes:
    TRANSCRIPTION_MODEL (str): The model to use for transcription ('openai', 'groq', 'deepgram', 'fastwhisperapi', 'local').
    RESPONSE_MODEL (str): The model to use for response generation ('openai', 'groq', 'local').
    TTS_MODEL (str): The model to use for text-to-speech ('openai', 'deepgram', 'elevenlabs', 'local').
    OPENAI_API_KEY (str): API key for OpenAI services.
    GROQ_API_KEY (str): API key for Groq services.
    DEEPGRAM_API_KEY (str): API key for Deepgram services.
    ELEVENLABS_API_KEY (str): API key for ElevenLabs services.
    LOCAL_MODEL_PATH (str): Path to the local model.
    """
    # Model selection
    # possible values: openai, groq, deepgram, fastwhisperapi
    TRANSCRIPTION_MODEL = 'openai'
    RESPONSE_MODEL = 'openai'  # possible values: openai, groq, ollama
    TTS_MODEL = 'openai'  # possible values: openai, deepgram, elevenlabs, melotts, cartesia

    # currently using the MeloTTS for local models. here is how to get started:
    # https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#linux-and-macos-install

    # LLM Selection
    OLLAMA_LLM = "llama3:8b"
    GROQ_LLM = "llama3-8b-8192"
    OPENAI_LLM = "gpt-4o"

    # API keys and paths
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")
    CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
    GEMINI_KEY = os.getenv("GEMINI_KEY")
    LLAMA_CLOUD_KEY = os.getenv("LLAMA_CLOUD_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2 = "true"
    LANGCHAIN_PROJECT = "Flipkart Support Bot Tutorial"
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = "https://grid-api.openai.azure.com/"

    # for serving the MeloTTS model
    TTS_PORT_LOCAL = 5150

    # temp file generated by the initial STT model
    INPUT_AUDIO = "test.mp3"

    @staticmethod
    def validate_config():
        """
        Validate the configuration to ensure all necessary environment variables are set.

        Raises:
        ValueError: If a required environment variable is not set.
        """
        if Config.TRANSCRIPTION_MODEL not in ['openai', 'groq', 'deepgram', 'fastwhisperapi', 'local']:
            raise ValueError(
                "Invalid TRANSCRIPTION_MODEL. Must be one of ['openai', 'groq', 'deepgram', 'fastwhisperapi', 'local']")
        if Config.RESPONSE_MODEL not in ['openai', 'groq', 'ollama', 'local']:
            raise ValueError(
                "Invalid RESPONSE_MODEL. Must be one of ['openai', 'groq', 'local']")
        if Config.TTS_MODEL not in ['openai', 'deepgram', 'elevenlabs', 'melotts', 'cartesia', 'local']:
            raise ValueError(
                "Invalid TTS_MODEL. Must be one of ['openai', 'deepgram', 'elevenlabs', 'melotts', 'cartesia', 'local']")

        if Config.TRANSCRIPTION_MODEL == 'openai' and not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI models")
        if Config.TRANSCRIPTION_MODEL == 'groq' and not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for Groq models")
        if Config.TRANSCRIPTION_MODEL == 'deepgram' and not Config.DEEPGRAM_API_KEY:
            raise ValueError(
                "DEEPGRAM_API_KEY is required for Deepgram models")

        if Config.RESPONSE_MODEL == 'openai' and not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI models")
        if Config.RESPONSE_MODEL == 'groq' and not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for Groq models")

        if Config.TTS_MODEL == 'openai' and not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI models")
        if Config.TTS_MODEL == 'deepgram' and not Config.DEEPGRAM_API_KEY:
            raise ValueError(
                "DEEPGRAM_API_KEY is required for Deepgram models")
        if Config.TTS_MODEL == 'elevenlabs' and not Config.ELEVENLABS_API_KEY:
            raise ValueError(
                "ELEVENLABS_API_KEY is required for ElevenLabs models")
        if Config.TTS_MODEL == 'cartesia' and not Config.CARTESIA_API_KEY:
            raise ValueError(
                "CARTESIA_API_KEY is required for Cartesia models")
