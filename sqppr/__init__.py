"""SQ-PPR library modules.

Loads environment variables from /workspace/storage/GraphRAG/.env on import so
OPENAI_API_KEY (and optional ANTHROPIC_API_KEY) are available for any
downstream LLM call.
"""
import os
try:
    from dotenv import load_dotenv
    for p in ("/workspace/storage/GraphRAG/.env",
                os.path.expanduser("~/.env")):
        if os.path.isfile(p):
            load_dotenv(p)
except ImportError:
    pass
