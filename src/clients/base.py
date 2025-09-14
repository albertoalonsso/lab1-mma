from typing import Dict, Any, Protocol

class LLMClient(Protocol):
    async def analyze(self, prompt: str) -> Dict[str, Any]:
        ...

def env_keys_status() -> Dict[str, str]:
    import os
    keys = ["OPENAI_API_KEY","ANTHROPIC_API_KEY","DEEPSEEK_API_KEY"]
    return {k: ("set" if os.getenv(k) else "missing") for k in keys}
