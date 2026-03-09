from .base import LLMProvider, LLMResponse
from mathmotion.utils.errors import LLMError


class GeminiProvider(LLMProvider):
    def __init__(self, config):
        import google.generativeai as genai
        self._genai = genai
        self.cfg = config.llm.gemini
        genai.configure(api_key=self.cfg.api_key)
        self._client = genai.GenerativeModel(
            model_name=self.cfg.model,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
            ),
        )

    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 8192, temperature: float = 0.2,
                 response_schema: dict | None = None) -> LLMResponse:
        genai = self._genai
        full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
        try:
            response = self._client.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
        except Exception as e:
            raise LLMError(f"Gemini error: {e}") from e

        return LLMResponse(
            content=response.text,
            model=self.cfg.model,
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )

    @property
    def model_name(self) -> str:
        return self.cfg.model
