import os
import time
import base64
import mimetypes
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

from GoogleLLMClient.exceptions import InvalidApiKeyError


class HuggingFaceClient:
    """
    A client wrapper for interacting with Hugging Face Inference API via the native huggingface_hub library.
    Supports Few-Shot Prompting through multimodal examples using raw Base64 image encoding.
    Includes built-in retry logic (Exponential Backoff).
    """

    def __init__(self, model_name: str, api_key: str = None, max_retries: int = 5, base_delay: float = 2.0):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY")
        self.max_retries = max_retries
        self.base_delay = base_delay

        if not self.api_key:
            raise InvalidApiKeyError("HUGGINGFACE_API_KEY not found.")

        self.client = InferenceClient(model=self.model_name, token=self.api_key)

    def _encode_image(self, image_path: str) -> str:
        """Reads the raw image file and encodes it to base64 without any resizing."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{encoded_string}"

    def _build_messages(self, image_path: str, prompt: str, system_instruction: str, examples: list[dict]) -> list[
        dict]:
        """Constructs the chat messages list, including system prompts, few-shot examples, and the target input."""
        messages = []

        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        if examples:
            for idx, ex in enumerate(examples, 1):
                ex_img_path = ex.get("image_path")
                ex_output = ex.get("output")

                user_content = [{"type": "text", "text": f"--- EXAMPLE {idx} INPUT ---"}]
                if ex_img_path:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": self._encode_image(ex_img_path)}
                    })

                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": ex_output})

        target_content = [{"type": "text", "text": f"--- TARGET INPUT ---\n{prompt}"}]
        if image_path:
            target_content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(image_path)}
            })

        messages.append({"role": "user", "content": target_content})

        return messages

    def _execute_with_retry(self, kwargs: dict) -> str | None:
        """Executes the API call handling network errors, rate limits, and retries using Exponential Backoff."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat_completion(**kwargs)
                return response.choices[0].message.content

            except Exception as e:
                error_msg = str(e).lower()

                if attempt == self.max_retries - 1:
                    print(
                        f"[HuggingFaceClient ERROR] API call failed completely after {self.max_retries} attempts: {e}")
                    return None

                sleep_time = self.base_delay * (2 ** attempt)

                if "429" in error_msg or "too many requests" in error_msg:
                    print(
                        f"[HF Rate Limit] Quota exceeded. Retrying in {sleep_time}s "
                        f"(Attempt {attempt + 1}/{self.max_retries})")
                elif "503" in error_msg or "unavailable" in error_msg:
                    print(
                        f"[HF Server Busy] Server loading model. Retrying in {sleep_time}s "
                        f"(Attempt {attempt + 1}/{self.max_retries})")
                elif "413" in error_msg or "payload too large" in error_msg:
                    print(
                        f"[HF Size Error] Raw image too large. Retrying in {sleep_time}s... "
                        f"(Attempt {attempt + 1}/{self.max_retries})")
                else:
                    print(
                        f"[HF Generic Error] {e}. Retrying in {sleep_time}s "
                        f"(Attempt {attempt + 1}/{self.max_retries})")

                time.sleep(sleep_time)

    def ask(self, image_path: str, prompt: str, system_instruction: str = None,
            thinking: bool = True, response_schema: any = None,
            examples: list[dict] = None) -> str | None:
        """
        Public orchestrator: builds messages, configures parameters, and calls the API.
        """
        messages = self._build_messages(image_path, prompt, system_instruction, examples)

        kwargs = {
            "messages": messages,
            "max_tokens": 3500
        }

        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}

        return self._execute_with_retry(kwargs)