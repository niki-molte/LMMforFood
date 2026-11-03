import os
import ollama
from OllamaClient.ollama_client import BaseVisionClient

class QwenClient(BaseVisionClient):
    """
    Concrete client for interacting with models of the Qwen family (e.g. qwen3.5:4b, qwen3.5:9b).
    Inherits from BaseVisionClient and implements its abstract methods,
    using the optimal parameters defined in the official Modelfile.
    """

    def _define_model_configuration(self, response_schema: any = None) -> dict:
        """Builds the Qwen-specific configuration kwargs for Ollama."""
        config_kwargs = {}

        if response_schema:
            config_kwargs["format"] = response_schema.model_json_schema()
        else:
            config_kwargs["format"] = "json"

        # Parameters taken from the official Ollama documentation/Modelfile for Qwen 3.5
        config_kwargs["options"] = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "num_predict": 32768,
            "num_ctx": 65536,  # Fixed context window
        }

        return config_kwargs

    def _prepare_contents(self, image_path: str, prompt: str, system_instruction: str = None,
                          examples: list[dict] = None) -> list[dict]:
        """Builds the message array in the Ollama/Qwen chat format."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target image not found: {image_path}")

        messages = []

        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        if examples:
            for idx, ex in enumerate(examples, 1):
                ex_img_path = ex.get("image_path")
                ex_output = ex.get("output")

                if ex_img_path and not os.path.exists(ex_img_path):
                    raise FileNotFoundError(f"Example image not found: {ex_img_path}")

                user_msg = {"role": "user", "content": f"--- EXAMPLE {idx} INPUT ---"}
                if ex_img_path:
                    user_msg["images"] = [ex_img_path]

                messages.append(user_msg)

                messages.append({
                    "role": "assistant",
                    "content": f"--- EXPECTED OUTPUT EXAMPLE {idx} ---\n{ex_output}"
                })

        target_msg = {
            "role": "user",
            "content": f"--- TARGET INPUT ---\n{prompt}",
            "images": [image_path]
        }
        messages.append(target_msg)

        return messages

    def _execute(self, model_kwargs: dict, messages: list[dict]) -> str | None:
        """Executes the local Ollama API call."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                **model_kwargs
            )

            # Debug output
            print(f"[{self.__class__.__name__} DEBUG] Response type: {type(response)}")
            print(f"[{self.__class__.__name__} DEBUG] repr(response): {repr(response)[:500]}")

            if hasattr(response, "message"):
                raw_message = response.message
                content = raw_message.content if hasattr(raw_message, "content") else None
            elif isinstance(response, dict):
                content = response.get("message", {}).get("content")
            else:
                print(f"[{self.__class__.__name__} ERROR] Unrecognized response format.")
                return None

            if not content or not content.strip():
                print(f"[{self.__class__.__name__} WARNING] The model returned an empty response.")
                return None

            return content

        except Exception as e:
            print(f"[{self.__class__.__name__} ERROR] Execution failed: {e}")
            return None