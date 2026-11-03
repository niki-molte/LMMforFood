import os
import ollama
from OllamaClient.ollama_client import BaseVisionClient


class GemmaClient(BaseVisionClient):
    """
    Concrete client for interacting with models of the Ollama/Gemma4 family.
    Inherits from BaseVisionClient and implements its abstract methods.
    """

    def _define_model_configuration(self, response_schema: any = None) -> dict:
        """Builds the Ollama-specific configuration kwargs."""
        config_kwargs = {}

        if response_schema:
            config_kwargs["format"] = response_schema.model_json_schema()
        else:
            config_kwargs["format"] = "json"

        config_kwargs["options"] = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 64,
            "num_predict": 32768,
            "num_ctx": 32768,  # Locally allocated context window
        }

        return config_kwargs

    def _prepare_contents(self, image_path: str, prompt: str, system_instruction: str = None,
                          examples: list[dict] = None) -> list[dict]:
        """Builds the message array in the Ollama chat format."""
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

                user_content = f"--- EXAMPLE {idx} INPUT ---"

                if "prompt" in ex:
                    user_content += f"\n{ex['prompt']}"

                user_msg = {"role": "user", "content": user_content}

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
            print(f"[OllamaClient DEBUG] Response type: {type(response)}")
            print(f"[OllamaClient DEBUG] repr(response): {repr(response)[:500]}")

            if hasattr(response, "message"):
                raw_message = response.message
                content = raw_message.content if hasattr(raw_message, "content") else None
            elif isinstance(response, dict):
                content = response.get("message", {}).get("content")
            else:
                print("[OllamaClient ERROR] Unrecognized response format.")
                return None

            if not content or not content.strip():
                print("[OllamaClient WARNING] The model returned an empty response.")
                return None

            return content

        except Exception as e:
            print(f"[OllamaClient ERROR] Execution failed: {e}")
            return None