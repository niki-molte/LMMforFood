import os
import ollama
from OllamaClient.ollama_client import BaseVisionClient


class MiniCPMClient(BaseVisionClient):
    """
    Concrete client for interacting with the MiniCPM-V family (e.g. minicpm-v4.5).
    Inherits from BaseVisionClient and adds control over the thinking level.
    """

    def __init__(self, model_name: str, thinking_mode: str = "fast"):
        """
        Initializes the MiniCPM client.

        :param model_name: Model name on Ollama (e.g. 'minicpm-v4.5').
        :param thinking_mode: Thinking level. Supports 'fast' or 'deep'.
        """
        super().__init__(model_name)

        if thinking_mode not in ["fast", "deep"]:
            raise ValueError("The 'thinking_mode' parameter must be 'fast' or 'deep'.")

        self.thinking_mode = thinking_mode

    def _define_model_configuration(self, response_schema: any = None) -> dict:
        """Builds the configuration kwargs. Varies with thinking_mode."""
        config_kwargs = {}

        if response_schema:
            config_kwargs["format"] = response_schema.model_json_schema()
        else:
            config_kwargs["format"] = "json"

        # Parameters depend on the requested thinking level
        if self.thinking_mode == "deep":
            options = {
                "temperature": 0.7,
                "top_p": 0.90,
                "num_predict": 32768,  # Very large generation budget for reasoning
                "num_ctx": 32768,  # Context fixed at 32k
            }
        else:  # "fast"
            options = {
                "temperature": 0.3,  # Low temperature to get straight to the point
                "top_p": 0.80,
                "num_predict": 32768,  # Quick cutoff for efficiency
                "num_ctx": 32768,
            }

        config_kwargs["options"] = options
        return config_kwargs

    def _prepare_contents(self, image_path: str, prompt: str, system_instruction: str = None,
                          examples: list[dict] = None) -> list[dict]:
        """Builds the message array with the examples and injects the thinking directive."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target image not found: {image_path}")

        messages = []

        # 1. System prompt handling and thinking level injection
        thinking_instruction = (
            "Engage in detailed, step-by-step reasoning before providing the final structured answer.\n"
            if self.thinking_mode == "deep"
            else "Provide a fast, direct, and exact answer without over-explaining.\n"
        )

        final_system_instruction = thinking_instruction
        if system_instruction:
            final_system_instruction += system_instruction

        messages.append({"role": "system", "content": final_system_instruction})

        # 2. Few-shot examples
        if examples:
            for idx, ex in enumerate(examples, 1):
                ex_img_path = ex.get("image_path")
                ex_output = ex.get("output")

                if ex_img_path and not os.path.exists(ex_img_path):
                    raise FileNotFoundError(f"Example image not found: {ex_img_path}")

                # Example input with image
                user_msg = {"role": "user", "content": f"--- EXAMPLE {idx} INPUT ---"}
                if ex_img_path:
                    user_msg["images"] = [ex_img_path]

                messages.append(user_msg)

                # Expected model output
                messages.append({
                    "role": "assistant",
                    "content": f"--- EXPECTED OUTPUT EXAMPLE {idx} ---\n{ex_output}"
                })

        # 3. Target input (actual prompt + actual image)
        target_msg = {
            "role": "user",
            "content": f"--- TARGET INPUT ---\n{prompt}",
            "images": [image_path]
        }
        messages.append(target_msg)

        return messages

    def _execute(self, model_kwargs: dict, messages: list[dict]) -> str | None:
        """Executes the local Ollama API call with dynamic logging."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                **model_kwargs
            )

            # Debug output
            print(f"[{self.__class__.__name__} ({self.thinking_mode.upper()}) DEBUG] Response type: {type(response)}")

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