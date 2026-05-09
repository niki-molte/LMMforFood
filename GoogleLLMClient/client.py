import os
from google import genai
from google.genai import errors, types
from dotenv import load_dotenv
load_dotenv()

from GoogleLLMClient.exceptions import InvalidApiKeyError, InvalidModelNameError, RateLimitExceededError


class GoogleClient:
    """
    A client wrapper for interacting with Google's Generative AI models.
    Supports Few-Shot Prompting through multimodal examples.
    """

    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")

        if not self.api_key:
            raise InvalidApiKeyError()

        self.client = genai.Client(api_key=self.api_key)

    def _define_model_configuration(self, system_instruction: str = None,
                                    thinking: bool = True,
                                    response_schema: any = None) -> types.GenerateContentConfig:
        config_kwargs = {}

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if thinking and not response_schema:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="high")

        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"  # ← restore this
            config_kwargs["response_schema"] = response_schema

        return types.GenerateContentConfig(**config_kwargs)

    def ask(self, image_path: str, prompt: str, system_instruction: str = None,
            thinking: bool = True, response_schema: any = None,
            examples: list[dict] = None) -> str | None:
        """
        Sends a multimodal request to the model, optionally including examples (few-shot prompting).

        Args:
            image_path (str): The local file path pointing to the target image.
            prompt (str): The main instruction for the target image.
            system_instruction (str, optional): High-level instructions for the model's persona.
            thinking (bool, optional): Whether to use advanced reasoning. Defaults to True.
            response_schema (Type[BaseModel] | dict, optional): Enforces a JSON output format.
            examples (list[dict], optional): A list of dictionaries representing examples.
                                             Format: [{"image_path": "...", "output": "..."}]

        Returns:
            str | None: The generated response, or None if an error occurs.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        model_config = self._define_model_configuration(system_instruction, thinking, response_schema)

        uploaded_files = []
        request_contents = []

        try:
            if examples:
                request_contents.append("Here are some examples of the expected task:\n")
                for idx, ex in enumerate(examples, 1):
                    ex_img_path = ex.get("image_path")
                    ex_output = ex.get("output")

                    if not ex_img_path or not os.path.exists(ex_img_path):
                        raise FileNotFoundError(f"Example image not found: {ex_img_path}")

                    ex_uploaded_img = self.client.files.upload(file=ex_img_path)
                    uploaded_files.append(ex_uploaded_img)

                    request_contents.append(f"--- EXAMPLE {idx} INPUT ---")
                    request_contents.append(ex_uploaded_img)
                    request_contents.append(f"--- EXAMPLE {idx} EXPECTED OUTPUT ---")
                    request_contents.append(ex_output)

                request_contents.append(
                    "\nNow, process the following target request exactly as shown in the examples:\n")

            target_image = self.client.files.upload(file=image_path)
            uploaded_files.append(target_image)

            request_contents.append("--- TARGET INPUT ---")
            request_contents.append(target_image)
            request_contents.append(prompt)

            response = self.client.models.generate_content(
                model=self.model_name,
                config=model_config,
                contents=request_contents
            )
            return response.text

        except errors.APIError as e:
            if e.code == 404:
                fetched_models = self.client.models.list()
                available_models = [model.name.replace("models/", "") for model in fetched_models]
                raise InvalidModelNameError(self.model_name, available_models)
            elif e.code == 429:
                raise RateLimitExceededError(f"Quota limit exceeded: {e.message}")
            else:
                print(f"Google API Error (Code {e.code}): {e.message}")
                return None
        except Exception as e:
            print(f"Unexpected system or network error: {e}")
            return None

        finally:
            for file_obj in uploaded_files:
                try:
                    self.client.files.delete(name=file_obj.name)
                except Exception as cleanup_error:
                    print(f"Error during cloud file cleanup for {file_obj.name}: {cleanup_error}")