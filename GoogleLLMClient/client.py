import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

from GoogleLLMClient.exceptions import InvalidApiKeyError, InvalidModelNameError, RateLimitExceededError


class GoogleClient:
    """
    A client wrapper for interacting with Google's Generative AI models.
    Supports Few-Shot Prompting through multimodal examples.
    Includes built-in retry logic (Exponential Backoff) and automatic cloud file cleanup.
    """

    def __init__(self, model_name: str, api_key: str = None, max_retries: int = 5, base_delay: float = 2.0):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.max_retries = max_retries
        self.base_delay = base_delay

        if not self.api_key:
            raise InvalidApiKeyError("GOOGLE_API_KEY not found.")

        self.client = genai.Client(api_key=self.api_key)

    def _define_model_configuration(self, system_instruction: str = None,
                                    thinking: bool = True,
                                    response_schema: any = None) -> types.GenerateContentConfig:
        """Builds the configuration object for the Google model."""
        config_kwargs = {}

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if thinking and not response_schema:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="high")

        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        return types.GenerateContentConfig(**config_kwargs)

    def _prepare_contents(self, image_path: str, prompt: str, examples: list[dict]) -> tuple[list, list]:
        """
        Uploads local images to Google's servers and constructs the request contents.
        Returns the content array and a list of uploaded file objects for later cleanup.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target image not found: {image_path}")

        uploaded_files = []
        request_contents = []

        # 1. Few-Shot Examples
        if examples:
            request_contents.append("Here are some examples of the expected task:\n")
            for idx, ex in enumerate(examples, 1):
                ex_img_path = ex.get("image_path")
                ex_output = ex.get("output")

                if ex_img_path:
                    if not os.path.exists(ex_img_path):
                        raise FileNotFoundError(f"Example image not found: {ex_img_path}")

                    ex_uploaded_img = self.client.files.upload(file=ex_img_path)
                    uploaded_files.append(ex_uploaded_img)

                    request_contents.append(f"--- EXAMPLE {idx} INPUT ---")
                    request_contents.append(ex_uploaded_img)

                request_contents.append(f"--- EXAMPLE {idx} EXPECTED OUTPUT ---")
                request_contents.append(ex_output)

            request_contents.append("\nNow, process the following target request exactly as shown in the examples:\n")

        # 2. Target Input
        target_image = self.client.files.upload(file=image_path)
        uploaded_files.append(target_image)

        request_contents.append("--- TARGET INPUT ---")
        request_contents.append(target_image)
        request_contents.append(prompt)

        return request_contents, uploaded_files

    def _execute_with_retry(self, model_config: types.GenerateContentConfig, contents: list) -> str | None:
        """Executes the API call handling network errors, rate limits, and retries using Exponential Backoff."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    config=model_config,
                    contents=contents
                )
                return response.text

            except errors.APIError as e:
                # Fatal errors that should NOT trigger a retry
                if e.code == 404:
                    print(f"[GoogleClient FATAL ERROR] Model '{self.model_name}' not found. Aborting.")
                    return None
                elif e.code == 400:
                    print(f"[GoogleClient FATAL ERROR] Bad Request (Check your schema or prompt): {e.message}")
                    return None

                # Transient errors that CAN be retried
                if attempt == self.max_retries - 1:
                    print(
                        f"[GoogleClient ERROR] API call failed completely after {self.max_retries} attempts: {e.message}")
                    return None

                sleep_time = self.base_delay * (2 ** attempt)

                if e.code == 429:
                    print(
                        f"[Google Rate Limit] Quota exceeded. Retrying in {sleep_time}s (Attempt {attempt + 1}/{self.max_retries})")
                elif e.code in [500, 503]:
                    print(
                        f"[Google Server Busy] Internal Error. Retrying in {sleep_time}s (Attempt {attempt + 1}/{self.max_retries})")
                else:
                    print(
                        f"[Google Generic Error] {e.message}. Retrying in {sleep_time}s (Attempt {attempt + 1}/{self.max_retries})")

                time.sleep(sleep_time)

            except Exception as e:
                # Catch-all for non-API errors (e.g., connection lost)
                if attempt == self.max_retries - 1:
                    print(f"[GoogleClient UNEXPECTED ERROR] {e}")
                    return None

                sleep_time = self.base_delay * (2 ** attempt)
                print(
                    f"[GoogleClient Unexpected] {e}. Retrying in {sleep_time}s (Attempt {attempt + 1}/{self.max_retries})")
                time.sleep(sleep_time)

    def ask(self, image_path: str, prompt: str, system_instruction: str = None,
            thinking: bool = True, response_schema: any = None,
            examples: list[dict] = None) -> str | None:
        """
        Public orchestrator: prepares configuration, uploads files, executes API, and cleans up cloud files.
        """
        uploaded_files = []
        try:
            # 1. Configure parameters
            model_config = self._define_model_configuration(system_instruction, thinking, response_schema)

            # 2. Upload files and build messages
            contents, uploaded_files = self._prepare_contents(image_path, prompt, examples)

            # 3. Execute with robust retry logic
            return self._execute_with_retry(model_config, contents)

        finally:
            # 4. Mandatory Cleanup: Delete files from Google's servers to avoid hitting storage quotas
            for file_obj in uploaded_files:
                try:
                    self.client.files.delete(name=file_obj.name)
                except Exception as cleanup_error:
                    print(f"[Warning] Error during cloud file cleanup for {file_obj.name}: {cleanup_error}")