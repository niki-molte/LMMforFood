import os
from google import genai
from dotenv import load_dotenv
from google.genai import errors
load_dotenv()

from GoogleLLMClient.exceptions import InvalidApiKeyError, InvalidModelNameError, RateLimitExceededError


class GoogleClient:
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name

        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")

        if not self.api_key:
            raise InvalidApiKeyError()

        self.client = genai.Client(api_key=self.api_key)

    def ask(self, prompt: str) -> str | None:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text

        except errors.APIError as e:
            if e.code == 404:
                fetched_models = self.client.models.list()
                available_models = []

                for model in fetched_models:
                    available_models.append(model.name.replace("models/", ""))

                raise InvalidModelNameError(self.model_name, available_models)

            elif e.code == 429:
                raise RateLimitExceededError(f"Quota limit exceeded: {e.message}")

            else:
                print(f"Google API Error (Code {e.code}): {e.message}")
                return None

        except Exception as e:
            print(f"Unexpected system or network error: {e}")
            return None