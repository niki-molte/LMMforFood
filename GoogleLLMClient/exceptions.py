class InvalidApiKeyError(Exception):
    def __init__(self,
                 message="Invalid API Key. Provide an API Key in the constructor or in the .env file (GOOGLE_API_KEY)."):
        super().__init__(message)


class InvalidModelNameError(Exception):
    def __init__(self, invalid_name: str, available_models: list[str]):
        models_list_str = "\n- ".join(available_models)
        message = f"The model '{invalid_name}' is invalid or does not exist.\n\nAvailable models:\n- {models_list_str}"
        super().__init__(message)


class RateLimitExceededError(Exception):
    def __init__(self, message="API quota limits exceeded (RPM, TPM, or RPD). Please wait and try again."):
        super().__init__(message)
