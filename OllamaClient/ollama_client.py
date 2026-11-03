from abc import ABC, abstractmethod
import os
import ollama


class BaseVisionClient(ABC):
    """
    Abstract base class (interface) for all local or cloud vision clients.
    Ensures that each model implements its own configuration logic,
    message preparation and execution, while keeping a uniform public interface.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def _define_model_configuration(self, response_schema: any = None) -> dict:
        """Builds the provider-specific configuration parameters/kwargs."""
        pass

    @abstractmethod
    def _prepare_contents(self, image_path: str, prompt: str, system_instruction: str = None,
                         examples: list[dict] = None) -> any:
        """Formats the messages, images and examples according to the format required by the provider."""
        pass

    @abstractmethod
    def _execute(self, model_kwargs: dict, messages: any) -> str | None:
        """Performs the actual inference call to the model API."""
        pass

    def ask(self, image_path: str, prompt: str, system_instruction: str = None,
            response_schema: any = None, examples: list[dict] = None) -> str | None:
        """
        Public orchestrator (Template Method).
        This flow stays the same for any present or future vision model.
        """
        model_kwargs = self._define_model_configuration(response_schema)
        messages = self._prepare_contents(image_path, prompt, system_instruction, examples)
        return self._execute(model_kwargs, messages)