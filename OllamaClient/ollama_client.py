from abc import ABC, abstractmethod
import os
import ollama


class BaseVisionClient(ABC):
    """
    Classe astratta di base (Interfaccia) per tutti i client di visione locale o cloud.
    Garantisce che ogni modello implementi la propria logica di configurazione,
    preparazione dei messaggi ed esecuzione, mantenendo un'interfaccia pubblica uniforme.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def _define_model_configuration(self, response_schema: any = None) -> dict:
        """Costruisce i parametri/kwargs di configurazione specifici per il provider."""
        pass

    @abstractmethod
    def _prepare_contents(self, image_path: str, prompt: str, system_instruction: str = None,
                         examples: list[dict] = None) -> any:
        """Formatta i messaggi, le immagini e gli esempi according al formato richiesto dal provider."""
        pass

    @abstractmethod
    def _execute(self, model_kwargs: dict, messages: any) -> str | None:
        """Esegue l'effettiva chiamata di inferenza all'API del modello."""
        pass

    def ask(self, image_path: str, prompt: str, system_instruction: str = None,
            response_schema: any = None, examples: list[dict] = None) -> str | None:
        """
        Orchestratore pubblico (Template Method).
        Questo flusso rimane identico per qualsiasi modello di visione presente e futuro.
        """
        model_kwargs = self._define_model_configuration(response_schema)
        messages = self._prepare_contents(image_path, prompt, system_instruction, examples)
        return self._execute(model_kwargs, messages)