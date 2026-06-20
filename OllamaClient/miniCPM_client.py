import os
import ollama
from OllamaClient.ollama_client import BaseVisionClient


class MiniCPMClient(BaseVisionClient):
    """
    Client concreto per interagire con la famiglia MiniCPM-V (es. minicpm-v4.5).
    Eredita da BaseVisionClient e introduce il controllo del livello di pensiero.
    """

    def __init__(self, model_name: str, thinking_mode: str = "fast"):
        """
        Inizializza il client per MiniCPM.

        :param model_name: Nome del modello su Ollama (es. 'minicpm-v4.5').
        :param thinking_mode: Specifica il livello di pensiero. Supporta 'fast' o 'deep'.
        """
        super().__init__(model_name)

        if thinking_mode not in ["fast", "deep"]:
            raise ValueError("Il parametro 'thinking_mode' deve essere 'fast' o 'deep'.")

        self.thinking_mode = thinking_mode

    def _define_model_configuration(self, response_schema: any = None) -> dict:
        """Costruisce i kwargs di configurazione. Varia in base al thinking_mode."""
        config_kwargs = {}

        if response_schema:
            config_kwargs["format"] = response_schema.model_json_schema()
        else:
            config_kwargs["format"] = "json"

        # Parametri dinamici in base al livello di pensiero richiesto
        if self.thinking_mode == "deep":
            options = {
                "temperature": 0.7,
                "top_p": 0.90,
                "num_predict": 32768,  # Spazio di generazione molto ampio per i ragionamenti
                "num_ctx": 32768,  # Contesto bloccato a 32k come da te richiesto
            }
        else:  # "fast"
            options = {
                "temperature": 0.3,  # Bassa temperatura per andare dritto al punto
                "top_p": 0.80,
                "num_predict": 32768,  # Taglio rapido per efficienza
                "num_ctx": 32768,
            }

        config_kwargs["options"] = options
        return config_kwargs

    def _prepare_contents(self, image_path: str, prompt: str, system_instruction: str = None,
                          examples: list[dict] = None) -> list[dict]:
        """Costruisce l'array di messaggi con gli esempi e inietta la direttiva di pensiero."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Immagine target non trovata: {image_path}")

        messages = []

        # 1. Gestione del System Prompt e iniezione del Thinking Level
        thinking_instruction = (
            "Engage in detailed, step-by-step reasoning before providing the final structured answer.\n"
            if self.thinking_mode == "deep"
            else "Provide a fast, direct, and exact answer without over-explaining.\n"
        )

        final_system_instruction = thinking_instruction
        if system_instruction:
            final_system_instruction += system_instruction

        messages.append({"role": "system", "content": final_system_instruction})

        # 2. Gestione degli Esempi Few-Shot (Context Example)
        if examples:
            for idx, ex in enumerate(examples, 1):
                ex_img_path = ex.get("image_path")
                ex_output = ex.get("output")

                if ex_img_path and not os.path.exists(ex_img_path):
                    raise FileNotFoundError(f"Immagine dell'esempio non trovata: {ex_img_path}")

                # L'input di esempio con immagine
                user_msg = {"role": "user", "content": f"--- EXAMPLE {idx} INPUT ---"}
                if ex_img_path:
                    user_msg["images"] = [ex_img_path]

                messages.append(user_msg)

                # L'output atteso dal modello
                messages.append({
                    "role": "assistant",
                    "content": f"--- EXPECTED OUTPUT EXAMPLE {idx} ---\n{ex_output}"
                })

        # 3. Gestione del Target Input (Prompt reale + Immagine reale)
        target_msg = {
            "role": "user",
            "content": f"--- TARGET INPUT ---\n{prompt}",
            "images": [image_path]
        }
        messages.append(target_msg)

        return messages

    def _execute(self, model_kwargs: dict, messages: list[dict]) -> str | None:
        """Esegue la chiamata API locale a Ollama con log dinamici."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                **model_kwargs
            )

            # --- DEEP DEBUG Dinamico ---
            print(f"[{self.__class__.__name__} ({self.thinking_mode.upper()}) DEBUG] Tipo risposta: {type(response)}")

            if hasattr(response, "message"):
                raw_message = response.message
                content = raw_message.content if hasattr(raw_message, "content") else None
            elif isinstance(response, dict):
                content = response.get("message", {}).get("content")
            else:
                print(f"[{self.__class__.__name__} ERRORE] Formato di risposta non riconosciuto.")
                return None

            if not content or not content.strip():
                print(f"[{self.__class__.__name__} AVVISO] Il modello ha restituito una risposta vuota.")
                return None

            return content

        except Exception as e:
            print(f"[{self.__class__.__name__} ERRORE] Esecuzione fallita: {e}")
            return None