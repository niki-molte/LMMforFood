import os
import ollama
from sympy.physics.units import temperature


class OllamaClient:
    """
    Client wrapper per interagire con modelli di visione Ollama in locale.
    Supporta il Few-Shot Prompting tramite esempi multimodali.
    Progettato per esecuzione locale: nessuna logica di retry o cleanup cloud necessaria.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def _define_model_configuration(self, response_schema: any = None) -> dict:
        """Costruisce i kwargs di configurazione per il modello Ollama."""
        config_kwargs = {}

        if response_schema:
            config_kwargs["format"] = response_schema.model_json_schema()
        else:
            config_kwargs["format"] = "json"

        config_kwargs["options"] = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "num_predict": 32768,
            "num_ctx": 32768, # massimo contesto gemma4 131072
        }

        return config_kwargs

    def _prepare_contents(self, image_path: str, prompt: str, system_instruction: str = None,
                          examples: list[dict] = None) -> list[dict]:
        """
        Costruisce l'array di messaggi per Ollama.
        Verifica le immagini localmente e formatta gli esempi few-shot come turni conversazionali.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Immagine target non trovata: {image_path}")

        messages = []

        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        if examples:
            for idx, ex in enumerate(examples, 1):
                ex_img_path = ex.get("image_path")
                ex_output = ex.get("output")

                if ex_img_path and not os.path.exists(ex_img_path):
                    raise FileNotFoundError(f"Immagine dell'esempio non trovata: {ex_img_path}")

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
        """Esegue la chiamata API locale a Ollama senza logica di retry."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                **model_kwargs
            )

            # --- DEEP DEBUG: mostra il tipo e il repr grezzo della risposta ---
            print(f"[OllamaClient DEBUG] Tipo risposta: {type(response)}")
            print(f"[OllamaClient DEBUG] repr(response): {repr(response)[:500]}")

            if hasattr(response, "message"):
                raw_message = response.message
                print(f"[OllamaClient DEBUG] Tipo message: {type(raw_message)}")
                print(f"[OllamaClient DEBUG] repr(message): {repr(raw_message)[:300]}")
                content = raw_message.content if hasattr(raw_message, "content") else None
            elif isinstance(response, dict):
                # Fallback per versioni legacy della libreria che restituiscono un dict
                content = response.get("message", {}).get("content")
            else:
                print("[OllamaClient ERRORE] Formato di risposta non riconosciuto.")
                return None

            # --- DEEP DEBUG: mostra tipo e valore esatto di content ---
            print(f"[OllamaClient DEBUG] Tipo content: {type(content)}")
            print(f"[OllamaClient DEBUG] repr(content): {repr(content)[:300]}")

            if not content or not content.strip():
                print("[OllamaClient AVVISO] Il modello ha restituito una risposta vuota.")
                return None

            return content

        except Exception as e:
            print(f"[OllamaClient ERRORE] Esecuzione fallita: {e}")
            return None

    def ask(self, image_path: str, prompt: str, system_instruction: str = None,
            response_schema: any = None, examples: list[dict] = None) -> str | None:
        """
        Orchestratore pubblico: prepara la configurazione, costruisce i messaggi ed esegue la chiamata locale.
        """
        model_kwargs = self._define_model_configuration(response_schema)
        messages = self._prepare_contents(image_path, prompt, system_instruction, examples)

        return self._execute(model_kwargs, messages)