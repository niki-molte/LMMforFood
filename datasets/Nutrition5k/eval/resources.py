"""
resources.py
=============
Contenitore "pigro" (lazy) per vocabolario / modello NLP / embeddings.

Il modello SentenceTransformer (e il calcolo degli embeddings del vocabolario)
viene caricato SOLO alla prima richiesta effettiva (cioè solo se serve
davvero valutare un file di predizioni non ancora in cache). Se tutte le
cache CSV/JSON sono già presenti, nessuna delle tre proprietà lazy viene
mai acceduta e il language model non viene caricato.
"""
from .data_utils import load_vocabulary


class LazyResources:
    def __init__(self, vocab_csv_path: str):
        self.vocab_csv_path = vocab_csv_path
        self._vocabulary = None
        self._nlp_model = None
        self._vocab_embeddings = None

    @property
    def vocabulary(self):
        if self._vocabulary is None:
            print("Loading vocabulary from CSV...")
            self._vocabulary = load_vocabulary(self.vocab_csv_path)
        return self._vocabulary

    @property
    def nlp_model(self):
        if self._nlp_model is None:
            print("Loading NLP model: all-MiniLM-L6-v2... (Happens only once)")
            from sentence_transformers import SentenceTransformer
            self._nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._nlp_model

    @property
    def vocab_embeddings(self):
        if self._vocab_embeddings is None:
            print("Calculating embeddings for the base vocabulary...")
            self._vocab_embeddings = self.nlp_model.encode(self.vocabulary)
        return self._vocab_embeddings
