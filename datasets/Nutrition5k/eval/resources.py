"""
resources.py
=============
Lazy container for the vocabulary / NLP model / embeddings.

The SentenceTransformer model (and the computation of the vocabulary embeddings)
is loaded ONLY on the first actual request (i.e. only if a predictions file
that is not yet cached really needs to be evaluated). If all the
CSV/JSON caches are already present, none of the three lazy properties is
ever accessed and the language model is not loaded.
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
