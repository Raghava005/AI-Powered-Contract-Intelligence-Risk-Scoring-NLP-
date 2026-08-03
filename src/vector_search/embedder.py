from sentence_transformers import SentenceTransformer


class ContractEmbedder:

    def __init__(self):

        print("Loading Sentence Transformer...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Model Loaded Successfully!")

    def generate_embedding(self, text):

        return self.model.encode(
            text,
            convert_to_numpy=True
        )

    def generate_embeddings(self, texts):

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )