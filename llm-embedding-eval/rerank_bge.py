class BgeReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

    def _compute_score(self, query: str, document: str) -> float:
        inputs = self.tokenizer(
            query,
            document,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits.item()

    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        scores = [self._compute_score(query, doc.page_content) for doc in documents]
        sorted_indices = np.argsort(scores)[::-1]
        return [documents[i] for i in sorted_indices[:top_n]]