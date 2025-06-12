from typing import List
from langchain.schema import Document
import cohere

class CohereReranker:
    def __init__(self, api_key: str, model_name: str = "rerank-multilingual-v2.0"):
        import cohere
        self.client = cohere.Client(api_key)
        self.model_name = model_name

    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        doc_texts = [doc.page_content for doc in documents]
        response = self.client.rerank(
            query=query,
            documents=doc_texts,
            top_n=top_n,
            model=self.model_name
        )
        return [documents[result.index] for result in response.results]