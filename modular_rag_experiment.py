from typing import Dict, List, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings.base import Embeddings

class ModularRAGExperiment:
    def __init__(self, config: Dict[str, Any], docs_by_type: Dict[str, List[Document]], all_docs: List[Document]):
        self.config = config
        self.docs_by_type = docs_by_type  # 목적별 문서
        self.docs = all_docs              # 전체 문서 통합
        self._init_components()

    def _init_components(self):
        # 1. 청킹
        chunk_config = self.config.get("chunking", {})
        chunk_method = chunk_config.get("method", "recursive")

        if chunk_method == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_config.get("chunk_size", 1000),
                chunk_overlap=chunk_config.get("chunk_overlap", 200),
                separators=chunk_config.get("separators", ["\n\n", "\n", " ", ""]),
                length_function=len,
                is_separator_regex=False
            )
        elif chunk_method == "fixed":
            self.splitter = CharacterTextSplitter(
                chunk_size=chunk_config.get("chunk_size", 1000),
                chunk_overlap=chunk_config.get("chunk_overlap", 200),
                separator=chunk_config.get("separator", "\n\n")
            )
        else:
            raise ValueError(f"지원하지 않는 청킹 방법: {chunk_method}")

        self.chunks = self.splitter.split_documents(self.docs)

        # 2. 임베딩
        self.embedding = get_embedding_model(self.config)

        # 3. 벡터DB
        if self.config["vector_db"] == "faiss":
            self.vectorstore = FAISS.from_documents(self.chunks, self.embedding)
        elif self.config["vector_db"] == "chroma":
            self.vectorstore = Chroma.from_documents(self.chunks, self.embedding)
        else:
            raise ValueError("지원하지 않는 벡터DB")

        # 4. 검색기
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.get("initial_top_k", 20)} # rerank 사용
            # search_kwargs={"k": len(self.chunks)} # rerank 미사용
        )

        # 5. 리랭커 (옵션)
        reranker_type = self.config.get("reranker", None)
        if reranker_type == "bge":
            self.reranker = BgeReranker()
        elif reranker_type == "cohere":
            self.reranker = CohereReranker(api_key=self.config["cohere_api_key"])
        else:
            self.reranker = None  # None이면 리랭킹 미적용

        # 6. LLM
        if self.config["llm"] in llm_model_map:
            model_name = llm_model_map[self.config["llm"]]
            system_prompt = system_prompt_map.get(self.config["llm"])
            self.llm = get_hf_llm(model_name, system_prompt)
        else:
            raise ValueError("지원하지 않는 LLM")

        # 7. 프롬프트
        self.prompt_template = self.config["prompt_template"]

    def _retrieve_docs(self, question):
      docs = self.retriever.get_relevant_documents(question)

      # 리랭킹 적용 시
      if self.reranker:
          docs = self.reranker.rerank(
              query=question,
              documents=docs,
              top_n=self.config.get("rerank_top_k", 5)
          )
      else:
          docs = docs[:self.config.get("rerank_top_k", 10)]

      # 중복 제거
      seen = set()
      unique_docs = []
      for doc in docs:
          content = doc.page_content.strip()
          if content not in seen:
              seen.add(content)
              unique_docs.append(doc)

      return "\n\n".join(
          f"[출처: {doc.metadata.get('source_file', '알 수 없음')}] {doc.page_content}"
          for doc in unique_docs
    )

    def gather_and_merge_docs(self, question, top_k: int = 30):
    # 전체 문서에서 검색
        vectorstore = FAISS.from_documents(
            [doc for docs in self.docs_by_type.values() for doc in docs],
            self.embedding
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.get_relevant_documents(question)

        # 중복 제거
        seen = set()
        unique_docs = []
        for doc in docs:
            content = doc.page_content.strip()
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)

        return "\n\n".join(
            f"[출처: {doc.metadata.get('source_file', '알 수 없음')}] {doc.page_content}"
            for doc in unique_docs
    )

    def ask(self, question, use_all_sources=False):
        if use_all_sources:
            # 전체 문서 대상으로 유사도 검색
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.get("initial_top_k", 20)}
            )
            docs = retriever.get_relevant_documents(question)

            if self.reranker:
                docs = self.reranker.rerank(
                    query=question,
                    documents=docs,
                    top_n=self.config.get("rerank_top_k", 5)
                )
            else:
                docs = docs[:self.config.get("rerank_top_k", 10)]

            # 중복 제거
            seen = set()
            unique_docs = []
            for doc in docs:
                content = doc.page_content.strip()
                if content not in seen:
                    seen.add(content)
                    unique_docs.append(doc)

            context = "\n\n".join(
                f"[출처: {doc.metadata.get('source_file', '알 수 없음')}] {doc.page_content}"
                for doc in unique_docs
            )
        else:
            context = self._retrieve_docs(question)

        if not context.strip():
            return "관련 문서 없음"

        prompt = self.prompt_template.format(context=context, question=question)
        return self.llm(prompt)