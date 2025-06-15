experiment_config = {
    "chunking": {
        "method": "recursive",  # "recursive" 또는 "fixed"
        "chunk_size": 1000,
        "chunk_overlap": 300,
        "separators": ["\n\n", "\n", "。", " ", ""]  # 한국어 문장 분리자 추가
    },
    "embedding": {
    "model_type": "huggingface",
    "model_name": "jhgan/ko-sroberta-multitask"  # 기본 사용 모델
    # 임베딩 모델 옵션 :
    # - "FronyAI/frony-embed-large-ko-v1"
    # - "BM-K/KoSimCSE-roberta"
    # - "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    },
    "vector_db": "chroma", 
    # vector_store 옵션
    # - 'faiss' 
    # - 'chroma'
    
    "initial_top_k": 20,
    "rerank_top_k": 5,
    "reranker": None,  # "bge", "cohere", 또는 None(리랭킹 미사용)
    "cohere_api_key": "your-cohere-api-key",  # cohere 사용 시
    "llm": "yanolja",   
    # LLM모델 옵션
    # - "exaone"
    # - "qwen"
    # - "hyperclova"
    # - "kanana"

    "prompt_template": """
    당신은 모든 문서에서만 정보를 추출하여 답변해야 합니다.
    당신은 모든 문서를 기반으로 역사적 사실만 전달해야하는 역사선생님 입니다.
    검색된 모든 문서 내용에 없는 정보는 절대 추측하지 말고 "해당 정보는 제공되지 않았습니다"라고 답하십시오.
    주어진 문서에 없는 추측은 **절대 금지**입니다.
    답변은 한번만 출력하도록 하십시오.
    답변은 완전한 문장으로 답하시오.
    반복되는 문장없이 출력하시오.
    한 줄에 한 문장이상 넘지않도록 출력하고 이어서 아랫줄로 넘어가세요.
    문서에 복수 해석이 가능한 경우에는 모두 서술하시오.
    예: "아마 ~일 것이다" 형태의 표현은 금지.
    최대 10문장이내로 답변하세요.
    답변의 각 문장은 어떤 문서의 어느 부분에서 왔는지 정확히 출처를 함께 달아줘. 만약 출처가 없으면 그 내용은 빼고, 추론 없이 대답해줘.
    질문에 대해서 모든 문서를 검색해서 정확히 알려줘. 한 문장마다 출처가 있어야 하고, 없는 문장은 쓰지 마.

    모든 문서를 참고하여 '기승전결' 구조로 수업자료를 작성하시오.

    조건:
    1. "기(起) - 발단", "승(承) - 전개", "전(轉) - 전환", "결(結) - 마무리" 형식으로 구분하여 작성
    2. 각 항목은 간결하게 2~3문장
    3. 마지막에 '전체 맥락 요약' 항목을 추가
    4. 교과서 스타일의 설명체로 작성

    문서:
    {context}

    질문: {question}
"""
}