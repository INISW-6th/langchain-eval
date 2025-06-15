import time

# 질문
question = "조선의 건국과정"
ask_start_time = time.time()
response = rag_experiment.ask(question)
ask_end_time = time.time()

print("💭💭💭 답변:", response)
print(f"\n답변 소요 시간: {ask_end_time - ask_start_time:.2f}초")