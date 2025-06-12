from typing import Dict, List
from langchain.schema import Document
import os
import re
import glob
import json

data_path = "/content/drive/MyDrive/Textbook-Data"
purpose_docs = load_purpose_docs(data_path)
print("RAG데이터 문서:", list(purpose_docs.keys()))