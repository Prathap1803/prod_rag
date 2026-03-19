import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

def load_documents(folder_path: str):
    documents = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        try:
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_name.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file_name.endswith(".md"):
                loader = TextLoader(file_path)
            else:
                print(f"Skipping unsupported file: {file_name}")
                continue

            docs = loader.load()
            for d in docs:
                d.metadata["source"] = file_name

            documents.extend(docs)

        except Exception as e:
            print(f"Failed to load {file_name}: {e}")

    if not documents:
        raise ValueError("No valid documents found")

    return documents
