import argparse
import json
import tiktoken
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pathlib
from typing import List, Tuple
from langchain_community.document_loaders import JSONLoader
import langchain
import wandb
from langchain_community.cache import SQLiteCache
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Configure OpenAI API Key
def configure_openai_api_key():
    if os.getenv("OPENAI_API_KEY") is None:
        with open("C:/dev/EE569/Assignment2-LLM/LLM/key.txt", "r") as f:
            os.environ["OPENAI_API_KEY"] = f.readline().strip()
    assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "Invalid OpenAI API key"
    print("OpenAI API key configured")

MODEL_NAME = "gpt-4o-mini"
configure_openai_api_key()

langchain.llm_cache = SQLiteCache(database_path="langchain.db")

logger = logging.getLogger(__name__)

def load_json_documents(json_directory: str):
    """Load all JSON files from a directory and add metadata."""
    documents = []

    def extract_documents(data, level=1):
        """Recursively extract documents from nested JSON structure."""
        for item in data:
            doc_metadata = {
                "title": item.get("text", ""),
                "level": level,
                "href": item.get("href", ""),
            }

            # Extract the content if available (e.g., a description or details for the document)
            content = item.get("content", "")  # Using 'text' as the content here for simplicity

            # Create a document from the content and metadata
            documents.append(Document(page_content=content, metadata=doc_metadata))

            # If there are children, recursively process them with an increased level
            if "children" in item:
                extract_documents(item["children"], level + 1)

    for filename in os.listdir(json_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(json_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)  # Load the entire JSON file

            # Extract documents starting from the top level
            extract_documents(data)

    print(f"Loaded {len(documents)} JSON documents.")
    return documents



def load_documents(data_dir: str) -> List[Document]:
    """Load documents and include metadata from a directory of JSON files."""
    json_files = list(map(str, pathlib.Path(data_dir).glob("*.json")))
    documents = []

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                doc_metadata = {
                    "title": item.get("text", ""),
                    "level": item.get("level", ""),
                    "href": item.get("href", ""),
                }
                documents.append(Document(page_content=item.get('content', ''), metadata=doc_metadata))
    return documents

def chunk_documents(
    documents: List[Document], chunk_size: int = 5000, chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into smaller chunks to avoid exceeding token limits."""
    tokenizer = tiktoken.encoding_for_model(MODEL_NAME)

    def count_tokens(documents):
        return [len(tokenizer.encode(doc.page_content)) for doc in documents]

    token_counts = count_tokens(documents) 

    print(f"Token counts per document: {token_counts}")
    print(f"Token counts sum: {sum(token_counts)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    document_sections = text_splitter.split_documents(documents)

    chunked_documents = []
    for doc in document_sections:
        doc_metadata = doc.metadata
        if(doc.page_content == ""):
            continue # Copy metadata from the original document
        chunked_documents.append(Document(page_content=doc.page_content, metadata=doc_metadata))

    print(f"Total document sections: {len(chunked_documents)}")
    print(f"Example document section content:\n{chunked_documents[0].page_content}")

    return chunked_documents

def create_vector_store(
    documents: List[Document],
    vector_store_path: str = "./vector_store",
) -> Chroma:
    """Create a ChromaDB vector store from a list of documents."""
    api_key = os.environ.get("OPENAI_API_KEY", None)
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)

    # Chroma expects documents, where each document contains page_content and metadata
    vector_store = Chroma.from_documents(
        documents=documents,  # Each document should already have its metadata
        embedding=embedding_function,
        persist_directory=vector_store_path
    )
    vector_store.persist()
    return vector_store


def log_dataset(documents: List[Document], run: "wandb.run"):
    """Log a dataset to wandb."""
    document_artifact = wandb.Artifact(name="documentation_dataset", type="dataset")
    with document_artifact.new_file("documents.json", mode="w", encoding="utf-8") as f:
        for document in documents:
            f.write(document.json() + "\n")
    run.log_artifact(document_artifact)

def log_index(vector_store_dir: str, run: "wandb.run"):
    """Log a vector store to wandb."""
    index_artifact = wandb.Artifact(name="vector_store", type="search_index")
    index_artifact.add_dir(vector_store_dir)
    run.log_artifact(index_artifact)

def log_prompt(prompt: dict, run: "wandb.run"):
    """Log a prompt to wandb."""
    prompt_artifact = wandb.Artifact(name="chat_prompt", type="prompt")
    with prompt_artifact.new_file("prompt.json") as f:
        f.write(json.dumps(prompt))
    run.log_artifact(prompt_artifact)

def ingest_data(
    docs_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    vector_store_path: str,
) -> Tuple[List[Document], Chroma]:
    """Ingest a directory of JSON files into a vector store."""
    documents = load_json_documents(docs_dir)

    split_documents = chunk_documents(documents, chunk_size, chunk_overlap)
    if(len(split_documents)>400):
      vector_store = create_vector_store(split_documents[0:400], vector_store_path)
      vector_store = create_vector_store(split_documents[400:], vector_store_path)
    else:
      vector_store = create_vector_store(split_documents, vector_store_path)
    return split_documents, vector_store

import html

def search_vector_store(query: str, vector_store: Chroma):
    """Search the vector store and return results including metadata."""
    results = vector_store.similarity_search(query, k=5)  # Adjust k as needed
    decoded_results = []
    
    for result in results:
        decoded_content = html.unescape(result.page_content)  # Decode HTML entities
        decoded_results.append((decoded_content, result.metadata))
    
    return decoded_results


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs_dir",
        type=str,
        required=False,
        default=r"C:\dev\EE569\Assignment2-LLM\MYLLM\docs_sample",
        help="The directory containing the JSON files",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="The number of tokens to include in each document chunk",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="The number of tokens to overlap between document chunks",
    )
    parser.add_argument(
        "--vector_store",
        type=str,
        default="./vector_store",
        help="The directory to save or load the Chroma db to/from",
    )
    parser.add_argument(
        "--prompt_file",
        type=pathlib.Path,
        default=r"C:\dev\EE569\Assignment2-LLM\prompts.json",
        help="The path to the chat prompt to use",
    )
    parser.add_argument(
        "--wandb_project",
        default="llmapps",
        type=str,
        help="The wandb project to use for storing artifacts",
    )
    return parser
def load_vector_store(vector_store_path: str) -> Chroma:
    """Load a Chroma vector store from the specified directory."""
    return Chroma(persist_directory=vector_store_path, embedding_function=OpenAIEmbeddings())

def main():
    parser = get_parser()
    args = parser.parse_args()
    run = wandb.init(project=args.wandb_project, config=args)
    # documents, vector_store = ingest_data(
    #     docs_dir=args.docs_dir,
    #     chunk_size=args.chunk_size,
    #     chunk_overlap=args.chunk_overlap,
    #     vector_store_path=args.vector_store,
    # )
    # log_dataset(documents, run)
    log_index(args.vector_store, run)
    query = "صلاة المريض"
    vector_store = load_vector_store(args.vector_store)
    results = search_vector_store(query, vector_store)
    print("Query Results:")
    for content, metadata in results:
        print(f"Content: {content}\nMetadata: {metadata}\n")
    log_prompt(json.load(args.prompt_file.open("r")), run)
    
    run.finish()

if __name__ == "__main__":
    main()
