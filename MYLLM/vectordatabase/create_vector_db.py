import argparse
import json
import logging
import os
import pathlib
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import wandb

# Configure logging
logger = logging.getLogger(__name__)

# Configure OpenAI API Key
def configure_openai_api_key():
    if os.getenv("OPENAI_API_KEY") is None:
        with open("C:/dev/EE569/Assignment2-LLM/LLM/key.txt", "r") as f:
            os.environ["OPENAI_API_KEY"] = f.readline().strip()
    assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "Invalid OpenAI API key"
    print("OpenAI API key configured")

MODEL_NAME = "gpt-4o-mini"
configure_openai_api_key()

def load_json_documents(json_directory: str):
    """Load all JSON files from a directory and process each كتاب as a batch."""
    all_batches = []

    def extract_books(data):
        """Extract all 'كتاب' (level=1) entries and their sub-documents."""
        for item in data:
            if item.get("level") == 1 and "كتاب" in item.get("text", ""):
                book_documents = []
                process_item(item, book_documents)
                all_batches.append(book_documents)
            elif "children" in item:
                extract_books(item["children"])

    def process_item(item, documents):
        """Recursively process each item and add it to the list."""
        doc_metadata = {
            "title": item.get("text", ""),
            "level": item.get("level", ""),
            "href": item.get("href", ""),
        }
        content = item.get("content", "")
        documents.append(Document(page_content=content, metadata=doc_metadata))

        if "children" in item:
            for child in item["children"]:
                process_item(child, documents)

    for filename in os.listdir(json_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(json_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                extract_books(data)

    print(f"Processed {len(all_batches)} 'كتاب' batches.")
    return all_batches

def chunk_documents(documents: List[Document], chunk_size: int = 10000, chunk_overlap: int = 1000) -> List[Document]:
    """Split documents into smaller chunks to avoid exceeding token limits."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document_sections = text_splitter.split_documents(documents)

    chunked_documents = []
    for doc in document_sections:
        doc_metadata = doc.metadata
        if doc.page_content.strip():
            chunked_documents.append(Document(page_content=doc.page_content, metadata=doc_metadata))

    print(f"Total document sections: {len(chunked_documents)}")
    return chunked_documents

def create_vector_store(documents: List[Document], vector_store_path: str) -> Chroma:
    """Create a ChromaDB vector store from a list of documents."""
    api_key = os.environ.get("OPENAI_API_KEY", None)
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)

    # Create and persist the vector store

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=vector_store_path
    )
    vector_store.persist()
    return vector_store

def ingest_data_in_batches(docs_dir: str, chunk_size: int, chunk_overlap: int, vector_store_path: str):
    """Ingest each 'كتاب' batch into the vector store separately."""
    book_batches = load_json_documents(docs_dir)

    for idx, batch in enumerate(book_batches):
        print(f"Ingesting batch {idx+1}/{len(book_batches)}...")
        split_documents = chunk_documents(batch, chunk_size, chunk_overlap)
        if(len(split_documents)>400):
           vector_store = create_vector_store(split_documents[0:400], vector_store_path)
           vector_store = create_vector_store(split_documents[400:], vector_store_path)
        else:
          vector_store = create_vector_store(split_documents, vector_store_path)
        print(f"Batch {idx+1} stored successfully.")

    print("All batches ingested into the vector store.")

def search_vector_store(query: str, vector_store: Chroma):
    """Search the vector store and return results including metadata."""
    results = vector_store.similarity_search(query, k=5)  # Adjust k as needed
    for result in results:
        print(f"Content: {result.page_content}\nMetadata: {result.metadata}\n")

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

def load_vector_store(vector_store_path: str) -> Chroma:
    """Load a Chroma vector store from the specified directory."""
    return Chroma(persist_directory=vector_store_path, embedding_function=OpenAIEmbeddings())

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
        default=10000,
        help="The number of tokens to include in each document chunk",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=1000,
        help="The number of tokens to overlap between document chunks",
    )
    parser.add_argument(
        "--vector_store",
        type=str,
        default="./vector_storev2",
        help="The directory to save or load the Chroma db to/from",
    )
    parser.add_argument(
        "--wandb_project",
        default="llmapps",
        type=str,
        help="The wandb project to use for storing artifacts",
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    run = wandb.init(project=args.wandb_project, config=args)

    ingest_data_in_batches(
        docs_dir=args.docs_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_store_path=args.vector_store,
    )

    # Load vector store and perform a sample search
    vector_store = load_vector_store(args.vector_store)
    query = "التيمم عند المرض"
    search_vector_store(query, vector_store)

    # Log the prompt
    with open("C:/dev/EE569/Assignment2-LLM/prompts.json", "r") as f:
        prompt = json.load(f)
    log_prompt(prompt, run)

    run.finish()

if __name__ == "__main__":
    main()
