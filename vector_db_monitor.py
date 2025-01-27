import os
import json
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="FinLang/finance-embeddings-investopedia")

# Directory containing downloaded announcements
data_dir = 'psx_announcements'
metadata_file = "delta_data.json"
collection_name = "psx_vectors"

# Process all announcements in the JSON file
def process_all_announcements():
    if not os.path.exists(metadata_file):
        print(f"Metadata file '{metadata_file}' not found.")
        return

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Process all announcements in the JSON file
    documents = []
    metadatas = []  # List to hold metadata for each document chunk
    for ann in metadata.get("announcements", []):
        # Extract metadata fields
        symbol = ann.get("symbol")
        title = ann.get("title")
        file_path = ann.get("file_path")
        ocr_file_path = ann.get('ocr_file_path')
        date_time = ann.get('date_time')  # Extract the date_time field

        if ocr_file_path and ocr_file_path.endswith('.txt') and os.path.exists(ocr_file_path):
            # Load text from the OCR file with encoding handling
            try:
                with open(ocr_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
            except UnicodeDecodeError:
                print(f"Failed to decode {ocr_file_path} with UTF-8. Retrying with 'latin-1'.")
                with open(ocr_file_path, 'r', encoding='latin-1') as f:
                    text_content = f.read()

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            chunks = text_splitter.split_text(text_content)

            # Append each chunk with its corresponding metadata
            for chunk in chunks:
                documents.append(chunk)
                metadatas.append({
                    "symbol": symbol,
                    "title": title,
                    "file_path": file_path,
                    "date_time": date_time,  # Add date_time to metadata
                })

    if not documents:
        print("No documents found in the JSON file.")
        return

    # Vectorize and store in Qdrant
    qdrant = Qdrant.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=metadatas,  # Include metadata with each vector
        url="http://localhost:6333",
        prefer_grpc=False,
        collection_name=collection_name,
    )
    print("Vector DB initialized with all announcements!")

# Process updates from the metadata file
def process_latest_announcements():
    print("Processing updates to the metadata file...")
    process_all_announcements()

# Watchdog Event Handler
class MetadataFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(metadata_file):
            print(f"Detected update to {metadata_file}. Processing updates...")
            process_latest_announcements()

# Monitor the metadata file for changes
def start_monitoring():
    print(f"Initializing vector database with all existing announcements...")
    process_all_announcements()
    print(f"Monitoring changes to '{metadata_file}'...")
    event_handler = MetadataFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(metadata_file) or ".", recursive=False)
    observer.start()
    try:
        while True:
            pass  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_monitoring()
