from flask import Flask, render_template, jsonify, request
import os
import json
import re
from transformers import AutoTokenizer
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain import PromptTemplate
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import threading
import time

app = Flask(__name__)

# Path to the GGUF model file
model_path = "finance-Llama3-8B.Q8_0.gguf"
local_llm = "finance-Llama3-8B.Q8_0.gguf"

# Initialize the LlamaCpp models
summarization_llm = LlamaCpp(
    model_path=model_path,
    temperature=0.3,
    max_tokens=2048,
    top_p=1,
)

chatbot_llm = LlamaCpp(
    model_path=local_llm,
    temperature=0.3,
    max_tokens=2048,
    top_p=1
)

print("Models initialized and loaded into memory.")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Metadata file
metadata_file = "delta_data.json"
summaries = []
lock = threading.Lock()

# Preprocessing function
def preprocess_text(text):
    # Remove phone numbers, emails, URLs, and other extraneous content
    text = re.sub(r'\b\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}\b', '', text)  # Phone numbers
    text = re.sub(r'\S+@\S+', '', text)  # Emails
    text = re.sub(r'http\S+|www\S+', '', text)  # URLs
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces

    # Remove repetitive words and irrelevant patterns
    text = re.sub(r'(fax|ntn|gst no|truly yours|dear sir)', '', text, flags=re.IGNORECASE)
    
    return text

# Extract relevant financial information
def extract_financial_information(text):
    # Use regex to extract sentences with financial terms
    financial_keywords = r'\b(dividend|profit|loss|eps|earnings|board meeting|share purchase|share sale)\b'
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)\s(?<=\.|\?)\s', text)  # Split by sentences
    relevant_sentences = [sentence for sentence in sentences if re.search(financial_keywords, sentence, re.IGNORECASE)]
    return ' '.join(relevant_sentences)

# Function to chunk long text into smaller parts (for multi-page documents)
def chunk_text(text, max_length=512):
    words = text.split()
    chunks = []
    
    # Split text into chunks of max_length
    while len(words) > max_length:
        chunk = ' '.join(words[:max_length])
        chunks.append(chunk)
        words = words[max_length:]
    
    # Add the remaining words as the last chunk
    if words:
        chunks.append(' '.join(words))
    
    return chunks

# Summarization logic
def process_and_summarize(ocr_text):
    try:
        # Step 1: Preprocess the OCR text
        cleaned_text = preprocess_text(ocr_text)
        
        # Step 2: Extract financial information
        financial_text = extract_financial_information(cleaned_text)
        
        # Step 3: Summarize with LLM
        summary = summarization_llm(f"Summarize the following financial information in 100-150 words: {financial_text}")
        return summary.strip()
    except Exception as e:
        print(f"Error during processing or summarization: {e}")
        return "Error generating summary"

# Summarization for long OCR text (handles chunking)
def process_and_summarize_long_text(ocr_text):
    try:
        # Step 1: Preprocess the OCR text
        cleaned_text = preprocess_text(ocr_text)
        
        # Step 2: Chunk the text if it's too long
        chunks = chunk_text(cleaned_text)
        
        # Step 3: Summarize each chunk and combine results
        full_summary = ""
        for chunk in chunks:
            financial_text = extract_financial_information(chunk)
            summary = process_and_summarize(financial_text)
            full_summary += summary + " "
        
        return full_summary.strip()
    except Exception as e:
        print(f"Error during processing or summarization: {e}")
        return "Error generating summary"

# Background thread for polling
def monitor_metadata():
    global summaries
    while True:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Get the top 2 latest files
            announcements = sorted(
                metadata.get("announcements", []),
                key=lambda x: x["date_time"],
                reverse=True
            )[:2]  # Limit to 2 summaries for backend processing

            new_summaries = []
            for ann in announcements:
                ocr_path = ann.get("ocr_file_path")
                if ocr_path and os.path.exists(ocr_path):
                    try:
                        with open(ocr_path, 'r', encoding='utf-8', errors='ignore') as f:
                            ocr_text = f.read()
                        # Generate summary
                        summary = process_and_summarize_long_text(ocr_text)
                        new_summaries.append({
                            "symbol": ann["symbol"],
                            "title": ann["title"],
                            "link": ann["file_path"],
                            "summary": summary,
                            "date_time": ann["date_time"]
                        })
                    except Exception as e:
                        print(f"Error reading OCR file {ocr_path}: {e}")
            
            # Update summaries if there are changes
            with lock:
                summaries = new_summaries

        # Poll every 1 minute to process next 2 summaries
        time.sleep(150)

# Chatbot Integration
embeddings = SentenceTransformerEmbeddings(model_name="FinLang/finance-embeddings-investopedia")

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="psx_vectors")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 documents

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        query = request.form['query']
        chain_type_kwargs = {"prompt": prompt}
        qa = RetrievalQA.from_chain_type(
            llm=chatbot_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
            verbose=True
        )
        response = qa(query)
        answer = response['result']
        # Aggregate source document contents
        source_documents = [
            {"content": doc.page_content, "source": doc.metadata.get("source", "Unknown")}
            for doc in response['source_documents']
        ]
        return jsonify({"answer": answer, "source_documents": source_documents})

    return render_template("chat.html")

@app.route('/')
def index():
    with lock:
        displayed_summaries = summaries
    return render_template("index.html", announcements=displayed_summaries)

@app.route('/api/summaries', methods=['GET'])
def get_summaries():
    with lock:
        if not summaries:
            return jsonify({"error": "No summaries available yet."}), 404
        return jsonify(summaries)

# Start background thread
thread = threading.Thread(target=monitor_metadata, daemon=True)
thread.start()

if __name__ == '__main__':
    app.run(debug=True)
