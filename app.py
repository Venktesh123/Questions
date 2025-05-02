import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Load environment variables from .env
load_dotenv()

# Load Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Google API Key not found. Please check your .env file.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# Load sentence transformer model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables for vector data
chunks = None
embeddings = None

# Load Files
def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Chunking the transcript
def chunk_text(text, chunk_size=500):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Save and Load vector data
def save_vector_data(chunks, embeddings):
    np.save("chunks.npy", np.array(chunks))
    np.save("embeddings.npy", embeddings)

def load_vector_data():
    if os.path.exists("chunks.npy") and os.path.exists("embeddings.npy"):
        chunks = np.load("chunks.npy", allow_pickle=True).tolist()
        embeddings = np.load("embeddings.npy")
        return chunks, embeddings
    return None, None

# Build Vector Embeddings
def build_vector_embeddings(chunks):
    embeddings = embed_model.encode(chunks)
    return chunks, embeddings

# Semantic search using NumPy (replacing FAISS)
def semantic_search(query_text, chunks, embeddings, top_k=1):
    # Encode the query
    query_embedding = embed_model.encode([query_text])[0]
    
    # Calculate L2 distances
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    
    # Get indices of top_k smallest distances
    top_indices = np.argsort(distances)[:top_k]
    
    # Get corresponding chunks
    retrieved_chunks = [chunks[i] for i in top_indices]
    
    return retrieved_chunks

# Question Generation
def generate_questions(retrieved_content, co_text, bloom_level):
    prompt_parts = [
        "You are a Question Generator Agent.",
        f"Course Outcome (CO): {co_text}",
        f"Bloom's Taxonomy Level: {bloom_level}",
        "Based on the content below, generate multiple questions:",
        "- Two Objective Type Questions",
        "- Two Short Answer Type Questions",
        "Content:\n" + retrieved_content,
        "\nOnly output the questions in the following format:",
        "Objective Questions:",
        "1. <question 1>",
        "2. <question 2>",
        "Short Answer Questions:",
        "1. <question 1>",
        "2. <question 2>"
    ]

    full_prompt = "\n".join(prompt_parts)

    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(full_prompt)

    output = response.text.strip()
    return output

# Parse generated questions into structured format
def parse_questions(questions_text):
    objective_questions = []
    subjective_questions = []
    
    if "Objective Questions:" in questions_text and "Short Answer Questions:" in questions_text:
        parts = questions_text.split("Short Answer Questions:")
        obj_part = parts[0].replace("Objective Questions:", "").strip()
        subj_part = parts[1].strip()
        
        # Extract objective questions
        for line in obj_part.split("\n"):
            if line.strip() and any(c.isdigit() for c in line[:2]):
                question = line.strip()
                # Remove the number prefix (e.g., "1. ", "2. ")
                if ". " in question[:3]:
                    question = question[question.find(". ")+2:]
                objective_questions.append(question)
        
        # Extract subjective questions
        for line in subj_part.split("\n"):
            if line.strip() and any(c.isdigit() for c in line[:2]):
                question = line.strip()
                # Remove the number prefix
                if ". " in question[:3]:
                    question = question[question.find(". ")+2:]
                subjective_questions.append(question)
    
    return {"objective": objective_questions, "subjective": subjective_questions}

# Save generated questions to JSON
def save_to_json(selected_co, selected_bloom, questions_dict, json_file="generated_questions.json"):
    new_entry = {
        "course_outcome": selected_co,
        "bloom_level": selected_bloom,
        "questions": questions_dict
    }

    # Load existing data if exists
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(new_entry)

    # Save back to JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    return new_entry

# Initialize vector database
def initialize_vector_db():
    global chunks, embeddings
    # Load or build vector DB
    chunks, embeddings = load_vector_data()
    if chunks is None or embeddings is None:
        print("Building vector database... please wait")
        transcript = load_file("cleaned_transcript.txt")
        chunks = chunk_text(transcript)
        chunks, embeddings = build_vector_embeddings(chunks)
        save_vector_data(chunks, embeddings)
        print("Vector database built and cached")
    else:
        print("Loaded cached vector database")

# Create Flask app for API
app = Flask(__name__)

@app.route('/', methods=['GET'])
def api_status():
    return jsonify({
        "status": "online",
        "message": "Course Outcome & Bloom's Level Question Generator API is running",
        "usage": {
            "endpoint": "/generate-questions",
            "method": "POST",
            "body": {
                "course_outcome": "CO1: Demonstrate understanding...",
                "bloom_level": "Understand",
                "save_to_json": True
            }
        }
    })

@app.route('/generate-questions', methods=['POST'])
def api_generate_questions():
    global chunks, embeddings
    
    if chunks is None or embeddings is None:
        initialize_vector_db()
    
    # Get request data
    data = request.get_json()
    
    if not data or 'course_outcome' not in data or 'bloom_level' not in data:
        return jsonify({
            "error": "Missing required parameters. Please provide 'course_outcome' and 'bloom_level'."
        }), 400
    
    selected_co = data['course_outcome']
    selected_bloom = data['bloom_level']
    
    try:
        # Generate questions
        best_chunk = semantic_search(selected_co, chunks, embeddings, top_k=1)[0]
        questions_text = generate_questions(best_chunk, selected_co, selected_bloom)
        
        # Parse the questions into the requested structure
        questions_dict = parse_questions(questions_text)
        
        # Save to JSON (optional)
        if data.get('save_to_json', False):
            save_to_json(selected_co, selected_bloom, questions_dict)
        
        # Return the generated questions
        return jsonify({
            "course_outcome": selected_co,
            "bloom_level": selected_bloom,
            "questions": questions_dict,
            "raw_text": questions_text
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    # Initialize vector database
    print("Initializing vector database...")
    initialize_vector_db()
    
    # Run the Flask app
    print("Starting API server...")
    app.run(host='0.0.0.0', port=5000, debug=False)