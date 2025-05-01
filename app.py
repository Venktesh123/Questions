import os
import streamlit as st
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Load Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please check your .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Load sentence transformer model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

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

# Save and Load vector index
def save_vector_data(index, chunks, embeddings):
    faiss.write_index(index, "faiss_index.index")
    np.save("chunks.npy", np.array(chunks))
    np.save("embeddings.npy", embeddings)

def load_vector_data():
    if os.path.exists("faiss_index.index") and os.path.exists("chunks.npy") and os.path.exists("embeddings.npy"):
        index = faiss.read_index("faiss_index.index")
        chunks = np.load("chunks.npy", allow_pickle=True).tolist()
        embeddings = np.load("embeddings.npy")
        return index, chunks, embeddings
    return None, None, None

# Build Vector Index
def build_vector_index(chunks):
    embeddings = embed_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, chunks, embeddings

# Semantic search
def semantic_search(co_text, index, chunks, embeddings, top_k=1):
    co_embedding = embed_model.encode([co_text])
    distances, indices = index.search(np.array(co_embedding), top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
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

# Streamlit App
def main():
    st.title("Course Outcome & Bloom's Level Based Question Generator")

    # Load course content and course outcomes
    transcript = load_file("cleaned_transcript.txt")
    course_outcomes = load_file("course_outcomes.txt")
    co_list = course_outcomes.strip().split("\n")
    bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

    # Load or build vector DB
    index, chunks, embeddings = load_vector_data()
    if index is None:
        st.info("Building vector database... please wait")
        chunks = chunk_text(transcript)
        index, chunks, embeddings = build_vector_index(chunks)
        save_vector_data(index, chunks, embeddings)
        st.success("Vector database built and cached")
    else:
        st.success("Loaded cached vector database")

    # Select CO and Bloom Level
    selected_co = st.selectbox("Select Course Outcome:", co_list)
    selected_bloom = st.selectbox("Select Bloom's Level:", bloom_levels)

    if st.button("Generate Question"):
        with st.spinner("Retrieving content and generating questions..."):
            try:
                best_chunk = semantic_search(selected_co, index, chunks, embeddings, top_k=1)[0]
                questions_text = generate_questions(best_chunk, selected_co, selected_bloom)
                
                # Parse the questions into the requested structure
                questions_dict = parse_questions(questions_text)

                st.subheader("Generated Questions")
                
                # Display objective questions
                st.write("**Objective Questions:**")
                for i, q in enumerate(questions_dict["objective"], 1):
                    st.write(f"{i}. {q}")
                
                # Display subjective questions
                st.write("**Short Answer Questions:**")
                for i, q in enumerate(questions_dict["subjective"], 1):
                    st.write(f"{i}. {q}")

                # Save to JSON
                save_to_json(selected_co, selected_bloom, questions_dict)

                st.success("Questions saved to generated_questions.json")

                # Download buttons
                st.download_button(
                    "Download Latest Questions (Text)",
                    questions_text,
                    file_name="latest_generated_questions.txt"
                )

                # Option to download full JSON
                with open("generated_questions.json", "r", encoding="utf-8") as f:
                    st.download_button(
                        "Download Full Questions (JSON)",
                        f,
                        file_name="generated_questions.json",
                        mime="application/json"
                    )

            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()