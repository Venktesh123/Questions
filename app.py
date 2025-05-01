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

# Load Gemini API Key - check both .env and environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please add it to your environment variables or .env file.")
    # Don't stop the app, but show a warning
    st.warning("The app will not be able to generate questions without an API key.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Load sentence transformer model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the embedding model
try:
    embed_model = load_embedding_model()
except Exception as e:
    st.error(f"Error loading the embedding model: {e}")
    embed_model = None

# Load Files
def load_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        st.error(f"Error loading file {file_path}: {e}")
        return ""

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
    try:
        faiss.write_index(index, "faiss_index.index")
        np.save("chunks.npy", np.array(chunks))
        np.save("embeddings.npy", embeddings)
    except Exception as e:
        st.error(f"Error saving vector data: {e}")

def load_vector_data():
    try:
        if os.path.exists("faiss_index.index") and os.path.exists("chunks.npy") and os.path.exists("embeddings.npy"):
            index = faiss.read_index("faiss_index.index")
            chunks = np.load("chunks.npy", allow_pickle=True).tolist()
            embeddings = np.load("embeddings.npy")
            return index, chunks, embeddings
    except Exception as e:
        st.error(f"Error loading vector data: {e}")
    return None, None, None

# Build Vector Index
def build_vector_index(chunks):
    try:
        embeddings = embed_model.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        return index, chunks, embeddings
    except Exception as e:
        st.error(f"Error building vector index: {e}")
        return None, chunks, None

# Semantic search
def semantic_search(co_text, index, chunks, embeddings, top_k=1):
    try:
        co_embedding = embed_model.encode([co_text])
        distances, indices = index.search(np.array(co_embedding), top_k)
        retrieved_chunks = [chunks[i] for i in indices[0]]
        return retrieved_chunks
    except Exception as e:
        st.error(f"Error in semantic search: {e}")
        return ["Error retrieving content. Please try again."]

# Question Generation
def generate_questions(retrieved_content, co_text, bloom_level):
    if not GOOGLE_API_KEY:
        return "API key is missing. Cannot generate questions."
    
    try:
        prompt_parts = [
            "You are a Question Generator Agent.",
            f"Course Outcome (CO): {co_text}",
            f"Bloom's Taxonomy Level: {bloom_level}",
            "Based on the content below, generate two questions:",
            "- One Objective Type",
            "- One Short Answer Type",
            "Content:\n" + retrieved_content,
            "\nOnly output the questions in the following format:",
            "Objective Question:\n1. <question>",
            "Short Answer Question:\n1. <question>"
        ]

        full_prompt = "\n".join(prompt_parts)

        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(full_prompt)

        output = response.text.strip()
        if "Objective Question" in output:
            output = output.split("Objective Question", 1)[1]
            output = "Objective Question" + output.strip()
        return output
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return f"Error: {str(e)}"

# Save generated questions to JSON
def save_to_json(selected_co, selected_bloom, questions, json_file="generated_questions.json"):
    try:
        # Extract Objective and Short Answer parts
        objective_q = ""
        short_answer_q = ""

        if "Short Answer Question:" in questions:
            parts = questions.split("Short Answer Question:")
            objective_q = parts[0].replace("Objective Question:", "").strip()
            short_answer_q = parts[1].strip()

        new_entry = {
            "course_outcome": selected_co,
            "bloom_level": selected_bloom,
            "objective_question": objective_q,
            "short_answer_question": short_answer_q
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
        
        return True
    except Exception as e:
        st.error(f"Error saving to JSON: {e}")
        return False

# Streamlit App
def main():
    st.title("Course Outcome & Bloom's Level Based Question Generator")
    
    # Add info about deployment
    st.info("This application is deployed on Azure Web App Service.")

    # Check for data files
    try:
        # Load course content and course outcomes
        transcript = load_file("cleaned_transcript.txt")
        if not transcript:
            st.error("Failed to load transcript file. Please check if cleaned_transcript.txt exists.")
            return
        
        course_outcomes = load_file("course_outcomes.txt")
        if not course_outcomes:
            st.error("Failed to load course outcomes file. Please check if course_outcomes.txt exists.")
            return
        
        co_list = [co.strip() for co in course_outcomes.strip().split("\n") if co.strip()]
        bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

        # Load or build vector DB
        index, chunks, embeddings = load_vector_data()
        if index is None:
            st.info("Building vector database... please wait")
            chunks = chunk_text(transcript)
            index, chunks, embeddings = build_vector_index(chunks)
            if index:
                save_vector_data(index, chunks, embeddings)
                st.success("Vector database built and cached")
            else:
                st.error("Failed to build vector database.")
                return
        else:
            st.success("Loaded cached vector database")

        # Select CO and Bloom Level
        selected_co = st.selectbox("Select Course Outcome:", co_list)
        selected_bloom = st.selectbox("Select Bloom's Level:", bloom_levels)

        if st.button("Generate Question"):
            if not GOOGLE_API_KEY:
                st.error("Cannot generate questions: Google API Key is missing. Please add it to your environment variables.")
                return
                
            with st.spinner("Retrieving content and generating question..."):
                try:
                    best_chunk = semantic_search(selected_co, index, chunks, embeddings, top_k=1)[0]
                    questions = generate_questions(best_chunk, selected_co, selected_bloom)

                    st.subheader("Generated Questions")
                    st.write(questions)

                    # Save to JSON
                    if save_to_json(selected_co, selected_bloom, questions):
                        st.success("Question saved to generated_questions.json")

                        # Download buttons
                        st.download_button(
                            "Download Latest Question (Text)",
                            questions,
                            file_name="latest_generated_question.txt"
                        )

                        # Option to download full JSON
                        if os.path.exists("generated_questions.json"):
                            with open("generated_questions.json", "r", encoding="utf-8") as f:
                                st.download_button(
                                    "Download Full Questions (JSON)",
                                    f,
                                    file_name="generated_questions.json",
                                    mime="application/json"
                                )

                except Exception as e:
                    st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()