import streamlit as st
from PyPDF2 import PdfReader
import re
import os
import json
import pandas as pd
import math
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# from dotenv import load_dotenv

# Load environment variables
# load_dotenv()


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# Securely load your OpenAI API Key from .env
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# ---- MCQ Parser ----

def parse_teacher_mcqs(pdf_file):
    text = ""
    for page in PdfReader(pdf_file).pages:
        text += page.extract_text() + "\n"

    mcq_pattern = re.compile(r"\d+\.\s*(.*?)\n\(A\)\s*(.*?)\s*\(B\)\s*(.*?)\s*\(C\)\s*(.*?)\s*\(D\)\s*(.*?)\n",
                             re.DOTALL)
    answer_key_pattern = re.findall(r"\d+\.\s*([A-D])", text)

    matches = mcq_pattern.findall(text)
    mcqs = []
    for i, match in enumerate(matches):
        if i < len(answer_key_pattern):
            mcqs.append({
                "question": match[0].strip(),
                "options": {
                    "A": match[1].strip(),
                    "B": match[2].strip(),
                    "C": match[3].strip(),
                    "D": match[4].strip()
                },
                "answer": answer_key_pattern[i]
            })
    return mcqs


# ---- Textbook Chunking ----
def chunk_textbook(pdf_file):
    text = ""
    for page in PdfReader(pdf_file).pages:
        text += page.extract_text() + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


# ---- FAISS Indexing ----
def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embedding=embeddings)


# ---- Topic Matching ----
def find_matching_teacher_mcqs(topic, teacher_mcqs):
    topic = topic.lower()
    matches = []
    for mcq in teacher_mcqs:
        qtext = mcq["question"].lower()
        if topic in qtext:
            matches.append(mcq)
    return matches[:2]


# ---- Prompt Construction ----
def build_prompt(examples, context, difficulty_counts):
    example_text = ""
    for ex in examples:
        correct = ex['options'][ex['answer']]
        example_text += f'''{{
  "question": "{ex['question']}",
  "options": [
    "{ex['options']['A']}",
    "{ex['options']['B']}",
    "{ex['options']['C']}",
    "{ex['options']['D']}"
  ],
  "correct_option": "{correct}",
  "explanation": "This is based on prior knowledge.",
  "difficulty": "easy"
}},\n'''

    difficulty_instructions = "\n".join([
        f"- {count} {level} questions"
        for level, count in difficulty_counts.items() if count > 0
    ])

    prompt_template = ChatPromptTemplate.from_template("""
You are a Social Studies teacher creating multiple-choice questions (MCQs) based on the following textbook content.

Your task:
- Create the total number of MCQs as per difficulty levels below.
- Each question must include:
  - "question": clear question string
  - "options": list of 4 strings
  - "correct_option": one of the options
  - "explanation": 1-2 sentence reason
  - "difficulty": one of ["easy", "medium", "hard"]

Difficulty breakdown:
{difficulty_instructions}

### Examples (JSON Format)
[{examples}]

### Context:
{context}

### Now generate the MCQs as a valid JSON array.
The output must start with `[` and end with `]`, and must be fully parseable by `json.loads()`.
Do not include any explanation or extra text outside the JSON.
""")

    return prompt_template.format(
        examples=example_text,
        context=context,
        difficulty_instructions=difficulty_instructions
    )


# ---- Streamlit UI ----
st.set_page_config(page_title="RAG MCQ Generator", layout="wide")
st.title("UGHS - Content Generation")

textbook_pdf = st.file_uploader("📥 Upload Textbook PDF", type="pdf")
teacher_pdf = st.file_uploader("📥 Upload Teacher MCQs PDF", type="pdf")

if textbook_pdf and teacher_pdf:
    st.success("PDFs uploaded successfully!")

    teacher_mcqs = parse_teacher_mcqs(teacher_pdf)
    chunks = chunk_textbook(textbook_pdf)
    vectorstore = build_vectorstore(chunks)

    mode = st.radio("Choose generation mode:", ["Topic-wise", "Full Chapter"])

    if mode == "Topic-wise":
        topic = st.text_input("🎯 Enter topic to generate MCQs (e.g., Himalayas)", value="Himalayas")

    col1, col2, col3 = st.columns(3)
    with col1:
        easy_count = st.number_input("🟢 Easy MCQs", min_value=0, max_value=50, value=10)
    with col2:
        medium_count = st.number_input("🟡 Medium MCQs", min_value=0, max_value=50, value=10)
    with col3:
        hard_count = st.number_input("🔴 Hard MCQs", min_value=0, max_value=50, value=10)

    difficulty_counts = {
        "easy": easy_count,
        "medium": medium_count,
        "hard": hard_count
    }

    total_mcqs = sum(difficulty_counts.values())

    if st.button("🚀 Generate MCQs"):
        llm = ChatOpenAI(temperature=0.7)

        if mode == "Topic-wise":
            retriever = vectorstore.as_retriever()
            relevant_docs = retriever.get_relevant_documents(topic)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            matching_examples = find_matching_teacher_mcqs(topic, teacher_mcqs)
            prompt = build_prompt(matching_examples, context, difficulty_counts)
            response = llm.invoke(prompt)
            responses = [response.content]
        else:
            matching_examples = teacher_mcqs[:2]
            num_segments = 3
            segment_size = math.ceil(len(chunks) / num_segments)
            responses = []
            for i in range(num_segments):
                segment_chunks = chunks[i * segment_size: (i + 1) * segment_size]
                segment_context = "\n".join(segment_chunks)
                segment_prompt = build_prompt(matching_examples, segment_context, {
                    "easy": easy_count // num_segments,
                    "medium": medium_count // num_segments,
                    "hard": hard_count // num_segments
                })
                segment_response = llm.invoke(segment_prompt)
                responses.append(segment_response.content)

        st.subheader("🧠 Final MCQ Set (Reused + Generated)")

        if matching_examples:
            st.markdown("### ♻️ Reused Teacher MCQs")
            for ex in matching_examples:
                st.markdown(f"**Q:** {ex['question']}")
                for key, val in ex['options'].items():
                    st.markdown(f"- {key}) {val}")
                st.markdown(f"**Answer:** {ex['answer']}")
                st.markdown(f"**Explanation:** (original style — explanation assumed)\n\n---")

        st.markdown("### ✨ Generated MCQs")
        all_mcqs = []
        for content in responses:
            try:
                mcqs = json.loads(content)
            except json.JSONDecodeError:
                try:
                    json_start = content.find('[')
                    json_end = content.rfind(']') + 1
                    clean_str = content[json_start:json_end]
                    mcqs = json.loads(clean_str)
                except Exception:
                    st.error("❌ Could not parse one of the segment responses as valid JSON.")
                    st.write(content)
                    mcqs = []
            all_mcqs.extend(mcqs)

        if all_mcqs:
            st.json(all_mcqs)
            json_str = json.dumps(all_mcqs, indent=2)
            st.download_button("⬇️ Download JSON", data=json_str, file_name="mcqs.json", mime="application/json")
            df = pd.DataFrame(all_mcqs)
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download CSV", data=csv, file_name="mcqs.csv", mime="text/csv")
