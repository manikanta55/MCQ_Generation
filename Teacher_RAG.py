import streamlit as st
from PyPDF2 import PdfReader
import os
import json
import pandas as pd
import math
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ---- Previous Year MCQ Parser ----
def parse_previous_year_mcqs(csv_file):
    df = pd.read_csv(csv_file)
    mcqs = []
    for _, row in df.iterrows():
        mcqs.append({
            "question": row["Question"],
            "options": {
                "A": row["option_A"],
                "B": row["option_B"],
                "C": row["option_C"],
                "D": row["option_D"]
            },
            "answer": row["answer"],
            "difficulty": str(row.get("difficulty", "easy")).lower(),
            "source": "previous_year"
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
  "explanation": "This is based on previous year exam pattern.",
  "difficulty": "{ex.get('difficulty', 'easy')}"
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

def select_mcqs_by_difficulty(mcqs, difficulty, n):
    filtered = [q for q in mcqs if q["difficulty"] == difficulty]
    return filtered[:n]

def tag_generated_mcqs(mcqs):
    for mcq in mcqs:
        mcq["source"] = "generated"
    return mcqs

# ---- Streamlit UI ----
st.set_page_config(page_title="MCQ Generator", layout="wide")
st.title("UGHS - MCQ Generator (Chapter + Previous Year Only)")

textbook_pdf = st.file_uploader("📥 Upload Chapter PDF", type="pdf")
previous_year_csv = st.file_uploader("📥 Upload Previous Year MCQs (CSV)", type="csv")

if textbook_pdf and previous_year_csv:
    st.success("Both files uploaded successfully!")

    previous_mcqs = parse_previous_year_mcqs(previous_year_csv)
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
        "easy": int(easy_count),
        "medium": int(medium_count),
        "hard": int(hard_count)
    }

    if st.button("🚀 Generate MCQs"):
        llm = ChatOpenAI(temperature=0.7)

        if mode == "Topic-wise":
            retriever = vectorstore.as_retriever()
            relevant_docs = retriever.get_relevant_documents(topic)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            prev_examples = [q for q in previous_mcqs if topic.lower() in q["question"].lower()][:2]

            prompt = build_prompt(prev_examples, context, difficulty_counts)
            response = llm.invoke(prompt)
            responses = [response.content]
            generated_mcqs = []
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
                tag_generated_mcqs(mcqs)
                generated_mcqs.extend(mcqs)
            all_mcqs = generated_mcqs  # (Or mix in matched previous if desired)

        else:  # Full Chapter
            prev_half = {k: v // 2 for k, v in difficulty_counts.items()}
            gen_half = {k: v - prev_half[k] for k, v in difficulty_counts.items()}

            # Gather previous year MCQs (up to 50% for each difficulty)
            reused_mcqs = []
            for level in ["easy", "medium", "hard"]:
                reused_mcqs.extend(select_mcqs_by_difficulty(previous_mcqs, level, prev_half[level]))

            prompt = build_prompt(reused_mcqs[:2], "\n".join(chunks), gen_half)
            response = llm.invoke(prompt)
            responses = [response.content]
            generated_mcqs = []
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
                tag_generated_mcqs(mcqs)
                generated_mcqs.extend(mcqs)

            all_mcqs = reused_mcqs + generated_mcqs

        st.subheader("🧠 Final MCQ Set (Previous Year + Generated, 50:50 mix)")
        if all_mcqs:
            st.json(all_mcqs)
            json_str = json.dumps(all_mcqs, indent=2)
            st.download_button("⬇️ Download JSON", data=json_str, file_name="mcqs.json", mime="application/json")
            df = pd.DataFrame(all_mcqs)
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download CSV", data=csv, file_name="mcqs.csv", mime="text/csv")
