import os
import base64
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import LLMChain
import json
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
Compare the following job description and candidate CV.

JOB DESCRIPTION:
{job_description}

CANDIDATE CV:
{candidate_cv}

Perform the following analysis and respond ONLY in the following JSON format:

{{
  "matching_percentage": <number between 0-100>,
  "experience_years": <number>,
  "missing_skills": [list of missing key skills],
  "fit_summary": "<short summary>"
}}
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["job_description", "candidate_cv"]
    )
    chain = LLMChain(
        llm=model,
        prompt=prompt
    )
    return chain

def extract_json_block(text):
    """
    Extracts JSON block from the model output.
    Handles cases like:
    ```json
    { ... }
    ```
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
    # Find the first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return text

def user_input(jd_text, cv_text):
    chain = get_conversational_chain()

    response = chain({
        "job_description": jd_text,
        "candidate_cv": cv_text
    })

    output_text = response["text"]

    json_text = extract_json_block(output_text)

    try:
        parsed = json.loads(json_text)

        st.markdown("### ✅ MATCH REPORT", unsafe_allow_html=True)

        st.markdown(f"**🔹 MATCHING PERCENTAGE %:** `{parsed.get('matching_percentage', 'N/A')}%`")
        st.markdown(f"**🔹 CANDIDATE EXPERIENCE-(Years):** `{parsed.get('experience_years', 'N/A')}`")

        missing_skills = parsed.get("missing_skills", [])
        if missing_skills:
            st.markdown("**🔹 MISSING SKILLS:**")
            st.markdown("- " + "\n- ".join(missing_skills))
        else:
            st.markdown("**🔹 MISSING SKILLS:** None")

        st.markdown("**🔹 FIT SUMMARY:**")
        st.markdown(f"> {parsed.get('fit_summary', 'No summary provided.')}")
    
    except json.JSONDecodeError:
        st.error("❌ Model did not return valid JSON.")
        st.write(output_text)



def main():
    st.set_page_config("CV MATCHING APP")
    st.header("AI CV Matching")

    with open("background3.gif", "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/gif;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    user_question = st.text_input("PASTE YOUR JD")

    # MOVE pdf_docs outside sidebar to make sure it's accessible:
    pdf_docs = None

    with st.sidebar:
        st.title("")
        st.image("thumbnail.png")

        pdf_docs = st.file_uploader("After Uploading Click on Submit", accept_multiple_files=True)
        if st.button("UPLOAD"):
            with st.spinner("Processing..........."):
                st.toast('Upload completed.!')

                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Only run if both JD and CV are provided
    if user_question and pdf_docs:
        # Load the processed vector store and extract CV text
        raw_text = get_pdf_text(pdf_docs)
        user_input(user_question, raw_text)

if __name__ == "__main__":
    main()
