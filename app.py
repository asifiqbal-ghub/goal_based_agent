import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import re
import fitz  # PyMuPDF
import pandas as pd
import docx

# Supported File Types
SUPPORTED_TYPES = ["xlsx", "xls", "csv", "txt", "md", "pdf", "docx"]

# Load environment variables (local dev)
load_dotenv()

# Read secrets: prefer st.secrets (Streamlit Cloud), fall back to os.getenv (local)
def get_secret(key):
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key)

# Initialize Azure LLM
llm = AzureChatOpenAI(
    azure_deployment=get_secret("AZURE_OPENAI_MODEL"),
    openai_api_version=get_secret("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=get_secret("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=get_secret("AZURE_OPENAI_API_KEY"),
    temperature=0
)

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Global storage
application_info = {
    "name": None,
    "email": None,
    "skills": None
}

# Extract info from chat
def extract_application_info(text: str):

    name_match = re.search(
        r"(?:my name is|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        text,
        re.IGNORECASE
    )

    email_match = re.search(
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        text
    )

    skills_match = re.search(
        r"(?:skills are|i know|i can use)\s+(.+)",
        text,
        re.IGNORECASE
    )

    if name_match:
        application_info["name"] = name_match.group(1).title()

    if email_match:
        application_info["email"] = email_match.group(0)

    if skills_match:
        application_info["skills"] = skills_match.group(1)

    return "Got it"


# Extract text from any file
def extract_text_from_file(uploaded_file):

    file_type = uploaded_file.name.split(".")[-1].lower()

    # PDF
    if file_type == "pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""

        for page in doc:
            text += page.get_text()

        doc.close()
        return text

    # TXT / MD
    elif file_type in ["txt", "md"]:
        return uploaded_file.read().decode("utf-8")

    # CSV
    elif file_type == "csv":
        df = pd.read_csv(uploaded_file)
        return df.to_string()

    # Excel
    elif file_type in ["xlsx", "xls"]:
        df = pd.read_excel(uploaded_file)
        return df.to_string()

    # DOCX
    elif file_type == "docx":
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    else:
        return ""


# Extract info from Resume
def extract_info_from_cv(text: str):

    extracted_info = {
        "name": None,
        "email": None,
        "skills": None
    }

    # Name Extraction
    lines = text.split("\n")

    for line in lines[:5]:
        line = line.strip()

        if (
            len(line.split()) <= 4
            and not any(
                x in line.lower()
                for x in ["email", "phone", "mobile", "@"]
            )
        ):
            extracted_info["name"] = line
            break

    # Email
    email_match = re.search(
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        text
    )

    if email_match:
        extracted_info["email"] = email_match.group(0)

    # Skills
    skills_match = re.search(
        r"(Core Competencies|Skills|Technical Skills)(.*?)(Experience|Projects|Education|Certifications|$)",
        text,
        re.IGNORECASE | re.DOTALL
    )

    if skills_match:
        skills = skills_match.group(2)

        skills = skills.replace("\n", ", ")
        skills = skills.replace("•", "")
        skills = skills.replace("-", "")

        extracted_info["skills"] = re.sub(
            r"\s+",
            " ",
            skills.strip()
        )

    return extracted_info


# Goal checker
def check_application_goal(_: str):

    if all(application_info.values()):
        return (
            f"✅ You're ready!\n\n"
            f"Name: {application_info['name']}\n"
            f"Email: {application_info['email']}\n"
            f"Skills: {application_info['skills']}"
        )

    else:
        missing = [
            k for k, v in application_info.items()
            if not v
        ]

        return f"⏳ Still need: {', '.join(missing)}"


# Tools
tools = [

    Tool(
        name="extract_application_info",
        func=extract_application_info,
        description="Extract name email skills"
    ),

    Tool(
        name="check_application_goal",
        func=check_application_goal,
        description="Check completion"
    )

]


# Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False
)


# UI
st.set_page_config(
    page_title="🎯 Job Application Assistant",
    layout="centered"
)

st.title("🧠 Goal-Based Agent: Job Application Assistant")

st.markdown(
    "Tell me your **name**, **email**, and **skills**"
)


# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "goal_complete" not in st.session_state:
    st.session_state.goal_complete = False

if "download_ready" not in st.session_state:
    st.session_state.download_ready = False

if "application_summary" not in st.session_state:
    st.session_state.application_summary = ""


# Upload Resume
st.sidebar.header("📤 Upload Resume")

resume = st.sidebar.file_uploader(
    "Upload your resume",
    type=SUPPORTED_TYPES
)

if resume:

    st.sidebar.success("Resume uploaded!")

    text = extract_text_from_file(resume)

    extracted = extract_info_from_cv(text)

    for key in application_info:
        if extracted[key]:
            application_info[key] = extracted[key]

    st.sidebar.info("🔍 Extracted Info")

    for key, value in application_info.items():
        st.sidebar.markdown(
            f"**{key.capitalize()}**: {value}"
        )


# Reset
if st.sidebar.button("Reset Chat"):

    st.session_state.chat_history.clear()
    st.session_state.goal_complete = False
    st.session_state.download_ready = False

    for key in application_info:
        application_info[key] = None

    st.rerun()


# Chat
user_input = st.chat_input("Type here...")


if user_input:

    st.session_state.chat_history.append(
        ("user", user_input)
    )

    extract_application_info(user_input)

    response = agent.invoke({
        "input": user_input
    })

    bot_reply = response["output"]

    st.session_state.chat_history.append(
        ("bot", bot_reply)
    )

    goal_status = check_application_goal("check")

    st.session_state.chat_history.append(
        ("status", goal_status)
    )

    if "ready" in goal_status.lower():

        st.session_state.goal_complete = True

        summary = (
            f"Name: {application_info['name']}\n"
            f"Email: {application_info['email']}\n"
            f"Skills: {application_info['skills']}"
        )

        st.session_state.application_summary = summary
        st.session_state.download_ready = True


# Chat UI
for sender, message in st.session_state.chat_history:

    if sender == "user":
        with st.chat_message("user"):
            st.markdown(message)

    elif sender == "bot":
        with st.chat_message("assistant"):
            st.markdown(message)

    elif sender == "status":
        st.info(message)


# Final
if st.session_state.goal_complete:
    st.success("🎉 Application Ready")


# Download
if st.session_state.download_ready:

    st.download_button(
        "📥 Download Summary",
        data=st.session_state.application_summary,
        file_name="application_summary.txt"
    )