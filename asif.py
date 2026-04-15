from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import re

load_dotenv()

# Azure LLM
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_MODEL"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

application_info = {
    "name": None,
    "email": None,
    "skills": None
}


def extract_application_info(text: str) -> str:

    name_match = re.search(
        r"(?:my name is|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        text,
        re.IGNORECASE
    )

    email_match = re.search(
        r"\b[\w.-]+@[\w.-]+\.\w+\b",
        text
    )

    skills_match = re.search(
        r"(?:skills are|i know|i can use)\s+(.+)",
        text,
        re.IGNORECASE
    )

    response = []

    if name_match:
        application_info["name"] = name_match.group(1).title()
        response.append("✅ Name saved.")

    if email_match:
        application_info["email"] = email_match.group(0)
        response.append("✅ Email saved.")

    if skills_match:
        application_info["skills"] = skills_match.group(1).strip()
        response.append("✅ Skills saved.")

    if not any([name_match, email_match, skills_match]):
        return "❓ Please provide your name, email, or skills."

    return " ".join(response) + " Checking..."


def check_application_goal(_: str) -> str:

    if all(application_info.values()):
        return (
            f"✅ You're ready!\n"
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


tools = [

    Tool(
        name="extract_application_info",
        func=extract_application_info,
        description="Extract name email skills"
    ),

    Tool(
        name="check_application_goal",
        func=check_application_goal,
        description="Check completion",
        return_direct=True
    )

]


SYSTEM_PROMPT = """You are a helpful job application assistant.
Your goal is to collect the user's name, email, and skills.
Use tools to extract info.
Once all info is collected, stop.
"""


agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"system_message": SYSTEM_PROMPT}
)


print("📝 Job Application Assistant")
print("Tell me your name, email, and skills")


while True:

    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("👋 Bye!")
        break

    response = agent.invoke({"input": user_input})

    print("Bot:", response["output"])

    if "you're ready" in response["output"].lower():
        print("🎉 Application complete!")
        break