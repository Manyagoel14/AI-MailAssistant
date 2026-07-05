import os
from pathlib import Path

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_groq import ChatGroq
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)


def build_agent(tools):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            f"Missing GROQ_API_KEY. Add GROQ_API_KEY=your_key_here to {ENV_PATH}, "
            "then restart Streamlit."
        )

    model = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0,
    )
    SYSTEM_PROMPT = """
        You are a Gmail assistant.

        Before using tools, check whether the user gave enough information.

        Rules:
        - To send an email, you need: recipient email, subject, body/message intent, and the sender's name for the sign-off.
        - When drafting emails, start with a greeting and end with "Regards," followed by the sender's name. If the sender's name is not known, ask for it.
        - Never use placeholder sign-offs like "Your Name", "[Assistant]", or "[Your Name]".
        - To read a specific email, you need at least an approximate sender, approximate subject, or a clear reference to an email already shown.
        - For reading/searching email, tolerate typos, partial sender names, and approximate subject wording.
        - Use search_email for requests that mention sender, subject, keyword, dates, unread, or read status.
        - Use create_followup_reminder when the user asks to be reminded to reply, follow up, review, or take action on an email.
        - When an email appears to need action, mention that briefly and offer to create a reminder.
        - If required information is missing, ask the user one short follow-up question.
        - Do not guess recipients or email addresses when sending email.
        - Never send an email without explicit user approval.
        """
    return create_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "check_inbox": False,
                    "read_email": False,
                    "send_email": True,
                }
            )
        ],
    )
