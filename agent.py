from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware

def build_agent(tools):
    model = ChatOllama(model="qwen3:1.7b")
    SYSTEM_PROMPT = """
        You are a Gmail assistant.

        Before using tools, check whether the user gave enough information.

        Rules:
        - To send an email, you need: recipient email and a body/message intent.
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
