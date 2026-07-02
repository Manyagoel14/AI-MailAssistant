from langchain.messages import HumanMessage
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
import uuid


from auth import get_credentials
from gmail_client import GmailClient
from tools import build_gmail_tools, get_stale_unread_notice
from agent import build_agent



def stream_agent(agent, input_data, config):
    final_response = None
    printed = ""
    tool_started = False

    print("thinking", end="", flush=True)

    for chunk in agent.stream(input_data, config=config, stream_mode="values"):
        final_response = chunk

        if not tool_started:
            print("\rAssistant: ", end="", flush=True)
            tool_started = True

        if "messages" not in chunk:
            continue

        message = chunk["messages"][-1]

        if isinstance(message, AIMessage):
            if getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    print(f"\n[Using tool: {tool_call['name']}] ", end="", flush=True)

            content = getattr(message, "content", "")

            if content and content != printed:
                new_text = content[len(printed):]
                print(new_text, end="", flush=True)
                printed = content

        elif isinstance(message, ToolMessage):
            print(f"\n[Tool finished: {message.name}]")
            print("Assistant: ", end="", flush=True)

    print()
    return final_response

def main():
    creds = get_credentials()
    gmail = GmailClient(creds)
    tools = build_gmail_tools(gmail)
    agent = build_agent(tools)

    config = {"configurable": {"thread_id": f"gmail-assistant-{uuid.uuid4()}"}}

    print("Gmail AI Assistant. Type 'exit' to quit.")
    stale_notice = get_stale_unread_notice(gmail, days_old=7, max_results=5)
    if stale_notice:
        print(f"\nAssistant: {stale_notice}")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in {"exit", "quit"}:
            break

        print("\nAssistant: ", end="", flush=True)

        response = stream_agent(
            agent,
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        if "__interrupt__" in response:
            request = response["__interrupt__"][0].value["action_requests"][0]

            print("\nApproval needed:")
            print(request)

            decision = input("Approve send? yes/no: ").strip().lower()

            if decision == "yes":
                print("\nAssistant: ", end="", flush=True)
                response = stream_agent(
                    agent,
                    Command(resume={"decisions": [{"type": "approve"}]}),
                    config=config,
                )
            else:
                print("\nAssistant: ", end="", flush=True)
                response = stream_agent(
                    agent,
                    Command(resume={"decisions": [{"type": "reject"}]}),
                    config=config,
                )


if __name__ == "__main__":
    main()
