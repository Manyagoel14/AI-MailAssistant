import uuid

import streamlit as st
from langchain.messages import HumanMessage
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from agent import build_agent
from auth import get_credentials
from gmail_client import GmailClient
from tools import build_gmail_tools, get_stale_unread_notice, load_reminders


st.set_page_config(page_title="Gmail AI Assistant", page_icon="@", layout="wide")


def init_assistant():
    creds = get_credentials()
    gmail = GmailClient(creds)
    tools = build_gmail_tools(gmail)
    agent = build_agent(tools)
    return {
        "gmail": gmail,
        "agent": agent,
        "config": {"configurable": {"thread_id": f"gmail-assistant-{uuid.uuid4()}"}},
    }


def run_agent(input_data):
    assistant = st.session_state.assistant
    final_response = None
    tool_events = []
    assistant_text = ""

    with st.status("Working...", expanded=False) as status:
        for chunk in assistant["agent"].stream(
            input_data,
            config=assistant["config"],
            stream_mode="values",
        ):
            final_response = chunk

            if "messages" not in chunk:
                continue

            message = chunk["messages"][-1]

            if isinstance(message, AIMessage):
                for tool_call in getattr(message, "tool_calls", []) or []:
                    tool_events.append(f"Using `{tool_call['name']}`")
                content = getattr(message, "content", "")
                if content:
                    assistant_text = content

            elif isinstance(message, ToolMessage):
                tool_events.append(f"Finished `{message.name}`")

        status.update(label="Done", state="complete")

    return final_response, assistant_text, tool_events


def add_message(role, content, tool_events=None):
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "tool_events": tool_events or [],
        }
    )


def show_message(message):
    with st.chat_message(message["role"]):
        if message.get("tool_events"):
            with st.expander("Tool activity", expanded=False):
                for event in message["tool_events"]:
                    st.write(event)
        st.markdown(message["content"])


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None
if "pending_draft_id" not in st.session_state:
    st.session_state.pending_draft_id = None


def get_pending_args(request):
    args = request.get("args", {})
    return {
        "to": args.get("to", ""),
        "subject": args.get("subject", ""),
        "body": args.get("body", ""),
        "sender_name": args.get("sender_name", ""),
    }


def load_pending_draft(request):
    draft_id = f"{request.get('name', '')}:{request.get('description', '')}"
    if st.session_state.pending_draft_id == draft_id:
        return

    args = get_pending_args(request)
    st.session_state.pending_draft_id = draft_id
    st.session_state.draft_to = args["to"]
    st.session_state.draft_subject = args["subject"]
    st.session_state.draft_body = args["body"]
    st.session_state.draft_sender_name = args["sender_name"]


def current_draft_args():
    return {
        "to": st.session_state.get("draft_to", "").strip(),
        "subject": st.session_state.get("draft_subject", "").strip(),
        "body": st.session_state.get("draft_body", "").strip(),
        "sender_name": st.session_state.get("draft_sender_name", "").strip(),
    }


st.title("Gmail AI Assistant")
st.caption("Search mail, read messages, detect action items, and save follow-up reminders locally.")

with st.sidebar:
    if st.button("Reconnect", use_container_width=True):
        st.session_state.pop("assistant", None)
        st.session_state.pop("stale_notice_loaded", None)
        st.rerun()

    st.subheader("Quick Actions")
    if st.button("Check stale unread", use_container_width=True):
        response, text, events = run_agent(
            {"messages": [HumanMessage(content="Check unread emails older than 7 days.")]}
        )
        add_message("assistant", text or "No response.", events)
        st.rerun()

    if st.button("List reminders", use_container_width=True):
        reminders = load_reminders()
        if reminders:
            content = "\n\n".join(
                f"**{item['remind_at']}**\n\n"
                f"{item['note']}\n\n"
                f"{item['email']['subject']} from {item['email']['from']}"
                for item in reminders
            )
        else:
            content = "No reminders saved yet."
        add_message("assistant", content)
        st.rerun()

    st.divider()
    st.write("Try:")
    st.code("Find emails from Internshala from last week")
    st.code("Read mail from google with subject security alert")
    st.code("Remind me to reply to this tomorrow")


if "assistant" not in st.session_state:
    try:
        st.session_state.assistant = init_assistant()
    except ValueError as error:
        st.error(str(error))
        st.stop()

if "stale_notice_loaded" not in st.session_state:
    notice = get_stale_unread_notice(st.session_state.assistant["gmail"], days_old=7, max_results=5)
    if notice:
        add_message("assistant", notice)
    st.session_state.stale_notice_loaded = True


for message in st.session_state.messages:
    show_message(message)


if st.session_state.pending_interrupt:
    request = st.session_state.pending_interrupt
    load_pending_draft(request)

    with st.chat_message("assistant"):
        st.warning("Approval needed before sending email.")

        st.text_input("To", key="draft_to")
        st.text_input("Subject", key="draft_subject")
        st.text_input("Your name for sign-off", key="draft_sender_name")
        st.text_area(
            "Email body",
            key="draft_body",
            height=220,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            approve = st.button("Approve Send", type="primary", use_container_width=True)
        with col2:
            reject = st.button("Reject", use_container_width=True)
        with col3:
            save_edit = st.button("Save Draft", use_container_width=True)

        edited_args = current_draft_args()

        if approve or reject:
            if approve:
                missing = [
                    label
                    for key, label in [
                        ("to", "recipient email"),
                        ("subject", "subject"),
                        ("body", "email body"),
                        ("sender_name", "your name for sign-off"),
                    ]
                    if not edited_args[key]
                ]

                if missing:
                    st.error(f"Please add: {', '.join(missing)}.")
                    st.stop()

                decision = {
                    "type": "edit",
                    "edited_action": {
                        "name": request["name"],
                        "args": edited_args,
                    },
                }
            else:
                decision = {"type": "reject"}

            response, text, events = run_agent(
                Command(resume={"decisions": [decision]})
            )
            st.session_state.pending_interrupt = None
            st.session_state.pending_draft_id = None
            add_message("assistant", text or ("Email sent." if approve else "Send rejected."), events)
            st.rerun()
        elif save_edit:
            st.session_state.pending_interrupt["args"] = edited_args
            st.toast("Draft updated. Review it, then approve when ready.")


prompt = st.chat_input("Ask about your Gmail...")
if prompt:
    add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    response, text, events = run_agent({"messages": [HumanMessage(content=prompt)]})

    if response and "__interrupt__" in response:
        st.session_state.pending_interrupt = response["__interrupt__"][0].value["action_requests"][0]
        add_message("assistant", "I need your approval before sending that email.", events)
        st.rerun()

    add_message("assistant", text or "Done.", events)
    st.rerun()
