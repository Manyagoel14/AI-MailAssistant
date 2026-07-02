import json
import re
import unicodedata
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

from langchain.tools import tool


REMINDERS_FILE = Path("reminders.json")

STOP_WORDS = {
    "a",
    "an",
    "and",
    "email",
    "from",
    "mail",
    "me",
    "my",
    "of",
    "read",
    "subject",
    "the",
    "to",
    "with",
}


ACTION_WORDS = {
    "action required",
    "asap",
    "confirm",
    "deadline",
    "due",
    "follow up",
    "needed",
    "please reply",
    "respond",
    "response required",
    "review",
    "rsvp",
    "submit",
    "urgent",
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", value.lower()).strip()


def meaningful_tokens(value: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", normalize_text(value))
    return [token for token in tokens if token not in STOP_WORDS]


def fuzzy_score(needle: str, haystack: str) -> float:
    needle = normalize_text(needle)
    haystack = normalize_text(haystack)

    if not needle:
        return 1.0
    if not haystack:
        return 0.0
    if needle in haystack:
        return 1.0

    needle_tokens = meaningful_tokens(needle)
    haystack_tokens = meaningful_tokens(haystack)

    if not needle_tokens or not haystack_tokens:
        return SequenceMatcher(None, needle, haystack).ratio()

    token_scores = []
    for needle_token in needle_tokens:
        best = 0.0
        for haystack_token in haystack_tokens:
            if needle_token in haystack_token or haystack_token in needle_token:
                best = max(best, 0.95)
            else:
                best = max(best, SequenceMatcher(None, needle_token, haystack_token).ratio())
        token_scores.append(best)

    return sum(token_scores) / len(token_scores)


def build_search_query(sender_filter: str, subject_filter: str) -> str:
    tokens = meaningful_tokens(f"{sender_filter} {subject_filter}")
    return " ".join(tokens[:8])


def parse_search_date(date_filter: str) -> str:
    value = normalize_text(date_filter)
    today = datetime.now().date()

    if not value:
        return ""
    if value in {"today"}:
        return f"after:{today.strftime('%Y/%m/%d')}"
    if value in {"yesterday"}:
        yesterday = today - timedelta(days=1)
        return (
            f"after:{yesterday.strftime('%Y/%m/%d')} "
            f"before:{today.strftime('%Y/%m/%d')}"
        )
    if value in {"last week", "past week", "this week"}:
        start = today - timedelta(days=7)
        return f"after:{start.strftime('%Y/%m/%d')}"

    match = re.search(r"(older than|before)\s+(\d+)\s+days?", value)
    if match:
        cutoff = today - timedelta(days=int(match.group(2)))
        return f"before:{cutoff.strftime('%Y/%m/%d')}"

    match = re.search(r"(last|past)\s+(\d+)\s+days?", value)
    if match:
        start = today - timedelta(days=int(match.group(2)))
        return f"after:{start.strftime('%Y/%m/%d')}"

    match = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", value)
    if match:
        year, month, day = match.groups()
        return f"after:{int(year):04d}/{int(month):02d}/{int(day):02d}"

    return date_filter.strip()


def parse_reminder_time(remind_when: str) -> str:
    value = normalize_text(remind_when)
    now = datetime.now()

    if not value:
        return ""
    if value == "tomorrow":
        return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0).isoformat()
    if value == "today":
        return now.replace(hour=18, minute=0, second=0, microsecond=0).isoformat()
    if value in {"next week", "in a week"}:
        return (now + timedelta(days=7)).replace(hour=9, minute=0, second=0, microsecond=0).isoformat()

    match = re.search(r"in\s+(\d+)\s+(hour|hours|day|days)", value)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        delta = timedelta(hours=amount) if unit.startswith("hour") else timedelta(days=amount)
        return (now + delta).replace(second=0, microsecond=0).isoformat()

    return remind_when.strip()


def build_gmail_query(
    sender: str = "",
    subject: str = "",
    keyword: str = "",
    date_filter: str = "",
    read_state: str = "",
) -> str:
    parts = []

    if sender.strip():
        parts.append(f"from:({sender.strip()})")
    if subject.strip():
        parts.append(f"subject:({subject.strip()})")
    if keyword.strip():
        parts.append(keyword.strip())

    date_query = parse_search_date(date_filter)
    if date_query:
        parts.append(date_query)

    state = normalize_text(read_state)
    if state == "unread":
        parts.append("is:unread")
    elif state == "read":
        parts.append("is:read")

    return " ".join(parts)


def looks_actionable(email: dict) -> bool:
    text = normalize_text(f"{email.get('subject', '')} {email.get('snippet', '')} {email.get('body', '')[:500]}")
    return any(word in text for word in ACTION_WORDS)


def format_email(email: dict) -> str:
    return (
        f"ID: {email['id']}\n"
        f"From: {email['from']}\n"
        f"Subject: {email['subject']}\n\n"
        f"{email['body'] or email['snippet']}"
    )


def format_email_preview(email: dict, include_action_hint: bool = True) -> str:
    unread = "unread" if "UNREAD" in email.get("labels", []) else "read"
    action_hint = "\nAction hint: may need a reply/follow-up" if include_action_hint and looks_actionable(email) else ""
    return (
        f"ID: {email['id']}\n"
        f"From: {email['from']}\n"
        f"Subject: {email['subject']}\n"
        f"Date: {email.get('date', '')}\n"
        f"State: {unread}\n"
        f"Preview: {email['snippet']}"
        f"{action_hint}"
    )


def load_reminders() -> list[dict]:
    if not REMINDERS_FILE.exists():
        return []

    try:
        return json.loads(REMINDERS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_reminders(reminders: list[dict]) -> None:
    REMINDERS_FILE.write_text(
        json.dumps(reminders, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def resolve_email(gmail, message_id: str = "", sender_filter: str = "", subject_filter: str = ""):
    if message_id.strip():
        return gmail.read_email(message_id.strip())

    search_query = build_search_query(sender_filter, subject_filter)
    messages = gmail.search_messages(query=search_query, max_results=25)

    if not messages and search_query:
        messages = gmail.search_messages(query="", max_results=50)

    ranked_matches = []
    seen_ids = set()

    for msg in messages:
        if msg["id"] in seen_ids:
            continue
        seen_ids.add(msg["id"])
        email = gmail.read_email(msg["id"])

        sender_score = fuzzy_score(sender_filter, email["from"])
        subject_score = fuzzy_score(subject_filter, email["subject"])
        combined_score = (sender_score + subject_score) / 2

        if sender_score >= 0.72 and subject_score >= 0.72:
            ranked_matches.append((combined_score, email))

    ranked_matches.sort(key=lambda item: item[0], reverse=True)
    return ranked_matches


def get_stale_unread_notice(gmail, days_old: int = 7, max_results: int = 5) -> str:
    query = build_gmail_query(date_filter=f"older than {days_old} days", read_state="unread")
    messages = gmail.search_messages(query=query, max_results=max_results)

    if not messages:
        return ""

    emails = [gmail.read_email(msg["id"]) for msg in messages]
    return (
        f"Note: You have unread emails older than {days_old} days:\n\n"
        + "\n\n".join(format_email_preview(email, include_action_hint=False) for email in emails)
    )


def build_gmail_tools(gmail):
    @tool
    def check_inbox(query: str = "is:unread", max_results: int = 5) -> str:
        """Check recent Gmail messages using a Gmail search query."""
        messages = gmail.search_messages(query=query, max_results=max_results)

        if not messages:
            return "No emails found."

        emails = []

        for msg in messages:
            email = gmail.read_email(msg["id"])
            emails.append(
                f"ID: {email['id']}\n"
                f"From: {email['from']}\n"
                f"Subject: {email['subject']}\n"
                f"Preview: {email['snippet']}"
            )

        return "\n\n".join(emails)

    @tool
    def search_email(
        sender: str = "",
        subject: str = "",
        keyword: str = "",
        date_filter: str = "",
        read_state: str = "",
        max_results: int = 10,
    ) -> str:
        """Search Gmail by sender, subject, keyword, date phrase/Gmail date query, and read/unread state."""
        query = build_gmail_query(
            sender=sender,
            subject=subject,
            keyword=keyword,
            date_filter=date_filter,
            read_state=read_state,
        )
        messages = gmail.search_messages(query=query, max_results=max_results)

        if not messages:
            return f"No emails found for query: {query or '(recent mail)'}"

        emails = [gmail.read_email(msg["id"]) for msg in messages]
        return (
            f"Search query: {query or '(recent mail)'}\n\n"
            + "\n\n".join(format_email_preview(email) for email in emails)
        )

    @tool
    def send_email(to: str = "", subject: str = "", body: str = "") -> str:
        """Send an email. Requires recipient email, subject, and body."""

        missing = []

        if not to.strip():
            missing.append("recipient email")
        if not subject.strip():
            missing.append("subject")
        if not body.strip():
            missing.append("email body")

        if missing:
            return (
                "MISSING_INFO: Cannot send email yet. "
                f"Ask the user for: {', '.join(missing)}."
            )

        return gmail.send_email(to=to, subject=subject, body=body)
    
    @tool
    def read_email(sender_filter: str = "", subject_filter: str = "", message_id: str = "") -> str:
        """Read a specific email by message ID, approximate sender, or approximate subject."""

        if message_id.strip():
            email = gmail.read_email(message_id.strip())
            return format_email(email)

        if not sender_filter.strip() and not subject_filter.strip():
            return (
                "MISSING_INFO: Cannot identify which email to read. "
                "Ask the user for the sender, subject, or message ID."
            )

        ranked_matches = resolve_email(gmail, sender_filter=sender_filter, subject_filter=subject_filter)

        if not ranked_matches:
            return "No matching email found. Ask the user for a more specific sender or subject."

        best_score, best_email = ranked_matches[0]
        close_matches = [
            email
            for score, email in ranked_matches[1:4]
            if best_score - score < 0.08
        ]

        if close_matches and best_score < 0.92:
            options = [best_email, *close_matches]
            return (
                "I found a few possible matches. Ask the user which one to open:\n\n"
                + "\n\n".join(
                    f"ID: {email['id']}\n"
                    f"From: {email['from']}\n"
                    f"Subject: {email['subject']}\n"
                    f"Preview: {email['snippet']}"
                    for email in options
                )
            )

        return format_email(best_email)

    @tool
    def create_followup_reminder(
        remind_when: str,
        note: str = "Reply to this email",
        message_id: str = "",
        sender_filter: str = "",
        subject_filter: str = "",
    ) -> str:
        """Store a local follow-up reminder for an email using message ID or approximate sender/subject."""
        reminder_time = parse_reminder_time(remind_when)
        if not reminder_time:
            return "MISSING_INFO: Ask the user when they want to be reminded."

        ranked_matches = resolve_email(
            gmail,
            message_id=message_id,
            sender_filter=sender_filter,
            subject_filter=subject_filter,
        )

        if isinstance(ranked_matches, dict):
            email = ranked_matches
        elif ranked_matches:
            email = ranked_matches[0][1]
        else:
            return "MISSING_INFO: Ask the user which email the reminder is for."

        reminders = load_reminders()
        reminder = {
            "id": f"reminder-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "remind_at": reminder_time,
            "note": note,
            "email": {
                "id": email["id"],
                "from": email["from"],
                "subject": email["subject"],
                "date": email.get("date", ""),
                "snippet": email["snippet"],
            },
            "status": "pending",
        }
        reminders.append(reminder)
        save_reminders(reminders)

        return (
            "Reminder saved locally.\n"
            f"Reminder ID: {reminder['id']}\n"
            f"When: {reminder['remind_at']}\n"
            f"Email: {email['subject']} from {email['from']}"
        )

    @tool
    def list_followup_reminders(status: str = "pending") -> str:
        """List locally stored follow-up reminders."""
        reminders = load_reminders()
        state = normalize_text(status)

        if state and state != "all":
            reminders = [reminder for reminder in reminders if reminder.get("status") == state]

        if not reminders:
            return "No reminders found."

        return "\n\n".join(
            f"ID: {reminder['id']}\n"
            f"When: {reminder['remind_at']}\n"
            f"Status: {reminder['status']}\n"
            f"Note: {reminder['note']}\n"
            f"Email: {reminder['email']['subject']} from {reminder['email']['from']}"
            for reminder in reminders
        )

    @tool
    def check_stale_unread(days_old: int = 7, max_results: int = 5) -> str:
        """Find unread emails older than the given number of days."""
        notice = get_stale_unread_notice(gmail, days_old=days_old, max_results=max_results)
        if not notice:
            return f"No unread emails older than {days_old} days."
        return notice

    return [
        check_inbox,
        search_email,
        read_email,
        create_followup_reminder,
        list_followup_reminders,
        check_stale_unread,
        send_email,
    ]
