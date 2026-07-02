import base64
from email.mime.text import MIMEText

from googleapiclient.discovery import build


class GmailClient:
    def __init__(self, credentials):
        self.service = build("gmail", "v1", credentials=credentials)

    def search_messages(self, query: str = "", max_results: int = 5):
        result = self.service.users().messages().list(
            userId="me",
            q=query,
            maxResults=max_results,
        ).execute()

        return result.get("messages", [])

    def get_message(self, message_id: str):
        return self.service.users().messages().get(
            userId="me",
            id=message_id,
            format="full",
        ).execute()

    def get_header(self, message, name: str):
        headers = message.get("payload", {}).get("headers", [])
        for header in headers:
            if header["name"].lower() == name.lower():
                return header["value"]
        return ""

    def get_body(self, payload):
        if "body" in payload and payload["body"].get("data"):
            data = payload["body"]["data"]
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

        for part in payload.get("parts", []):
            body = self.get_body(part)
            if body:
                return body

        return ""

    def read_email(self, message_id: str):
        message = self.get_message(message_id)
        payload = message.get("payload", {})

        return {
            "id": message_id,
            "from": self.get_header(message, "From"),
            "subject": self.get_header(message, "Subject"),
            "date": self.get_header(message, "Date"),
            "internal_date": message.get("internalDate", ""),
            "labels": message.get("labelIds", []),
            "snippet": message.get("snippet", ""),
            "body": self.get_body(payload),
        }

    def send_email(self, to: str, subject: str, body: str):
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        sent = self.service.users().messages().send(
            userId="me",
            body={"raw": raw},
        ).execute()

        return sent["id"]
