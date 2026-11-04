"""In-Reply-To and References, parsed only when the service asks for them."""

from email import message_from_bytes

from inference_core.core.email_config import FullEmailConfig
from inference_core.services.imap_service import (
    MAX_REFERENCES,
    ImapService,
    thread_header_fields,
)

RAW = (
    b"Message-ID: <reply@example.org>\r\n"
    b"From: Anna <anna@example.org>\r\n"
    b"To: office@example.com\r\n"
    b"Subject: Re: invoice\r\n"
    b"In-Reply-To: <first@example.com> (sent by Office)\r\n"
    b"References: <root@example.com>\r\n"
    b" <first@example.com>\r\n"
    b"\t<second@example.com> trailing words\r\n"
    b"Content-Type: text/plain\r\n"
    b"\r\n"
    b"Thanks.\r\n"
)


def _service(**kwargs) -> ImapService:
    return ImapService(config=FullEmailConfig.model_construct(), **kwargs)


def test_off_by_default_the_fields_stay_empty():
    parsed = _service()._parse_message(message_from_bytes(RAW), "7", "INBOX")
    assert parsed.in_reply_to is None
    assert parsed.references == []
    assert parsed.message_id == "<reply@example.org>"


def test_on_the_identifiers_are_kept_and_nothing_else():
    parsed = _service(thread_headers=True)._parse_message(
        message_from_bytes(RAW), "7", "INBOX"
    )
    assert parsed.in_reply_to == "<first@example.com>"
    assert parsed.references == [
        "<root@example.com>",
        "<first@example.com>",
        "<second@example.com>",
    ]


def test_a_message_with_neither_header():
    msg = message_from_bytes(b"Subject: hi\r\n\r\nbody\r\n")
    assert thread_header_fields(msg) == {"in_reply_to": None, "references": []}


def test_a_long_references_keeps_the_root_and_the_newest():
    ids = " ".join(f"<m{i}@example.com>" for i in range(50))
    msg = message_from_bytes(f"References: {ids}\r\n\r\nx\r\n".encode())
    refs = thread_header_fields(msg)["references"]
    assert len(refs) == MAX_REFERENCES
    assert refs[0] == "<m0@example.com>"
    assert refs[-1] == "<m49@example.com>"
    assert refs[1] == f"<m{50 - MAX_REFERENCES + 1}@example.com>"


def test_garbage_is_no_identifier():
    msg = message_from_bytes(
        b"In-Reply-To: not an id\r\nReferences: <a b@c> <>\r\n\r\nx\r\n"
    )
    assert thread_header_fields(msg) == {"in_reply_to": None, "references": []}
