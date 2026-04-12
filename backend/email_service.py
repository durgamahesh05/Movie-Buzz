"""
Email helpers for MovieBuzz notifications.
"""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from typing import Iterable

from config import env, env_int

log = logging.getLogger("email_service")

DEFAULT_HOSTS = ("smtp.zoho.in", "smtp.zoho.com")
SERVICE_HOSTS = {
    "gmail": ("smtp.gmail.com",),
    "google": ("smtp.gmail.com",),
    "zoho": DEFAULT_HOSTS,
    "outlook": ("smtp.office365.com",),
    "hotmail": ("smtp.office365.com",),
    "live": ("smtp.office365.com",),
    "microsoft": ("smtp.office365.com",),
    "yahoo": ("smtp.mail.yahoo.com",),
}
SSL_PORT = env_int("SMTP_SSL_PORT", "MOVIEBUZZ_SMTP_SSL_PORT", default=465)
TLS_PORT = env_int("SMTP_PORT", "MOVIEBUZZ_SMTP_TLS_PORT", default=587)
SMTP_TIMEOUT = env_int("SMTP_TIMEOUT", "MOVIEBUZZ_SMTP_TIMEOUT", default=15)

SMTP_USERNAME = env(
    "SMTP_USER",
    "MOVIEBUZZ_SMTP_EMAIL",
    "EMAIL_USERNAME",
    default="",
)


def _normalize_secret(value: str) -> str:
    cleaned = str(value or "").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1]
    cleaned = cleaned.replace("\\r", "\r").replace("\\n", "\n")
    return cleaned.rstrip("\r\n")


SMTP_PASSWORD = _normalize_secret(
    env(
        "SMTP_PASSWORD",
        "MOVIEBUZZ_SMTP_PASSWORD",
        "EMAIL_PASSWORD",
        default="",
    )
)
SENDER_EMAIL = _normalize_secret(
    env(
        "SMTP_FROM",
        "MOVIEBUZZ_SMTP_EMAIL",
        "EMAIL_FROM",
        default=SMTP_USERNAME,
    )
)
SUPPORT_EMAIL = _normalize_secret(
    env("SUPPORT_EMAIL", "MOVIEBUZZ_SUPPORT_EMAIL", default=SENDER_EMAIL)
)


def has_email_configuration() -> bool:
    return bool(SMTP_USERNAME.strip() and SMTP_PASSWORD.strip())


def _smtp_hosts() -> list[str]:
    hosts: list[str] = []
    configured = env("SMTP_HOST", "MOVIEBUZZ_SMTP_HOST", "EMAIL_HOST", default="").strip()
    if configured:
        hosts.append(configured)

    service = env(
        "EMAIL_SERVICE",
        "SMTP_SERVICE",
        "MOVIEBUZZ_SMTP_SERVICE",
        default="",
    ).strip().lower()
    if not configured and service:
        mapped_hosts = SERVICE_HOSTS.get(service)
        if mapped_hosts:
            hosts.extend(mapped_hosts)
        elif "." in service:
            hosts.append(service)

    if not hosts:
        hosts.extend(DEFAULT_HOSTS)

    deduped_hosts: list[str] = []
    seen: set[str] = set()
    for host in hosts:
        normalized = host.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_hosts.append(normalized)
    return deduped_hosts


def _send_with_ssl(hosts: Iterable[str], message: EmailMessage) -> bool:
    for host in hosts:
        try:
            with smtplib.SMTP_SSL(host, SSL_PORT, timeout=SMTP_TIMEOUT) as server:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(message)
            log.info("Email sent via %s:%s to %s", host, SSL_PORT, message["To"])
            return True
        except Exception as exc:
            log.warning("SMTP SSL send failed via %s:%s: %s", host, SSL_PORT, exc)
    return False


def _send_with_starttls(hosts: Iterable[str], message: EmailMessage) -> bool:
    for host in hosts:
        try:
            with smtplib.SMTP(host, TLS_PORT, timeout=SMTP_TIMEOUT) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(message)
            log.info("Email sent via %s:%s to %s", host, TLS_PORT, message["To"])
            return True
        except Exception as exc:
            log.warning("SMTP STARTTLS send failed via %s:%s: %s", host, TLS_PORT, exc)
    return False


def send_email(receiver_email: str, subject: str, body: str) -> bool:
    if not receiver_email.strip():
        return False
    if not SMTP_USERNAME.strip() or not SMTP_PASSWORD.strip():
        log.error("SMTP credentials are missing")
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = f"MovieBuzz <{SENDER_EMAIL or SMTP_USERNAME}>"
    message["To"] = receiver_email
    message.set_content(body)

    hosts = _smtp_hosts()
    if _send_with_ssl(hosts, message):
        return True
    return _send_with_starttls(hosts, message)


def send_verification_otp_email(receiver_email: str, otp_code: str, name: str = "") -> bool:
    greeting = name.strip() or "there"
    body = (
        f"Hello {greeting},\n\n"
        "Your MovieBuzz account verification OTP is:\n\n"
        f"{otp_code}\n\n"
        "This OTP is valid for the next 5 minutes.\n\n"
        "Please do not share this OTP with anyone.\n\n"
        "If you did not request this, you can safely ignore this email.\n\n"
        "— MovieBuzz Team"
    )
    return send_email(receiver_email, "MovieBuzz Account Verification OTP", body)


def send_account_created_email(receiver_email: str) -> bool:
    body = (
        "Hello,\n\n"
        "Welcome to MovieBuzz! 🎉\n\n"
        "Your account has been successfully created. You can now explore movies, "
        "manage your wishlist, and enjoy personalized recommendations.\n\n"
        "Account Details:\n\n"
        f"* Email: {receiver_email}\n\n"
        "If you did not create this account, please contact our support team immediately.\n\n"
        "Happy streaming! 🍿\n"
        "— MovieBuzz Team"
    )
    return send_email(receiver_email, "Welcome to MovieBuzz", body)


def send_password_reset_otp_email(receiver_email: str, otp_code: str) -> bool:
    body = (
        "Hello,\n\n"
        "Your MovieBuzz password reset OTP is:\n\n"
        f"{otp_code}\n\n"
        "This OTP is valid for the next 10 minutes.\n\n"
        "Please do not share this OTP with anyone.\n\n"
        "If you did not request this, you can safely ignore this email.\n\n"
        "— MovieBuzz Team"
    )
    return send_email(receiver_email, "MovieBuzz Password Reset OTP", body)


def send_account_deletion_otp_email(receiver_email: str, otp_code: str) -> bool:
    body = (
        "Hello,\n\n"
        "Your MovieBuzz account deletion OTP is:\n\n"
        f"{otp_code}\n\n"
        "This OTP is valid for the next 10 minutes.\n\n"
        "Please do not share this OTP with anyone.\n\n"
        f"If you did not request this, please contact {SUPPORT_EMAIL}.\n\n"
        "— MovieBuzz Team"
    )
    return send_email(receiver_email, "MovieBuzz Account Deletion OTP", body)


def send_account_deleted_email(receiver_email: str) -> bool:
    body = (
        "Hello,\n\n"
        "We're sorry to see you go.\n\n"
        "Your MovieBuzz account has been successfully deleted. All your data, including "
        "preferences and wishlist, has been permanently removed from our system.\n\n"
        "If this action was not performed by you, please contact us immediately.\n\n"
        "We hope to see you again in the future.\n\n"
        "Best regards,\n"
        "— MovieBuzz Team"
    )
    return send_email(receiver_email, "Your MovieBuzz Account Has Been Deleted", body)
