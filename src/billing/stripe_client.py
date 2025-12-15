import stripe
from typing import Optional, Dict, Any
from src.config import settings


def _require(key: Optional[str], name: str) -> str:
    if not key:
        raise RuntimeError(f"Missing {name}. Set it in multi_agent_search/.env")
    return key


def get_stripe() -> stripe:
    stripe.api_key = _require(settings.stripe_secret_key, "STRIPE_SECRET_KEY")
    return stripe


def verify_webhook(payload: bytes, sig_header: Optional[str]) -> stripe.Event:
    secret = _require(settings.stripe_webhook_secret, "STRIPE_WEBHOOK_SECRET")
    if not sig_header:
        raise RuntimeError("Missing Stripe-Signature header")
    return stripe.Webhook.construct_event(payload=payload, sig_header=sig_header, secret=secret)


def create_customer(clerk_user_id: str, email: Optional[str] = None) -> stripe.Customer:
    st = get_stripe()
    metadata: Dict[str, str] = {"clerk_user_id": clerk_user_id}
    params: Dict[str, Any] = {"metadata": metadata}
    if email:
        params["email"] = email
    return st.Customer.create(**params)


def create_checkout_session(
    stripe_customer_id: str,
    price_id: str,
    success_url: str,
    cancel_url: str,
) -> stripe.checkout.Session:
    st = get_stripe()
    return st.checkout.Session.create(
        mode="subscription",
        customer=stripe_customer_id,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        allow_promotion_codes=True,
    )


def create_billing_portal_session(
    stripe_customer_id: str,
    return_url: str,
) -> stripe.billing_portal.Session:
    st = get_stripe()
    return st.billing_portal.Session.create(customer=stripe_customer_id, return_url=return_url)


