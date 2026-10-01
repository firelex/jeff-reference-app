"""The demo's fictional company: Larkspur Outdoor, an online outdoor-gear shop. Its support teams, the tools its
support agent can call, and the policy excerpts replies are written from. Invented for this demo (not training data).

Each decision question is worded exactly like the questions in the matching adapter's training data (state fields,
instructions and option descriptions), so Jeff's adapters see the kind of question they were trained on."""

COMPANY = ("Larkspur Outdoor, an online shop in the United Kingdom selling tents, sleeping bags, walking boots, "
           "outdoor clothing and camping stoves.")
APPLICATION = ("A customer-support assistant for Larkspur Outdoor, an online outdoor-gear shop, that reads incoming "
               "customer emails and chats and drafts replies.")
AGENT = ("A customer-support assistant for Larkspur Outdoor, an online outdoor-gear shop, that helps customers with "
         "orders, deliveries, returns, refunds, accounts, invoices and warranty claims.")

# Triage teams (option keys k1..k8 plus "other", as in the triage data).
TEAMS = {
    "other": "Not for any of these teams",
    "k1": "Orders and Delivery: Order status, tracking, late or missing parcels, delivery address changes and cancellations before dispatch.",
    "k2": "Returns and Refunds: Returns of unwanted items, exchanges for another size, and refund status and timing.",
    "k3": "Product Advice: Questions about sizes, materials, waterproof ratings, stock and which product suits a trip.",
    "k4": "Accounts and Login: Password resets, login problems, account details, newsletter settings and account deletion.",
    "k5": "Warranty and Repairs: Faulty products within the two-year warranty, repairs and replacement parts.",
    "k6": "Payments and Invoices: Payment failures, double charges, accepted payment methods and VAT invoices.",
    "k7": "Product Safety: Products that caused injury, fire, gas leaks or other danger, recalls and incident reports.",
    "k8": "Complaints: Formal complaints about service, staff or couriers, and requests for compensation.",
}

# Tool options (answer_directly, ask_user, t1..), worded like the tools data: "name(arguments): description".
TOOLS = {
    "answer_directly": "No tool is needed: answer the user directly from the conversation",
    "ask_user": "A tool is needed but required information is missing: ask the user first",
    "t1": "lookup_order(order_id: string): Get an order's status, items, delivery estimate and tracking link",
    "t2": "start_return(order_id: string, items: string[], reason: string): Open a return and email the customer a prepaid returns label",
    "t3": "refund_status(order_id: string): Check whether a refund has been issued for a returned order and when it will arrive",
    "t4": "update_shipping_address(order_id: string, new_address: string): Change the delivery address of an order that has not shipped yet",
    "t5": "cancel_order(order_id: string, reason?: string): Cancel an order that has not shipped yet",
    "t6": "send_password_reset(email: string): Email a password reset link to the account's email address",
    "t7": "send_invoice(order_id: string, vat_number?: string): Email a VAT invoice for an order",
    "t8": "check_stock(product: string, size?: string, colour?: string): Check whether a product is in stock",
    "t9": "open_warranty_claim(order_id: string, product: string, fault: string): Open a warranty claim for a faulty product",
    "t10": "close_account(email: string): Permanently delete a customer account and its personal data",
    "t11": "escalate_to_human(summary: string, team: string, priority: number): Hand the conversation to a human agent with a summary",
}

# Policy excerpts the reply is written from (a tiny knowledge base; retrieval is plain word overlap, the same in both modes).
POLICIES = [
    ("Delivery times", "Standard delivery within the UK takes 3 to 5 working days and is free on orders over £50. Express "
     "delivery takes 1 working day if ordered before 2 pm. Every order gets a tracking link by email when it is dispatched."),
    ("Changing or cancelling an order", "An order can be changed or cancelled free of charge until it is dispatched, "
     "usually within 24 hours of ordering. After dispatch it cannot be changed; the customer can return it instead."),
    ("Returns", "Unused items can be returned within 30 days of delivery for a full refund. Returns are free: we email a "
     "prepaid returns label. Worn footwear can only be returned if it is faulty."),
    ("Refunds", "Refunds are issued to the original payment method within 3 working days of the return reaching our "
     "warehouse. Banks can take a further 3 to 5 working days to show the money."),
    ("Warranty", "All tents, stoves and sleeping bags have a two-year warranty against manufacturing faults. We repair or "
     "replace faulty items; the customer needs the order number and a short description or photo of the fault."),
    ("Product safety", "If a product has caused injury, fire, a gas leak or other danger, stop using it straight away. "
     "Our product safety team contacts the customer within 4 hours, arranges collection of the product and investigates."),
    ("Accounts and passwords", "Customers can reset a password from the sign-in page or ask us to send a reset link to the "
     "email address on the account. For security, we never change the email address on an account over chat."),
    ("Account deletion", "Customers can ask us to delete their account and personal data. We confirm by email and "
     "complete the deletion within 30 days, except records we must keep for tax purposes."),
    ("Payments and invoices", "We accept Visa, Mastercard, American Express, PayPal, Apple Pay and Klarna. A VAT invoice "
     "for any order can be emailed on request; business customers can add their VAT number."),
    ("Compensation and complaints", "Complaints are answered by a named member of the complaints team within 2 working "
     "days. For late deliveries we refund the delivery charge; other compensation is decided case by case."),
]


def retrieve(text: str, count: int = 3) -> list[tuple[str, str]]:
    """The policy excerpts sharing the most words (of four letters or more) with the text."""
    words = {w.strip(".,!?;:'\"()").lower() for w in text.split()}
    words = {w for w in words if len(w) >= 4}

    def overlap(policy: tuple[str, str]) -> int:
        body = {w.strip(".,!?;:'\"()").lower() for w in (policy[0] + " " + policy[1]).split()}
        return len(words & body)

    return sorted(POLICIES, key=overlap, reverse=True)[:count]


def sources_text(policies: list[tuple[str, str]]) -> str:
    """Sources in the ground data's format: "[1] Title\\ntext", blank line between."""
    return "\n\n".join(f"[{i}] {title}\n{body}" for i, (title, body) in enumerate(policies, 1))
