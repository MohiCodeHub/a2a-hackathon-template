"""Rho-Bank customer service agent: policy + env tools + KB search (RAG)."""

import os
from pathlib import Path

from google.adk.agents import LlmAgent

from calc_tool import calculate
from env_toolset import EnvApiToolset
from rag_tools import kb_search_bm25, kb_search_graph

MODEL = os.environ.get("MODEL", "gemini-3.5-flash")
POLICY_PATH = Path(os.environ.get("KB_POLICY_PATH", "/app/kb/policy.md"))

# Stranger-facing agent-card description: read by other teams' personal agents
# when routing to us. States the verification contract up front so a foreign
# assistant supplies the right details instead of round-tripping or giving up.
CARD_DESCRIPTION = (
    "Rho-Bank customer service agent. Serves bank customers and their "
    "authorized personal assistants over A2A. Identity verification is required "
    "before accessing or modifying any customer data: provide the customer's "
    "full name PLUS any two of {date of birth, email, phone number, address}. "
    "Details relayed by the customer's personal assistant on the customer's "
    "behalf are accepted — no separate third-party authorization is needed. "
    "Once verified, resolves account, card, transaction, dispute, refund, and "
    "policy requests using internal bank tools, and hands user-side actions to "
    "the assistant to perform when an action must be done by the customer."
)

RAG_GUIDANCE = """

## Knowledge Base Access

You do NOT have the knowledge base inlined. Before answering policy questions
or performing scenario-specific procedures, search the knowledge base:
- kb_search_bm25(query): keyword search.
- kb_search_graph(query): semantic search for natural-language questions. Also
  returns documents commonly needed alongside the best match (e.g. the full set
  of comparable products), so prefer it when a request may span related docs.

Search before you act; procedures, eligibility rules, internal tool names,
and scenario-specific guidance all live in the knowledge base. If a search
comes up empty, rephrase and try again before telling the customer you can't
find the information.
"""

# C1 (enumerate worksheet) + C3 (net semantics) + C2 (calculate tool). Targets
# the audit-and-correct precision failures: acting on the wrong record set,
# refunding gross instead of net, and small arithmetic drift.
RECONCILIATION_GUIDANCE = """

## Corrections, refunds, and audits — compute precisely

Many requests ask you to audit a set of records and correct errors (fee
refunds, reward/cash-back corrections, dispute reconciliation). For these,
follow this procedure exactly:

1. Enumerate before acting. Retrieve EVERY relevant record (e.g. each
   transaction on each account) and the governing rule from the knowledge base
   (the account tier's fee schedule, the card's reward rate, applicable
   min/max caps and free allowances). Build an explicit line-by-line
   worksheet — for each record list: its actual value, the correct value per
   the rule, and the signed difference (correct - actual).

2. Correct only TRUE discrepancies. A record is an error only if its actual
   value differs from the rule-derived correct value. Do not act on records
   that are already correct, and never invent records that are not in the data.

3. Apply the NET result. When one corrective action covers several records
   (e.g. a single per-account credit), the amount is the NET of the signed
   differences across those records: overcharges and owed rebates increase it;
   undercharges and fees the customer still owes (including fees that were
   missing and should have been charged) decrease it. Write the EXACT
   recomputed value — not a rounded or approximate one.

4. Never do arithmetic in your head. Use the calculate(expression) tool for
   every fee, rate, percentage, cap, and especially the final net sum. Pass the
   full expression and use the returned result verbatim.
"""

# C8 — discoverable-call discipline. Every call_discoverable_agent_tool writes a
# row to the scored database (agent_discoverable_tools), and the task is graded
# on an exact whole-DB hash. So a single exploratory discoverable read the gold
# procedure didn't make fails the task even when every write is correct.
TOOL_DISCIPLINE_GUIDANCE = """

## Tool-call discipline (this affects scoring)

Every time you CALL an agent discoverable tool (via call_discoverable_agent_tool)
it is permanently recorded in the bank's records. Calling a discoverable tool
that this request does not actually require leaves a stray record that fails the
task — even if everything else you did was correct.

- Call ONLY the discoverable tools the knowledge-base procedure for THIS specific
  request requires. Do not make exploratory or "just to be safe" discoverable
  calls (e.g. pulling all accounts, payment history, or dispute history) when the
  procedure and the conversation already give you what you need.
- Unlocking a tool is free; CALLING it is what gets recorded. Never call a
  discoverable tool you do not need for the action you are completing.
- For routine identity/account lookups, prefer the always-available base tools
  (e.g. get_user_information_by_name, get_credit_card_accounts_by_user) — those
  are not recorded — over discoverable equivalents, unless the procedure names a
  specific discoverable tool.
- Before each discoverable call, ask: does the procedure actually require this to
  complete the request? If not, don't call it.
"""

# C9 — date discipline. Confirmed via DB diff: disputes were failing only because
# issue_noticed_date was back-dated (11/11) instead of the current date (11/14).
# A date that represents "now" must come from get_current_time, not be inferred.
DATE_GUIDANCE = """

## Dates in tool arguments

Use the exact correct date for every date argument — never invent, approximate,
or back-date one.

- A date that represents the present — when the customer is noticing or
  reporting an issue during this conversation, or when you take an action today
  — is the CURRENT date. Get it from get_current_time; do not guess. For
  example, when a customer reviews their statement and notices a problem during
  this call, the "issue noticed" date is today's date, not an earlier date.
- A date that refers to a specific past event (e.g. when a transaction
  occurred) must come exactly from the records or from what the customer states.
"""

root_agent = LlmAgent(
    name="cs_agent",
    model=MODEL,
    description=CARD_DESCRIPTION,
    instruction=POLICY_PATH.read_text()
    + RAG_GUIDANCE
    + RECONCILIATION_GUIDANCE
    + TOOL_DISCIPLINE_GUIDANCE
    + DATE_GUIDANCE,
    tools=[EnvApiToolset(), kb_search_bm25, kb_search_graph, calculate],
)
