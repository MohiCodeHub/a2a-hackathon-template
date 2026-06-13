"""Rho-Bank customer service agent: policy + env tools + KB search (RAG)."""

import os
from pathlib import Path

from google.adk.agents import LlmAgent

from env_toolset import EnvApiToolset
from rag_tools import kb_search_bm25, kb_search_vector

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
- kb_search_vector(query): semantic search for natural-language questions.

Search before you act; procedures, eligibility rules, internal tool names,
and scenario-specific guidance all live in the knowledge base. If a search
comes up empty, rephrase and try again before telling the customer you can't
find the information.
"""

root_agent = LlmAgent(
    name="cs_agent",
    model=MODEL,
    description=CARD_DESCRIPTION,
    instruction=POLICY_PATH.read_text() + RAG_GUIDANCE,
    tools=[EnvApiToolset(), kb_search_bm25, kb_search_vector],
)
