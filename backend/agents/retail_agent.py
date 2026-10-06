"""
agents/retail_agent.py
-----------------------
LangGraph StateGraph definition for the Dubai Retail Intelligence Agent.

Graph topology:
                     ┌─────────────────────┐
                     │  rag_retrieval_node  │  Node 1
                     └──────────┬──────────┘
                                │
                    ┌───────────▼───────────┐
                    │  route_after_retrieval │  Conditional edge
                    └──┬────────┬─────────┬─┘
                       │        │         │
                  fallback   search/    reorder/analysis
                       │      trend         │
              ┌────────▼───┐    │    ┌──────▼────────────────┐
              │fallback_node│    │    │  sales_analysis_node   │  Node 2
              └────────┬───┘    │    └────────┬───────────────┘
                       │        │             │
                      END       │    ┌────────▼───────────────┐
                                │    │  inventory_check_node   │  Node 3
                                │    └────────┬───────────────┘
                                │             │
                                │    ┌────────▼────────────────┐
                                │    │  seasonality_check_node  │  Node 4
                                │    └────────┬────────────────┘
                                │             │
                             ┌──▼─────────────▼──┐
                             │ recommendation_node │  Node 5
                             └─────────┬─────────┘
                                       │
                                      END

fallback_node (no LLM) writes the user-facing answer when retrieval was not confident:
hard "insufficient data" for factual intents, a partial answer with provenance for narrative ones.

Engineering decision: node names use _node suffix throughout to avoid
clashing with AgentState TypedDict field names. LangGraph 0.2.x raises
ValueError if a node name matches a state key. Fields like 'recommendation',
'inventory_signals', 'sales_signals' exist in AgentState — adding _node
suffix to all node names sidesteps this constraint cleanly.
"""

from __future__ import annotations

import structlog
from langgraph.graph import END, StateGraph

from .nodes.fallback_node import fallback_node
from .nodes.inventory_node import inventory_check_node
from .nodes.rag_node import rag_retrieval_node
from .nodes.recommendation_node import recommendation_node
from .nodes.sales_node import sales_analysis_node
from .nodes.seasonality_node import seasonality_check_node
from .state import AgentState, SeasonalitySignal

logger = structlog.get_logger(__name__)


# ── Conditional edge functions ─────────────────────────────────────────────────

def route_after_retrieval(state: AgentState) -> str:
    """
    Decide which node to run after RAG retrieval.

    - Fallback triggered → fallback_node, then END (deterministic answer; the router key stays
      "end_fallback" so existing routing tests keep working)
    - reorder/analysis intent → sales_analysis_node (needs full signal stack)
    - search/trend intent → recommendation_node (retrieval is sufficient)
    """
    if state.get("fallback_triggered"):
        logger.info(
            "routing_to_fallback",
            confidence=state.get("retrieval_confidence"),
            margin=state.get("retrieval_margin"),
        )
        return "end_fallback"

    intent = state.get("intent", "search")
    if intent in ("reorder", "analysis"):
        logger.info("routing_to_sales", intent=intent)
        return "sales_analysis_node"

    logger.info("routing_to_recommendation", intent=intent)
    return "recommendation_node"


def route_after_sales(state: AgentState) -> str:
    """
    After sales analysis, route to inventory check or skip to recommendation.
    Skips inventory if sales node hard-errored (rare — node degrades internally).
    """
    if state.get("error") and not state.get("sales_signals"):
        logger.warning("sales_error_routing_to_recommendation")
        return "recommendation_node"
    return "inventory_check_node"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_retail_agent() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Compiled once at module import time, reused across all FastAPI requests.
    LangGraph compiled graphs are safe for concurrent reads — each invocation
    gets its own isolated state copy.
    """
    graph = StateGraph(AgentState)

    # _node suffix on all names avoids clash with AgentState field names
    graph.add_node("rag_retrieval_node",      rag_retrieval_node)
    graph.add_node("fallback_node",           fallback_node)
    graph.add_node("sales_analysis_node",     sales_analysis_node)
    graph.add_node("inventory_check_node",    inventory_check_node)
    graph.add_node("seasonality_check_node",  seasonality_check_node)
    graph.add_node("recommendation_node",     recommendation_node)

    graph.set_entry_point("rag_retrieval_node")

    graph.add_conditional_edges(
        "rag_retrieval_node",
        route_after_retrieval,
        {
            "sales_analysis_node":  "sales_analysis_node",
            "recommendation_node":  "recommendation_node",
            "end_fallback":         "fallback_node",
        },
    )

    graph.add_conditional_edges(
        "sales_analysis_node",
        route_after_sales,
        {
            "inventory_check_node": "inventory_check_node",
            "recommendation_node":  "recommendation_node",
        },
    )

    graph.add_edge("fallback_node",          END)
    graph.add_edge("inventory_check_node",   "seasonality_check_node")
    graph.add_edge("seasonality_check_node", "recommendation_node")
    graph.add_edge("recommendation_node",    END)

    compiled = graph.compile()
    logger.info("retail_agent_compiled")
    return compiled


# Module-level singleton — compiled once, reused across all requests
retail_agent = build_retail_agent()


# ── Public entry point ─────────────────────────────────────────────────────────

async def run_agent(
    query:          str,
    query_language: str,
    intent:         str,
) -> AgentState:
    """
    Invoke the retail agent and return the final state.

    Args:
        query:          User query string
        query_language: "en" | "ar" | "mixed"
        intent:         "reorder" | "analysis" | "search" | "trend"

    Returns:
        Final AgentState with all applicable node outputs populated
    """
    initial_state = AgentState(
        query=query,
        query_language=query_language,
        intent=intent,
        retrieved_chunks=[],
        retrieval_confidence=0.0,
        retrieval_margin=0.0,
        sales_signals=[],
        inventory_signals=[],
        seasonality_signal=SeasonalitySignal(
            upcoming_event=None,
            days_until_event=None,
            expected_demand_multiplier=1.0,
            alert_level="none",
        ),
        reasoning_trace=[f"Agent started | intent={intent} | lang={query_language}"],
        recommendation=None,
        recommendation_confidence=0.0,
        fallback_triggered=False,
        error=None,
    )

    logger.info("agent_invoked", query=query[:60], intent=intent, language=query_language)

    result: AgentState = await retail_agent.ainvoke(initial_state)

    logger.info(
        "agent_complete",
        fallback=result.get("fallback_triggered"),
        confidence=result.get("recommendation_confidence"),
        trace_steps=len(result.get("reasoning_trace", [])),
    )

    return result