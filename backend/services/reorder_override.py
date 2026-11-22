"""
services/reorder_override.py
-----------------------------
The HUMAN OVERRIDE rules (Phase 3). Pure: no database, no I/O. A recommendation is a proposal; a person decides.

  approve   accept the system's quantity as it is                      -> status "approved", final_qty = order_qty
  modify    accept the order but change the quantity (a reason is required) -> status "modified", final_qty = the person's number
  reject    do not order (a reason is required)                         -> status "rejected", final_qty = None

DECISION: the system's recommendation is NEVER edited. The person's decision is stored beside it (final_qty,
decided_by, decision_reason) and appended to an event log. The original quantity, trace and confidence stay exactly
as the system produced them.
  Alternative: overwrite order_qty with the human number (simpler, one column).
  Why not: the project's precision metric is "how often was the system right". That is only measurable if the system's
  number and the human's number both survive. Overwriting destroys the ground truth the metric needs, and an auditor
  could no longer tell what the model said from what a person changed. It is also the thing an interviewer at a retailer
  asks first: "can you show me what it recommended before the buyer edited it?"
  Trade-off: two quantities per row, and every consumer must read final_qty (not order_qty) for what was actually ordered.

DECISION: one decision per recommendation; a decided recommendation is final.
  A change of mind is a new scan/recommendation, not an edit of history. Two people deciding at once is resolved by the
  database (the second update finds the row no longer pending and is refused), not by a lock in application code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

DECISIONS = {"approve": "approved", "reject": "rejected", "modify": "modified"}
MIN_REASON_CHARS = 3
MAX_QTY = 1_000_000


class OverrideError(ValueError):
    """A decision that cannot be applied. `code` is 'conflict' (already decided) or 'invalid' (bad input)."""

    def __init__(self, message: str, code: str = "invalid"):
        super().__init__(message)
        self.code = code


def apply_decision(rec: dict, decision: str, actor: str, reason: Optional[str] = None,
                   qty: Optional[int] = None, now: Optional[datetime] = None) -> tuple[dict, dict]:
    """
    Returns (updated copy of the recommendation, audit event). Raises OverrideError. Never mutates `rec`.
    """
    if decision not in DECISIONS:
        raise OverrideError(f"decision must be one of {sorted(DECISIONS)}")
    actor = (actor or "").strip()
    if not actor:
        raise OverrideError("actor is required: who is making this decision")
    if rec.get("status") != "pending":
        raise OverrideError(f"recommendation {rec.get('rec_id')} is already {rec.get('status')}; a decision is final",
                            code="conflict")
    reason = (reason or "").strip() or None
    if decision in ("reject", "modify") and (reason is None or len(reason) < MIN_REASON_CHARS):
        raise OverrideError(f"a reason is required to {decision} a recommendation")

    original = rec.get("order_qty")
    if decision == "approve":
        final = original
    elif decision == "reject":
        final = None
    else:
        if qty is None or isinstance(qty, bool) or not isinstance(qty, int):
            raise OverrideError("modify needs an integer quantity")
        if qty < 0 or qty > MAX_QTY:
            raise OverrideError(f"quantity must be between 0 and {MAX_QTY}")
        if qty == original:
            raise OverrideError("the quantity is unchanged; use approve instead")
        final = qty

    now = now or datetime.now(timezone.utc)
    stamp = now.isoformat(timespec="seconds")
    updated = {**rec, "status": DECISIONS[decision], "decided_by": actor, "decided_at": stamp,
               "decision_reason": reason, "final_qty": final}
    if decision == "modify" and original:
        updated["modification_delta_pct"] = round(100.0 * (final - original) / original, 1)
    event = {"rec_id": rec["rec_id"], "event_type": DECISIONS[decision], "actor": actor, "reason": reason,
             "qty_before": original, "qty_after": final, "created_at": stamp}
    return updated, event
