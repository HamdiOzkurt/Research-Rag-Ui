"""
HITL (Human-in-the-Loop) Approval Flow
Backend pause/resume mechanism for agent approval requests
"""

import asyncio
import logging
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HITLApprovalManager:
    """
    Manages HITL approval flow:
    1. Agent requests approval (pauses execution)
    2. Frontend shows approval modal
    3. User approves/rejects
    4. Backend resumes execution
    """
    
    def __init__(self):
        # Store pending approval requests: {approval_id: ApprovalRequest}
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}
        # Store approval responses: {approval_id: {"approved": bool, "feedback": str}}
        self._approval_responses: Dict[str, Dict[str, Any]] = {}
        
        logger.info("[HITL] ApprovalManager initialized")
    
    async def request_approval(
        self,
        action: str,
        context: Dict[str, Any],
        timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Request human approval and wait for response.
        
        Args:
            action: Action description (e.g., "Run code", "Search web")
            context: Additional context (code snippet, URL, etc.)
            timeout_seconds: Max wait time (default: 5 minutes)
            
        Returns:
            {"approved": bool, "feedback": Optional[str]}
            
        Usage:
            approval = await hitl.request_approval(
                action="Run Python code",
                context={"code": "import os; os.listdir()"}
            )
            if approval["approved"]:
                # Execute action
            else:
                # Skip or handle rejection
        """
        approval_id = str(uuid.uuid4())
        
        # Store pending request
        self._pending_approvals[approval_id] = {
            "action": action,
            "context": context,
            "created_at": datetime.now(),
            "status": "pending"
        }
        
        logger.info(f"[HITL] Approval requested: {approval_id[:8]}... - {action}")
        
        # Wait for user response (with timeout)
        start_time = datetime.now()
        timeout = timedelta(seconds=timeout_seconds)
        
        while (datetime.now() - start_time) < timeout:
            # Check if response received
            if approval_id in self._approval_responses:
                response = self._approval_responses.pop(approval_id)
                self._pending_approvals.pop(approval_id, None)
                
                logger.info(f"[HITL] Approval {approval_id[:8]}... - {'✅ Approved' if response['approved'] else '❌ Rejected'}")
                return response
            
            # Wait 0.5s before checking again
            await asyncio.sleep(0.5)
        
        # Timeout - auto-reject
        logger.warning(f"[HITL] Approval timeout: {approval_id[:8]}...")
        self._pending_approvals.pop(approval_id, None)
        
        return {
            "approved": False,
            "feedback": "Approval request timed out"
        }
    
    def submit_approval(
        self,
        approval_id: str,
        approved: bool,
        feedback: Optional[str] = None
    ):
        """
        Submit user's approval decision.
        
        Args:
            approval_id: Approval request ID
            approved: User decision
            feedback: Optional user feedback/reason
        """
        if approval_id not in self._pending_approvals:
            logger.warning(f"[HITL] Unknown approval ID: {approval_id}")
            return False
        
        self._approval_responses[approval_id] = {
            "approved": approved,
            "feedback": feedback or ""
        }
        
        logger.info(f"[HITL] Approval submitted: {approval_id[:8]}... - {'✅' if approved else '❌'}")
        return True
    
    def get_pending_approvals(self) -> Dict[str, Dict[str, Any]]:
        """Get all pending approval requests."""
        return self._pending_approvals.copy()
    
    def cancel_approval(self, approval_id: str):
        """Cancel a pending approval request."""
        if approval_id in self._pending_approvals:
            self._pending_approvals.pop(approval_id, None)
            self._approval_responses[approval_id] = {
                "approved": False,
                "feedback": "Cancelled by user"
            }
            logger.info(f"[HITL] Approval cancelled: {approval_id[:8]}...")


# Global singleton
_hitl_manager: Optional[HITLApprovalManager] = None


def get_hitl_manager() -> HITLApprovalManager:
    """Get or create HITL manager singleton."""
    global _hitl_manager
    if _hitl_manager is None:
        _hitl_manager = HITLApprovalManager()
    return _hitl_manager


# =============================================================================
# SSE EVENT HELPERS
# =============================================================================

def create_approval_event(
    approval_id: str,
    action: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create SSE event for frontend approval modal.
    
    Returns:
        {"status": "needs_approval", "approval_id": "...", "action": "...", "context": {...}}
    """
    return {
        "status": "needs_approval",
        "approval_id": approval_id,
        "action": action,
        "context": context,
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# EXAMPLE USAGE IN AGENTS
# =============================================================================
"""
# In agent code:
from src.memory.hitl_approval import get_hitl_manager, create_approval_event

async def run_agent_with_hitl(query: str):
    hitl = get_hitl_manager()
    
    # 1. Agent wants to run code
    code = "import os; os.system('rm -rf /')"  # Dangerous!
    
    # 2. Request approval
    approval_event = create_approval_event(
        approval_id=str(uuid.uuid4()),
        action="Run Python code",
        context={"code": code, "risk": "high"}
    )
    
    # 3. Yield SSE event to frontend
    yield approval_event
    
    # 4. Wait for approval
    approval = await hitl.request_approval(
        action="Run Python code",
        context={"code": code}
    )
    
    # 5. Execute based on approval
    if approval["approved"]:
        yield {"status": "running", "message": "Executing code..."}
        result = exec(code)
        yield {"status": "complete", "result": result}
    else:
        yield {"status": "rejected", "reason": approval["feedback"]}
"""


# =============================================================================
# FASTAPI ENDPOINT
# =============================================================================
"""
# Add to simple_copilot_backend.py:

from src.memory.hitl_approval import get_hitl_manager
from pydantic import BaseModel

class ApprovalSubmission(BaseModel):
    approval_id: str
    approved: bool
    feedback: Optional[str] = None

@app.post("/api/approval/submit")
async def submit_approval(submission: ApprovalSubmission):
    '''Submit user's approval decision'''
    hitl = get_hitl_manager()
    success = hitl.submit_approval(
        approval_id=submission.approval_id,
        approved=submission.approved,
        feedback=submission.feedback
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Approval request not found")
    
    return {"success": True, "message": "Approval submitted"}

@app.get("/api/approval/pending")
async def get_pending_approvals():
    '''Get all pending approval requests'''
    hitl = get_hitl_manager()
    return hitl.get_pending_approvals()
"""
