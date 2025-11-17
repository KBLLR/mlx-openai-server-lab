"""MCP (Model Context Protocol) schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job status enumeration."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Job priority enumeration."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class JobCreate(BaseModel):
    """Request to create a new job."""

    model_id: str = Field(..., description="Model identifier")
    operation: str = Field(..., description="Operation: chat, embed, image_gen, etc.")
    input: Dict[str, Any] = Field(..., description="Operation-specific input")
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")
    priority: JobPriority = Field(default=JobPriority.NORMAL, description="Job priority")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class Job(BaseModel):
    """Job representation."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    model_id: str = Field(..., description="Model identifier")
    operation: str = Field(..., description="Operation type")
    priority: JobPriority = Field(..., description="Job priority")
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    failed_at: Optional[datetime] = Field(None, description="Failure timestamp")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Job metadata")
    position_in_queue: Optional[int] = Field(None, description="Position in queue (if queued)")
    estimated_completion_ms: Optional[int] = Field(None, description="Estimated completion time")
    progress: Optional[float] = Field(None, description="Progress 0.0-1.0 (if processing)")


class JobListResponse(BaseModel):
    """Response for list jobs endpoint."""

    jobs: List[Job] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs")
    limit: int = Field(..., description="Limit parameter")
    offset: int = Field(..., description="Offset parameter")


class WebhookEvent(str, Enum):
    """Webhook event types."""

    JOB_QUEUED = "job.queued"
    JOB_PROCESSING = "job.processing"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"


class WebhookCreate(BaseModel):
    """Request to create a webhook."""

    url: str = Field(..., description="Webhook URL")
    events: List[WebhookEvent] = Field(..., description="Events to subscribe to")
    secret: Optional[str] = Field(None, description="Secret for HMAC signature verification")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class Webhook(BaseModel):
    """Webhook representation."""

    webhook_id: str = Field(..., description="Unique webhook identifier")
    url: str = Field(..., description="Webhook URL")
    events: List[WebhookEvent] = Field(..., description="Subscribed events")
    status: str = Field(..., description="Webhook status: active, disabled")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_triggered_at: Optional[datetime] = Field(None, description="Last trigger timestamp")
    success_count: int = Field(default=0, description="Successful delivery count")
    failure_count: int = Field(default=0, description="Failed delivery count")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Webhook metadata")


class WebhookPayload(BaseModel):
    """Webhook event payload."""

    event: WebhookEvent = Field(..., description="Event type")
    timestamp: datetime = Field(..., description="Event timestamp")
    job: Job = Field(..., description="Job data")
