"""Fusion orchestration schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WorkflowStepType(str, Enum):
    """Workflow step types."""

    MODEL_CALL = "model_call"
    VECTOR_SEARCH = "vector_search"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


class WorkflowStep(BaseModel):
    """Single step in a fusion workflow."""

    step_id: str = Field(..., description="Unique step identifier")
    type: WorkflowStepType = Field(..., description="Step type")
    model_id: Optional[str] = Field(None, description="Model ID for model_call steps")
    operation: Optional[str] = Field(None, description="Operation to perform")
    input: Any = Field(..., description="Step input (supports templating)")
    config: Optional[Dict[str, Any]] = Field(None, description="Step-specific configuration")


class FusionWorkflow(BaseModel):
    """Fusion workflow definition."""

    workflow_id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Human-readable workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    steps: List[WorkflowStep] = Field(..., description="Workflow steps")
    required_models: List[str] = Field(default_factory=list, description="Required model types")
    optional_models: List[str] = Field(default_factory=list, description="Optional model types")


class FusionRequest(BaseModel):
    """Request to execute a fusion workflow."""

    workflow: str = Field(..., description="Workflow ID or 'custom'")
    input: Dict[str, Any] = Field(..., description="Workflow input data")
    models: Optional[Dict[str, str]] = Field(None, description="Model overrides")
    config: Optional[Dict[str, Any]] = Field(None, description="Workflow configuration")


class ExecutionStep(BaseModel):
    """Execution trace for a single step."""

    step: str = Field(..., description="Step identifier")
    model: Optional[str] = Field(None, description="Model used")
    status: str = Field(..., description="Step status: completed, failed")
    duration_ms: int = Field(..., description="Step duration in milliseconds")
    error: Optional[str] = Field(None, description="Error message if failed")


class FusionResult(BaseModel):
    """Result of a fusion workflow execution."""

    workflow_id: str = Field(..., description="Workflow identifier")
    status: str = Field(..., description="Workflow status: completed, failed, partial")
    result: Any = Field(..., description="Workflow result")
    execution_trace: List[ExecutionStep] = Field(..., description="Execution trace")
    total_latency_ms: int = Field(..., description="Total execution time")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class ModelCapabilities(BaseModel):
    """Model capabilities for fusion routing."""

    model_id: str = Field(..., description="Model identifier")
    capabilities: List[str] = Field(..., description="List of capabilities")
    context_length: Optional[int] = Field(None, description="Context length")
    vram_usage_gb: Optional[float] = Field(None, description="VRAM usage in GB")
    status: str = Field(..., description="Model status: loaded, unloaded")


class CapabilitiesResponse(BaseModel):
    """Response for capability discovery."""

    capabilities: Dict[str, List[str]] = Field(..., description="Capability to models mapping")
    models: List[ModelCapabilities] = Field(..., description="Model capabilities")
