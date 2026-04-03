from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import numpy as np
import uuid
import asyncio
from contextlib import asynccontextmanager

from .regret_env import RegretZeroEnv


# Pydantic models for API requests and responses
class ActionRequest(BaseModel):
    """Request model for taking an action in the environment."""
    action: int = Field(..., ge=0, le=7, description="Action to take (0-7)")


class ResetRequest(BaseModel):
    """Request model for resetting the environment."""
    seed: Optional[int] = Field(None, ge=0, description="Random seed for reproducibility")
    max_steps: Optional[int] = Field(None, ge=1, le=1000, description="Maximum steps per episode")
    scenario_type: Optional[str] = Field(None, description="Specific scenario type")
    difficulty_level: Optional[str] = Field(None, description="Difficulty level: easy, medium, hard")
    stakes_level: Optional[str] = Field(None, description="Stakes level: low, medium, high")


class StepResponse(BaseModel):
    """Response model for environment step."""
    observation: List[float] = Field(..., description="Next state observation (32-dim vector)")
    reward: float = Field(..., description="Reward signal (negative regret)")
    terminated: bool = Field(..., description="Whether episode ended naturally")
    truncated: bool = Field(..., description="Whether episode ended due to step limit")
    info: Dict[str, Any] = Field(..., description="Additional episode information")


class ResetResponse(BaseModel):
    """Response model for environment reset."""
    observation: List[float] = Field(..., description="Initial observation (32-dim vector)")
    info: Dict[str, Any] = Field(..., description="Initial episode information")


class EnvInfoResponse(BaseModel):
    """Response model for environment information."""
    action_space: Dict[str, Any] = Field(..., description="Action space specification")
    observation_space: Dict[str, Any] = Field(..., description="Observation space specification")
    max_steps: int = Field(..., description="Maximum steps per episode")
    action_meanings: List[str] = Field(..., description="Human-readable action descriptions")
    env_version: str = Field(..., description="Environment version")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service health status")
    env_initialized: bool = Field(..., description="Whether environment is initialized")
    active_sessions: int = Field(..., description="Number of active environment sessions")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class EpisodeStatsResponse(BaseModel):
    """Response model for episode statistics."""
    current_step: int = Field(..., description="Current step in episode")
    total_regret: float = Field(..., description="Accumulated regret")
    decision_count: int = Field(..., description="Number of decisions made")
    avg_regret: float = Field(..., description="Average regret per decision")
    scenario_progress: float = Field(..., description="Progress through scenario (0-1)")
    scenario_type: str = Field(..., description="Current scenario type")
    termination_reason: Optional[str] = Field(..., description="Reason for episode termination")


# Global environment manager
class EnvironmentManager:
    """Manages multiple environment sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, RegretZeroEnv] = {}
        self.start_time = asyncio.get_event_loop().time()
    
    def create_session(self, session_id: str, **kwargs) -> RegretZeroEnv:
        """Create a new environment session."""
        env = RegretZeroEnv(**kwargs)
        self.sessions[session_id] = env
        return env
    
    def get_session(self, session_id: str) -> Optional[RegretZeroEnv]:
        """Get an existing environment session."""
        return self.sessions.get(session_id)
    
    def remove_session(self, session_id: str) -> bool:
        """Remove an environment session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return asyncio.get_event_loop().time() - self.start_time
    
    def get_active_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.sessions)


# Global environment manager instance
env_manager = EnvironmentManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("RegretZero OpenEnv API starting up...")
    yield
    # Shutdown
    print("RegretZero OpenEnv API shutting down...")
    # Clean up all sessions
    env_manager.sessions.clear()


# Create FastAPI application
app = FastAPI(
    title="RegretZero OpenEnv API",
    description="A Gymnasium-style OpenEnv environment for regret minimization in decision-making",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper functions
def get_or_create_session(session_id: Optional[str] = None, **env_kwargs) -> tuple[str, RegretZeroEnv]:
    """Get existing session or create new one."""
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    env = env_manager.get_session(session_id)
    if env is None:
        env = env_manager.create_session(session_id, **env_kwargs)
    
    return session_id, env


def validate_session(session_id: str) -> RegretZeroEnv:
    """Validate and return environment session."""
    env = env_manager.get_session(session_id)
    if env is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Session {session_id} not found. Call /reset first."
        )
    return env


# API Endpoints

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RegretZero OpenEnv API",
        "version": "1.0.0",
        "description": "A Gymnasium-style OpenEnv environment for regret minimization",
        "endpoints": {
            "health": "/health",
            "env_info": "/env/info",
            "reset": "/env/reset",
            "step": "/env/step",
            "stats": "/env/stats",
            "render": "/env/render"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        env_initialized=len(env_manager.sessions) > 0,
        active_sessions=env_manager.get_active_session_count(),
        uptime_seconds=env_manager.get_uptime()
    )


@app.get("/env/info", response_model=EnvInfoResponse)
async def get_env_info():
    """Get environment specifications and information."""
    # Create a temporary environment to get specs
    temp_env = RegretZeroEnv()
    
    return EnvInfoResponse(
        action_space={
            "type": "Discrete",
            "n": temp_env.action_space.n,
            "description": "8 discrete decision-making actions"
        },
        observation_space={
            "type": "Box",
            "shape": temp_env.observation_space.shape,
            "low": temp_env.observation_space.low.tolist(),
            "high": temp_env.observation_space.high.tolist(),
            "description": "32-dimensional vector: decision_context(16) + emotional_state(8) + historical_context(8)"
        },
        max_steps=temp_env.max_steps,
        action_meanings=temp_env.get_action_meanings(),
        env_version="1.0.0"
    )


@app.post("/env/reset", response_model=ResetResponse)
async def reset_environment(request: ResetRequest, session_id: Optional[str] = None):
    """
    Reset the environment to start a new episode.
    
    Args:
        request: Reset parameters
        session_id: Optional session ID (creates new if None)
    
    Returns:
        Initial observation and episode information
    """
    try:
        # Prepare environment initialization parameters
        env_kwargs = {}
        if request.max_steps is not None:
            env_kwargs["max_steps"] = request.max_steps
        if request.seed is not None:
            env_kwargs["seed"] = request.seed
        
        # Get or create session
        session_id, env = get_or_create_session(session_id, **env_kwargs)
        
        # Reset environment
        observation, info = env.reset(seed=request.seed)
        
        # Add session info
        info["session_id"] = session_id
        info["available_actions"] = env.get_action_meanings()
        
        return ResetResponse(
            observation=observation.tolist(),
            info=info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Environment reset failed: {str(e)}")


@app.post("/env/step", response_model=StepResponse)
async def step_environment(request: ActionRequest, session_id: str):
    """
    Take a step in the environment.
    
    Args:
        request: Action to take
        session_id: Environment session ID
    
    Returns:
        Next observation, reward, termination status, and info
    """
    try:
        # Validate session
        env = validate_session(session_id)
        
        # Take step
        observation, reward, terminated, truncated, info = env.step(request.action)
        
        # Add session info
        info["session_id"] = session_id
        info["action_taken"] = env.ACTIONS[request.action]
        
        # Clean up session if episode ended
        if terminated or truncated:
            env_manager.remove_session(session_id)
            info["session_closed"] = True
        else:
            info["session_closed"] = False
        
        return StepResponse(
            observation=observation.tolist(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Environment step failed: {str(e)}")


@app.get("/env/stats", response_model=EpisodeStatsResponse)
async def get_episode_stats(session_id: str):
    """
    Get current episode statistics.
    
    Args:
        session_id: Environment session ID
    
    Returns:
        Current episode statistics
    """
    try:
        # Validate session
        env = validate_session(session_id)
        
        # Calculate statistics
        avg_regret = env.regret_accumulator / max(1, len(env.decision_history))
        
        return EpisodeStatsResponse(
            current_step=env.current_step,
            total_regret=env.regret_accumulator,
            decision_count=len(env.decision_history),
            avg_regret=avg_regret,
            scenario_progress=env._get_scenario_progress(),
            scenario_type=env.scenario_type,
            termination_reason=env._get_termination_reason() if env._check_termination() else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/env/render")
async def render_environment(session_id: str):
    """
    Render the current environment state.
    
    Args:
        session_id: Environment session ID
    
    Returns:
        Rendered environment information
    """
    try:
        # Validate session
        env = validate_session(session_id)
        
        # Render environment (captures output)
        import io
        import sys
        from contextlib import redirect_stdout
        
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            env.render()
        
        render_output = captured_output.getvalue()
        
        return {
            "session_id": session_id,
            "render_output": render_output,
            "step": env.current_step,
            "scenario_type": env.scenario_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Environment render failed: {str(e)}")


@app.delete("/env/session/{session_id}")
async def close_session(session_id: str):
    """
    Close an environment session.
    
    Args:
        session_id: Environment session ID to close
    
    Returns:
        Session closure confirmation
    """
    try:
        success = env_manager.remove_session(session_id)
        if success:
            return {"message": f"Session {session_id} closed successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to close session: {str(e)}")


@app.get("/env/sessions")
async def list_sessions():
    """
    List all active environment sessions.
    
    Returns:
        List of active session IDs and basic info
    """
    try:
        sessions_info = []
        for session_id, env in env_manager.sessions.items():
            sessions_info.append({
                "session_id": session_id,
                "current_step": env.current_step,
                "scenario_type": env.scenario_type,
                "total_regret": env.regret_accumulator,
                "decision_count": len(env.decision_history)
            })
        
        return {
            "active_sessions": len(env_manager.sessions),
            "uptime_seconds": env_manager.get_uptime(),
            "sessions": sessions_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {"error": "Not found", "detail": str(exc.detail)}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return {"error": "Internal server error", "detail": str(exc.detail)}


# Run server if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "regret_openenv:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
