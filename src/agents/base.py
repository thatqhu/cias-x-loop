from abc import ABC
import logging
from ..llm.client import LLMClient
from ..core.world_model_base import WorldModelBase

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, name: str, llm_client: LLMClient, world_model: WorldModelBase):
        self.name = name
        self.llm_client = llm_client
        self.world_model = world_model
        logger.debug(f"Agent {name} initialized")
