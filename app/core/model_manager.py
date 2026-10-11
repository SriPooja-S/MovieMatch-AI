import gc
import torch
from typing import Dict, Any, Optional
from app.utils.logger import logger

# Restrict PyTorch CPU thread allocation to prevent system saturation on 8GB RAM
try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
except Exception as e:
    logger.debug(f"Could not set PyTorch thread limits: {e}")


class ModelManager:
    """
    Singleton Resource & Model Lifecycle Manager for 8GB RAM CPU execution.
    Handles dynamic loading, thread scoping, and aggressive memory eviction.
    """

    _instance: Optional["ModelManager"] = None
    _models: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._models = {}
        return cls._instance

    def get_model(self, key: str, loader_fn) -> Any:
        """Get or load model on-demand."""
        if key not in self._models or self._models[key] is None:
            logger.info(f"[ModelManager] Loading model into memory: {key}")
            self._models[key] = loader_fn()
        return self._models[key]

    def unload_model(self, key: str):
        """Unload specific model and force garbage collection."""
        if key in self._models and self._models[key] is not None:
            logger.info(f"[ModelManager] Unloading model from memory: {key}")
            del self._models[key]
            self._models[key] = None
            self.collect_garbage()

    def unload_all(self):
        """Unload all models to free RAM."""
        logger.info("[ModelManager] Evicting all loaded models from memory...")
        self._models.clear()
        self.collect_garbage()

    @staticmethod
    def collect_garbage():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


model_manager = ModelManager()
