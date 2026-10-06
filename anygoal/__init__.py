from .envs.registration import register_envs
from .registry import get_registry
from .version import __version__

register_envs()

__all__ = ["__version__", "get_registry", "register_envs"]
