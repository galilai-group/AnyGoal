"""World-model implementations maintained by anygoal."""

__all__ = [
    "DriftWorldWorldModel",
    "DriftWorldWorldModelPolicy",
    "IRASimWorldModel",
    "IRASimWorldModelPolicy",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    if name.startswith("DriftWorld"):
        from . import driftworld

        return getattr(driftworld, name)
    if name.startswith("IRASim"):
        from . import irasim

        return getattr(irasim, name)

    raise AttributeError(name)
