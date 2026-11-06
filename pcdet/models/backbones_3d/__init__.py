"""Explicit registration of the two public compression variants."""
from importlib import import_module

_VARIANTS = {
    'VirConvL_TFFC': 'spconv_backbone_single_branch_without_model_split',
    'VirConvL_AFFC': 'spconv_backbone_three_branches_without_model_split',
}


def __getattr__(name):
    if name in _VARIANTS:
        return import_module('.' + _VARIANTS[name], __name__).VirConvL8x
    raise AttributeError(name)


class _ResearchRegistry(dict):
    def __missing__(self, key):
        if key not in _VARIANTS:
            raise KeyError(key)
        value = __getattr__(key)
        self[key] = value
        return value


__all__ = _ResearchRegistry()
