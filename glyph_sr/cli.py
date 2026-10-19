"""Lightweight CLI dispatcher; model dependencies load only inside a command."""

import argparse
from importlib import import_module

_COMMANDS = {
    "infer": ("glyph_sr.inference", "main"),
    "train": ("glyph_sr.training", "main"),
    "score": ("glyph_sr.evaluation", "score"),
}
_DATA_COMMANDS = {
    name: "glyph_sr.data_generation." + name for name in ("prompts", "synthesize", "negatives")
}


def main(argv=None):
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="glyph-sr", description="Scene-text super-resolution, training, and evaluation."
    )
    parser.add_argument("command", choices=[*_COMMANDS, "data"])
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command = argv.pop(0)
    if command == "data":
        data_parser = argparse.ArgumentParser(
            prog="glyph-sr data", description="Synthetic training-data preparation."
        )
        data_parser.add_argument("command", choices=list(_DATA_COMMANDS))
        if not argv or argv[0] in {"-h", "--help"}:
            data_parser.print_help()
            return 0
        name = argv.pop(0)
        if name not in _DATA_COMMANDS:
            data_parser.error(f"unknown data command: {name}")
        return import_module(_DATA_COMMANDS[name]).main(argv)
    if command not in _COMMANDS:
        parser.error(f"unknown command: {command}")
    module, entrypoint = _COMMANDS[command]
    return getattr(import_module(module), entrypoint)(argv)
