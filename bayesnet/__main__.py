"""Run the browser app or query a model directly from the terminal."""

import argparse
import json
import math
import os
from pathlib import Path

from . import DecisionAnalysis, Inference, Enumeration, ModelError, Network
from .examples import load_example
from .server import serve

TRUE_WORDS = ("1", "true", "yes", "on")


def parse_assignments(items):
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ModelError(f"Expected NODE=STATE, received {item!r}.")
        name, state = item.split("=", 1)
        if name in result:
            raise ModelError(f"Duplicate assignment for {name}.")
        result[name] = state
    return result


def read_model(path):
    """Load a model file, explaining unreadable files and invalid JSON in plain words."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ModelError(f"Model file not found: {path}") from None
    except IsADirectoryError:
        raise ModelError(f"{path} is a folder; give the path of a model JSON file.") from None
    except UnicodeDecodeError:
        raise ModelError(f"{path} is not UTF-8 text.") from None
    except OSError as exc:
        raise ModelError(f"Cannot read {path}: {exc.strerror or exc}.") from None
    try:
        return json.loads(text)
    except RecursionError:
        raise ModelError(f"{path} is nested too deeply.") from None
    except json.JSONDecodeError as exc:
        raise ModelError(f"{path} is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno}).") from None


def server_options(args, environ=None):
    """Combine command-line options with environment variables for hosting services.

    PORT is the port most hosts assign. BAYESLAB_PUBLIC=1 turns on public mode,
    BAYESLAB_HOST overrides the address, and BAYESLAB_TIME_LIMIT sets the
    seconds per calculation (0 means no limit). Command-line options win.
    """
    environ = os.environ if environ is None else environ
    public = bool(getattr(args, "public", False)) or environ.get("BAYESLAB_PUBLIC", "").strip().lower() in TRUE_WORDS
    host = getattr(args, "host", None) or environ.get("BAYESLAB_HOST") or ("0.0.0.0" if public else "127.0.0.1")
    port = getattr(args, "port", None)
    if port is None:
        raw = environ.get("PORT", "").strip()
        try:
            port = int(raw) if raw else 8000
        except ValueError:
            raise ModelError(f"PORT must be a whole number, not {raw!r}.") from None
    if not 0 <= port <= 65535:
        raise ModelError(f"The port must be between 0 and 65535, not {port}.")
    limit = getattr(args, "time_limit", None)
    if limit is None and environ.get("BAYESLAB_TIME_LIMIT", "").strip():
        raw = environ["BAYESLAB_TIME_LIMIT"].strip()
        try:
            limit = float(raw)
        except ValueError:
            raise ModelError(f"BAYESLAB_TIME_LIMIT must be a number of seconds, not {raw!r}.") from None
    if limit is not None and not math.isfinite(limit):
        raise ModelError("The time limit must be a finite number of seconds (0 means no limit).")
    return {"host": host, "port": port, "public": public, "time_limit": limit}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bayeslab", description="Bayesian network probability and utility determination")
    commands = parser.add_subparsers(dest="command")
    web = commands.add_parser("serve", help="Start the browser interface (default)")
    web.add_argument("--host", help="Address to listen on (default 127.0.0.1, or 0.0.0.0 with --public)")
    web.add_argument("--port", type=int, help="Port to listen on (default: $PORT or 8000)")
    web.add_argument("--public", action="store_true",
                     help="Hosting mode: listen on all interfaces, shorter time limits, fewer calculations at once")
    web.add_argument("--time-limit", type=float, metavar="SECONDS",
                     help="Stop any calculation after this many seconds (default 60, or 10 with --public; 0 = none)")
    for name in ("query", "utility", "validate"):
        command = commands.add_parser(name, help={"query": "Calculate an exact probability", "utility": "Compare utilities and find an optimal policy", "validate": "Validate and summarize a network"}[name])
        source = command.add_mutually_exclusive_group()
        source.add_argument("--model", type=Path, help="Path to a network JSON file")
        source.add_argument("--example", choices=("umbrella", "alarm", "sprinkler"), default="umbrella")
        if name != "validate":
            command.add_argument("--evidence", nargs="*", default=[], metavar="NODE=STATE")
            command.add_argument("--method", choices=("enumeration", "elimination"), default="elimination")
        if name == "query":
            command.add_argument("--event", nargs="+", required=True, metavar="NODE=STATE")
            command.add_argument("--action", help="Fixed action (required when actions change chance probabilities)")
    args = parser.parse_args(argv)
    if args.command in (None, "serve"):
        try:
            serve(**server_options(args))
        except ModelError as exc:
            parser.exit(2, f"Error: {exc}\n")
        except OSError as exc:
            parser.exit(1, f"Unable to start server: {exc}\n")
        return
    try:
        data = read_model(args.model) if args.model else load_example(args.example)
        network = Network(data)
        if args.command == "validate":
            result = network.summary()
        elif args.command == "query":
            engine = Enumeration(network) if args.method == "enumeration" else Inference(network)
            result = engine.query(parse_assignments(args.event), parse_assignments(args.evidence), args.action)
        else:
            result = DecisionAnalysis(network, args.method).analyze(parse_assignments(args.evidence))
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ModelError, ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
