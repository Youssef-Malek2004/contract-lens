#!/usr/bin/env python3
"""
agent.py — CLI entry point for the ContractLens Conversation Agent.

Loads Qwen3-4B once, retrieves RAG context, and answers a natural language
question about the specified NDA contract. Conversation history persists in
conversation_history.json between runs for multi-turn sessions.

Usage:
    python agent.py --contract data/test.json --idx 0 \\
                    --retrieval vector \\
                    --prompt "Does this NDA allow sharing with consultants?"

    python agent.py --contract data/test.json --idx 0 \\
                    --retrieval graph \\
                    --prompt "What are the termination obligations?"
"""

import argparse
import sys

from src.conversation_agent import ConversationAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="ContractLens Conversation Agent — NDA question answering with RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python agent.py --contract data/test.json --idx 0 \\\n"
            "                  --retrieval vector \\\n"
            "                  --prompt \"Does this NDA allow sharing with consultants?\"\n\n"
            "  python agent.py --contract data/test.json --idx 5 \\\n"
            "                  --retrieval graph \\\n"
            "                  --prompt \"What are the return-or-destroy obligations?\""
        ),
    )
    parser.add_argument(
        "--contract",
        type=str,
        default="data/test.json",
        metavar="PATH",
        help="Path to ContractNLI JSON file (default: data/test.json)",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        metavar="N",
        help="Zero-based document index within the contract file (default: 0)",
    )
    parser.add_argument(
        "--retrieval",
        type=str,
        choices=["vector", "graph"],
        default="vector",
        metavar="MODE",
        help="RAG retrieval backend: vector | graph (default: vector)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        metavar="QUESTION",
        help="Natural language question about the contract",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("ContractLens — Conversation Agent")
    print("=" * 60)
    print(f"  Contract : {args.contract}  (idx={args.idx})")
    print(f"  Retrieval: {args.retrieval}")
    print(f"  Question : {args.prompt}")
    print("=" * 60)

    try:
        agent = ConversationAgent(retrieval_mode=args.retrieval)
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)

    print()
    try:
        response = agent.run_turn(
            contract_path=args.contract,
            idx=args.idx,
            user_prompt=args.prompt,
        )
    except IndexError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Generation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 60)
    print("Answer:")
    print("=" * 60)
    print(response)


if __name__ == "__main__":
    main()
