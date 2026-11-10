"""Use original public validation rows for an offline or Gemini-assisted copilot demo."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from scripts.stage5_samples import validation_request
from src.chatbot import ask_copilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6: source-labelled LangChain Gemini+Groq automatic-fallback public-data copilot")
    parser.add_argument("--track", choices=("iranian", "ibm"), required=True)
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Use an existing public VALIDATION row, NOT synthetic data")
    parser.add_argument("--question", default="Explain this churn prediction and its limitations")
    parser.add_argument("--live", action="store_true", help="Use only verified Free Tier provider(s); requires project .env API keys")
    parser.add_argument("--chat", action="store_true", help="Interactive CLI chat using the same validation row")
    parser.add_argument("--agent-tools", action="store_true",
                        help="Demonstrate LLM-initiated tool calling (typically 2 calls; economy mode needs 1)")
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error("--sample-index must be nonnegative")
    request = validation_request(args.track, row_index=args.sample_index)
    print(f"Stage 6 demo: {args.track} | public validation sample index {args.sample_index}")
    print("No real Teletalk customer data. No raw feature vector is sent to Gemini or Groq.")
    print("LLM tool mode:", "model-directed (usually 2 calls)" if args.agent_tools else "economy local LangChain tool (usually 1 call)")
    if args.chat:
        print("Chat mode: type 'exit' to stop. Stage 7 verified offers are not enabled.")
        while True:
            try:
                question = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nDone")
                break
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            reply = ask_copilot(question, track=args.track, features=request["features"], live=args.live, agent_tools=args.agent_tools)
            print("\nCopilot> " + reply.message + "\n")
            print("MODE:", reply.mode, "| PROVIDER:", reply.provider or "none", "| MODEL:", reply.model_id or "none")
    else:
        reply = ask_copilot(args.question, track=args.track, features=request["features"], live=args.live, agent_tools=args.agent_tools)
        print("\n" + reply.message)
        print("\nMODE:", reply.mode)
        print("PROVIDER:", reply.provider or "none", "| MODEL:", reply.model_id or "none")


if __name__ == "__main__":
    main()
