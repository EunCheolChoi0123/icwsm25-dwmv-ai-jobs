import argparse
import json
from waii import PatentRetriever

def main():
    parser = argparse.ArgumentParser(description="Compute AI exposure for job tasks using wAII method.")
    parser.add_argument('--input', nargs='+', required=True, help='List of job tasks or task-weight pairs.')
    parser.add_argument('--weight', type=str, default=None, choices=['softmax', None], help='Weighting strategy (softmax or None)')
    parser.add_argument('--out', type=str, default="waii_output.json", help='Path to save JSON result')

    args = parser.parse_args()

    # Detect weighted inputs if formatted as task:weight
    tasks = []
    try:
        for item in args.input:
            if ":" in item:
                task, w = item.rsplit(":", 1)
                tasks.append((task, float(w)))
            else:
                tasks.append(item)
    except Exception as e:
        print("Invalid input format. Use either plain strings or 'task:weight'.")
        raise e

    retriever = PatentRetriever()
    result = retriever.ai_exposure(tasks, weight=args.weight)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ AI exposure written to {args.out}")