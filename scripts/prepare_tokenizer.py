import argparse
import os

from src.data.tokenizer import CharTokenizer
from src.utils import load_object


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="JSONL file path (repeatable)")
    parser.add_argument("--output", required=True, help="Output tokenizer JSON path")
    parser.add_argument("--fields", default="product,template", help="Comma-separated fields to use")
    parser.add_argument(
        "--tokenizer_class", default=None, help="Optional tokenizer class path (module.Class or module:Class)"
    )
    args = parser.parse_args()

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    tokenizer_cls = CharTokenizer
    if args.tokenizer_class:
        tokenizer_cls = load_object(args.tokenizer_class)
    tokenizer = tokenizer_cls.build_from_jsonl_files(args.input, fields)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tokenizer.save(args.output)
    print(f"Saved tokenizer with vocab size {len(tokenizer.vocab)} to {args.output}")


if __name__ == "__main__":
    main()
