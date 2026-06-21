"""FlashVLM Command-Line Interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import flashvlm


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flashvlm",
        description="FlashVLM - High-Performance Vision-Language Models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # version
    subparsers.add_parser("version", help="Show FlashVLM version")

    # settings
    settings_parser = subparsers.add_parser("settings", help="Show current settings")
    settings_parser.add_argument("--config", type=str, default=None, help="Config file path")

    # check
    check_parser = subparsers.add_parser("check", help="Check system environment")
    check_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # train
    train_parser = subparsers.add_parser("train", help="Train or fine-tune a VLM")
    train_parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    train_parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    train_parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    train_parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")

    # chat
    chat_parser = subparsers.add_parser("chat", help="Interactive multimodal chat")
    chat_parser.add_argument("--model", type=str, default="llava-v1.5-7b", help="Model name/path")
    chat_parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    chat_parser.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    # caption
    caption_parser = subparsers.add_parser("caption", help="Generate image captions")
    caption_parser.add_argument("--image", type=str, required=True, help="Image path or URL")
    caption_parser.add_argument("--model", type=str, default="llava-v1.5-7b", help="Model name")
    caption_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens")
    caption_parser.add_argument("--detail", type=str, default="brief", choices=["brief", "detailed"])

    # vqa
    vqa_parser = subparsers.add_parser("vqa", help="Visual Question Answering")
    vqa_parser.add_argument("--image", type=str, required=True, help="Image path or URL")
    vqa_parser.add_argument("--question", type=str, required=True, help="Question about image")
    vqa_parser.add_argument("--model", type=str, default="llava-v1.5-7b", help="Model name")
    vqa_parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens")

    # export
    export_parser = subparsers.add_parser("export", help="Export model to deployment format")
    export_parser.add_argument("--model", type=str, required=True, help="Model name/path")
    export_parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "torchscript", "safetensors"])
    export_parser.add_argument("--output", type=str, default="exported_model", help="Output path")
    export_parser.add_argument("--quantize", action="store_true", help="Apply quantization")

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run VLM benchmarks")
    bench_parser.add_argument("--model", type=str, required=True, help="Model name/path")
    bench_parser.add_argument("--dataset", type=str, default="vqav2", help="Benchmark dataset")
    bench_parser.add_argument("--split", type=str, default="val", help="Dataset split")
    bench_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    bench_parser.add_argument("--output", type=str, default=None, help="Results output file")

    return parser


def cmd_version(args: argparse.Namespace) -> None:
    print(f"FlashVLM v{flashvlm.__version__}")
    print(f"Python: {sys.version.split()[0]}")
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: Not installed")
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: Not installed")


def cmd_settings(args: argparse.Namespace) -> None:
    from flashvlm.cfg.config import get_config

    if args.config:
        config = get_config(args.config)
    else:
        config = get_config()
    print("Current FlashVLM Settings:")
    print("-" * 40)
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")


def cmd_check(args: argparse.Namespace) -> None:
    print("FlashVLM System Check")
    print("=" * 40)

    checks = []

    try:
        import torch
        checks.append(("PyTorch", True, torch.__version__))
        checks.append(("CUDA", torch.cuda.is_available(), torch.version.cuda or "N/A"))
    except ImportError:
        checks.append(("PyTorch", False, "Not installed"))
        checks.append(("CUDA", False, "N/A"))

    try:
        import transformers
        checks.append(("Transformers", True, transformers.__version__))
    except ImportError:
        checks.append(("Transformers", False, "Not installed"))

    try:
        import PIL
        checks.append(("Pillow", True, PIL.__version__))
    except ImportError:
        checks.append(("Pillow", False, "Not installed"))

    try:
        import safetensors
        checks.append(("Safetensors", True, safetensors.__version__))
    except ImportError:
        checks.append(("Safetensors", False, "Not installed"))

    try:
        import peft
        checks.append(("PEFT (LoRA)", True, peft.__version__))
    except ImportError:
        checks.append(("PEFT (LoRA)", False, "Not installed (optional)"))

    for name, status, version in checks:
        icon = "[OK]" if status else "[--]"
        print(f"  {icon} {name}: {version}")

    all_ok = all(s for _, s, _ in checks[:4])
    print()
    if all_ok:
        print("All core dependencies satisfied.")
    else:
        print("Some dependencies are missing. Run: pip install flashvlm")


def cmd_train(args: argparse.Namespace) -> None:
    from flashvlm.cfg.config import get_config
    from flashvlm.models.vlm import FlashVLM
    from flashvlm.engine.trainer import Trainer

    config = get_config(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    config.training.output_dir = args.output_dir

    print(f"Initializing model from config: {args.config}")
    model = FlashVLM(config)
    trainer = Trainer(model, config)

    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.resume(args.resume)

    print("Starting training...")
    trainer.train()


def cmd_chat(args: argparse.Namespace) -> None:
    from flashvlm.solutions.multimodal_chat import MultimodalChat

    print(f"Loading model: {args.model}")
    chat = MultimodalChat(
        model_name=args.model,
        device=args.device,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print("Model loaded. Type 'quit' to exit, '/image <path>' to add an image.")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "/quit"):
            print("Goodbye!")
            break
        if user_input.startswith("/image "):
            image_path = user_input[7:].strip()
            chat.add_image(image_path)
            print(f"[Image added: {image_path}]")
            continue

        response = chat.send(user_input)
        print(f"\nAssistant: {response}")


def cmd_caption(args: argparse.Namespace) -> None:
    from flashvlm.models.vlm import FlashVLM

    print(f"Loading model: {args.model}")
    model = FlashVLM.from_pretrained(args.model)

    prompt = "Describe this image in detail." if args.detail == "detailed" else "Describe this image briefly."
    caption = model.generate(
        image=args.image,
        prompt=prompt,
        max_new_tokens=args.max_tokens,
    )
    print(f"\nCaption: {caption}")


def cmd_vqa(args: argparse.Namespace) -> None:
    from flashvlm.models.vlm import FlashVLM

    print(f"Loading model: {args.model}")
    model = FlashVLM.from_pretrained(args.model)

    answer = model.ask(args.question, image=args.image, max_new_tokens=args.max_tokens)
    print(f"\nAnswer: {answer}")


def cmd_export(args: argparse.Namespace) -> None:
    from flashvlm.engine.exporter import Exporter
    from flashvlm.models.vlm import FlashVLM

    print(f"Loading model: {args.model}")
    model = FlashVLM.from_pretrained(args.model)

    exporter = Exporter(model)
    output_path = exporter.export(
        format=args.format,
        output_path=args.output,
        quantize=args.quantize,
    )
    print(f"Model exported to: {output_path}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from flashvlm.analytics.benchmark import Benchmark
    from flashvlm.models.vlm import FlashVLM

    print(f"Loading model: {args.model}")
    model = FlashVLM.from_pretrained(args.model)

    bench = Benchmark(model, dataset=args.dataset, split=args.split)
    results = bench.run(batch_size=args.batch_size)

    print(f"\nBenchmark Results ({args.dataset}):")
    print("-" * 40)
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    if args.output:
        import json
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {args.output}")


COMMANDS = {
    "version": cmd_version,
    "settings": cmd_settings,
    "check": cmd_check,
    "train": cmd_train,
    "chat": cmd_chat,
    "caption": cmd_caption,
    "vqa": cmd_vqa,
    "export": cmd_export,
    "benchmark": cmd_benchmark,
}


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
