"""VLM Benchmarking example with FlashVLM."""

from flashvlm import FlashVLM, Benchmark
from flashvlm.analytics.metrics import compute_bleu, compute_cider, vqa_accuracy


def main():
    """Demonstrate VLM benchmarking and evaluation."""
    print("=" * 60)
    print("FlashVLM - Benchmarking Example")
    print("=" * 60)

    model = FlashVLM.from_pretrained("llava-v1.5-7b")
    print(f"Model: {model.config.model_name}")
    print(f"Architecture: {model.config.architecture}")
    print(f"Parameters: {model.num_parameters():,}")

    print("\n--- Performance Benchmark ---")
    bench = Benchmark(model, dataset="vqav2", split="val")
    results = bench.run(batch_size=8)

    print("\nResults:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")

    print("\n--- Metric Demonstrations ---")

    print("\nVQA Accuracy:")
    predictions = ["yes", "cat", "2", "blue", "outdoors"]
    ground_truths = [
        ["yes", "yes", "yes"],
        ["cat", "cat", "kitten"],
        ["2", "two", "2"],
        ["blue", "dark blue", "navy"],
        ["outdoors", "outside", "outdoor"],
    ]
    acc = vqa_accuracy(predictions, ground_truths)
    print(f"  Accuracy: {acc:.4f}")

    print("\nBLEU Score:")
    pred_captions = [
        "a cat sitting on a mat",
        "a dog playing in the park",
    ]
    ref_captions = [
        ["a cat is sitting on the mat", "the cat sits on a mat"],
        ["a dog plays in the park", "a dog is playing in a green park"],
    ]
    bleu = compute_bleu(pred_captions, ref_captions)
    print(f"  BLEU-4: {bleu:.4f}")

    print("\nCIDEr Score:")
    cider = compute_cider(pred_captions, ref_captions)
    print(f"  CIDEr: {cider:.4f}")

    print("\n--- Benchmark Summary ---")
    print(bench.summary())


if __name__ == "__main__":
    main()
