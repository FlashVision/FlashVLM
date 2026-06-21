"""Visual Question Answering example with FlashVLM."""

from pathlib import Path

from flashvlm import FlashVLM
from flashvlm.tasks import VQATask


def main():
    """Demonstrate VQA capabilities with FlashVLM."""
    print("=" * 60)
    print("FlashVLM - Visual Question Answering Example")
    print("=" * 60)

    model = FlashVLM.from_pretrained("llava-v1.5-7b")
    print(f"Model loaded: {model.config.model_name}")
    print(f"Parameters: {model.num_parameters():,}")

    image_path = "assets/sample.jpg"

    print("\n--- Direct VQA ---")
    answer = model.ask("What objects are visible in this image?", image=image_path)
    print(f"Q: What objects are visible in this image?")
    print(f"A: {answer}")

    answer = model.ask("What is the main color in the image?", image=image_path)
    print(f"\nQ: What is the main color in the image?")
    print(f"A: {answer}")

    print("\n--- Task-based VQA ---")
    vqa = VQATask(model, prompt_style="detailed", max_new_tokens=256)

    questions = [
        "How many people are in this image?",
        "Is this image taken indoors or outdoors?",
        "What time of day does this appear to be?",
    ]

    for question in questions:
        answer = vqa.answer(image_path, question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")

    print("\n--- Multiple Choice VQA ---")
    answer = vqa.answer(
        image_path,
        "What is the weather like?",
        options=["Sunny", "Cloudy", "Rainy", "Snowy"],
    )
    print(f"Q: What is the weather like?")
    print(f"A: {answer}")

    print("\n--- Evaluation ---")
    predictions = ["2", "outdoors", "morning"]
    ground_truths = ["2", "outdoors", "afternoon"]
    metrics = vqa.evaluate(predictions, ground_truths)
    print(f"VQA Accuracy: {metrics['accuracy']:.2%}")


if __name__ == "__main__":
    main()
