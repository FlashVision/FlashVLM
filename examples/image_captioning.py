"""Image Captioning example with FlashVLM."""

from flashvlm import FlashVLM
from flashvlm.tasks import CaptioningTask


def main():
    """Demonstrate image captioning with multiple styles."""
    print("=" * 60)
    print("FlashVLM - Image Captioning Example")
    print("=" * 60)

    model = FlashVLM.from_pretrained("llava-v1.5-7b")
    captioner = CaptioningTask(model, max_new_tokens=200)

    image_path = "assets/sample.jpg"

    print("\n--- Brief Caption ---")
    caption = captioner.caption(image_path, style="brief")
    print(f"Caption: {caption}")

    print("\n--- Detailed Caption ---")
    caption = captioner.caption(image_path, style="detailed")
    print(f"Caption: {caption}")

    print("\n--- Creative Caption ---")
    caption = captioner.caption(image_path, style="creative")
    print(f"Caption: {caption}")

    print("\n--- Technical Caption ---")
    caption = captioner.caption(image_path, style="technical")
    print(f"Caption: {caption}")

    print("\n--- Accessibility Caption ---")
    caption = captioner.caption(image_path, style="accessibility")
    print(f"Caption: {caption}")

    print("\n--- Contextual Caption ---")
    caption = captioner.caption_with_context(
        image_path, context="This is a photograph from a nature documentary"
    )
    print(f"Caption: {caption}")

    print("\n--- Direct model captioning ---")
    caption = model.caption(image_path, max_new_tokens=100)
    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
