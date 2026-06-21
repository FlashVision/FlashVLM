"""Multimodal Chat example with FlashVLM."""

from flashvlm.solutions import MultimodalChat


def main():
    """Demonstrate interactive multimodal chat."""
    print("=" * 60)
    print("FlashVLM - Multimodal Chat Example")
    print("=" * 60)

    chat = MultimodalChat(
        model_name="llava-v1.5-7b",
        temperature=0.7,
        max_tokens=512,
        system_prompt="You are a helpful visual AI assistant that can analyze images and engage in conversation.",
    )

    print("\n--- Programmatic Chat ---")

    chat.add_image("assets/sample.jpg")
    print("[Image loaded]")

    response = chat.send("What do you see in this image?")
    print(f"Assistant: {response}")

    response = chat.send("Can you describe the colors in more detail?")
    print(f"\nAssistant: {response}")

    response = chat.send("What mood does this image convey?")
    print(f"\nAssistant: {response}")

    print(f"\nConversation turns: {chat.num_turns}")

    chat.reset()
    print("\n--- New Conversation ---")

    response = chat.send("Hello! What can you help me with today?")
    print(f"Assistant: {response}")

    print("\n--- Interactive Mode ---")
    print("To start interactive chat, run: flashvlm chat --model llava-v1.5-7b")


if __name__ == "__main__":
    main()
