"""Visual Grounding example with FlashVLM."""

from flashvlm import FlashVLM
from flashvlm.tasks import GroundingTask
from flashvlm.utils.visualize import draw_bbox


def main():
    """Demonstrate visual grounding: locating objects from text descriptions."""
    print("=" * 60)
    print("FlashVLM - Visual Grounding Example")
    print("=" * 60)

    model = FlashVLM.from_pretrained("llava-v1.5-7b")
    grounding = GroundingTask(model, prompt_style="default")

    image_path = "assets/sample.jpg"

    expressions = [
        "the red car on the left",
        "the person wearing a hat",
        "the tall building in the background",
    ]

    print("\n--- Grounding Results ---")
    boxes = []
    labels = []

    for expr in expressions:
        bbox = grounding.ground(image_path, expr)
        print(f"\nExpression: '{expr}'")
        if bbox:
            print(f"  Bbox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")
            boxes.append(bbox)
            labels.append(expr)
        else:
            print("  Could not locate object.")

    if boxes:
        print("\n--- Visualization ---")
        result_image = draw_bbox(image_path, boxes, labels=labels)
        output_path = "outputs/grounding_result.jpg"
        result_image.save(output_path)
        print(f"Visualization saved to: {output_path}")

    print("\n--- Evaluation ---")
    predictions = [[0.1, 0.2, 0.4, 0.6], [0.5, 0.3, 0.8, 0.9]]
    ground_truths = [[0.1, 0.2, 0.4, 0.6], [0.4, 0.2, 0.7, 0.8]]
    metrics = grounding.evaluate(predictions, ground_truths)
    print(f"Accuracy@0.5: {metrics['accuracy@0.5']:.2%}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")


if __name__ == "__main__":
    main()
