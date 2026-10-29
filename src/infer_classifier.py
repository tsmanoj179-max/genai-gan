import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import classifier_transforms
from src.models.classifier import MiniCNNClassifier
from src.utils.device import get_device
from src.utils.paths import project_path


@torch.no_grad()
def predict_image(model, image_path: Path, image_size: int, device):
    transform = classifier_transforms(image_size=image_size, train=False)
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    logit = model(tensor)
    prob_fake = torch.sigmoid(logit).item()
    return {"real": 1.0 - prob_fake, "fake": prob_fake}


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify one image as real or fake.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=project_path("models", "classifier", "classifier_best.pt"))
    parser.add_argument("--image-size", type=int, default=128, choices=[64, 128, 256])
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}")
        return 0

    if not args.checkpoint.exists():
        print(f"Classifier checkpoint not found: {args.checkpoint}")
        print("Train the classifier first with: python -m src.train_classifier")
        return 0

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", args.image_size))

    model = MiniCNNClassifier().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    probs = predict_image(model, args.image, image_size=image_size, device=device)
    predicted = "fake" if probs["fake"] >= 0.5 else "real"
    print(f"Prediction: {predicted}")
    print(f"Real probability: {probs['real']:.4f}")
    print(f"Fake probability: {probs['fake']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
