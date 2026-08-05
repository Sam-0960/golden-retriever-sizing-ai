"""
predict_breed.py
------------------
predict_breed(image_path): inference only. Training lives in
training/train_breed_classifier.py.

If no trained model exists yet at models/breed_classifier.pth, this returns
a clearly-labeled PLACEHOLDER prediction (uniform 50/50, is_placeholder=True)
so the rest of the pipeline can be built/tested before training is done.
Once you train a real model, this automatically switches to using it --
no code changes needed elsewhere.
"""

import os

CLASS_NAMES = ["Golden Retriever", "Labrador Retriever"]  # matches folder order convention
LOW_CONFIDENCE_THRESHOLD = 0.6


def predict_breed(image_path, checkpoint_path="models/breed_classifier.pth"):
    """
    Returns:
        {
          "predicted_breed": str,
          "confidence": float (0-1),
          "all_probs": {breed: prob, ...},
          "is_placeholder": bool,
          "needs_manual_confirmation": bool,
        }
    """
    if not os.path.exists(checkpoint_path):
        return {
            "predicted_breed": CLASS_NAMES[0],
            "confidence": 0.5,
            "all_probs": {c: 0.5 for c in CLASS_NAMES},
            "is_placeholder": True,
            "needs_manual_confirmation": True,
        }

    import torch
    from PIL import Image
    from torchvision import transforms, models
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    eval_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = eval_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = probs.argmax()
    predicted_breed = classes[pred_idx]
    confidence = float(probs[pred_idx])
    all_probs = {cls: float(p) for cls, p in zip(classes, probs)}

    return {
        "predicted_breed": predicted_breed,
        "confidence": confidence,
        "all_probs": all_probs,
        "is_placeholder": False,
        "needs_manual_confirmation": confidence < LOW_CONFIDENCE_THRESHOLD,
    }
