import torch
from torchvision import transforms

import model_creation
import setup_data, engine, utils,models

def main():
    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    HIDDEN_UNITS = 32
    LEARNING_RATE = 0.001

    train_dir = "data/desert101/train"
    test_dir = "data/desert101/test"

    data_transform = transforms.Compose([
        
        transforms.Resize(size=(64,64)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5483, 0.4638, 0.3865],
                             std=[0.2173, 0.2279, 0.2263])
    ])

    train_dataloader, test_dataloader, class_names= setup_data.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=BATCH_SIZE,
        transform=data_transform
    )

    model = model_creation.DesertClassifier(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    )

    loss_fn= torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS
    )

    utils.save_model(
        model=model,
        target_dir="models",
        model_name="enhanced_desert_classifier.pth"
    )


if __name__ == "__main__":
    main()
