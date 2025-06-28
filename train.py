import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--save_dir', type=str, default='.')
parser.add_argument('--arch', type=str, default='vgg16')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--hidden_units', type=int, default=512)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

# Load data
train_dir = f"{args.data_dir}/train"
valid_dir = f"{args.data_dir}/valid"
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor()])
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# Load model
model = getattr(models, args.arch)(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[0].in_features, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(args.hidden_units, len(train_data.classes)),
    nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
model.to(device)

# Train loop
for e in range(args.epochs):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    valid_loss, accuracy = 0, 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss = criterion(logps, labels)
            valid_loss += batch_loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print(f"Epoch {e+1}/{args.epochs}.. "
          f"Train loss: {running_loss/len(train_loader):.3f}.. "
          f"Valid loss: {valid_loss/len(valid_loader):.3f}.. "
          f"Accuracy: {accuracy/len(valid_loader):.3f}")

# Save checkpoint
checkpoint = {'arch': args.arch,
              'state_dict': model.state_dict(),
              'class_to_idx': train_data.class_to_idx}
torch.save(checkpoint, f"{args.save_dir}/checkpoint.pth")
