import torch
from torchvision import datasets, transforms
from train import train_frl19dv2_1, evaluate

# Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=True)

# Prepare data
X, Y = next(iter(train_loader))
X = X.view(X.shape[0], -1).numpy()  # Flatten to (1000, 784)
Y = torch.nn.functional.one_hot(Y, 10).numpy().astype(np.float32)

# Train
model = train_frl19dv2_1(X, Y, epochs=50)
accuracy = evaluate(model, torch.tensor(X, dtype=torch.float32).to(model.weights.device),
                   torch.tensor(Y, dtype=torch.float32).to(model.weights.device))
print(f"Accuracy: {accuracy:.4f}")