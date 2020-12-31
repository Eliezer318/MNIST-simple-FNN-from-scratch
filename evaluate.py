import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import train as train_process


def evaluate():
    # Load Data
    batch_size = 256
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Run the model
    new_model = train_process.Neural_Network()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    new_model.params = torch.load('model_weights.pkl', map_location=device)
    acc = train_process.check_accuracy(test_loader, new_model)
    return acc


print(evaluate())
