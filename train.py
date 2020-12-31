import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
batch_size = 256


class Layers:
    @staticmethod
    def linear_forward(x, W, b):
        cache = (x, W)
        return x.mm(W) + b, cache

    @staticmethod
    def linear_backward(dout, cache):
        # dout [N, H]
        x, W = cache
        dx = dout.mm(W.T)  # [N, H] x [H, D] = [N, D]
        dw = x.T.mm(dout)  # [D, N] x [N, H] = [D, H]
        db = dout.sum(axis=0)
        return dx, dw, db

    @staticmethod
    def relu_forward(X):
        cache = (X >= 0,)
        return X.clamp(0), cache

    @staticmethod
    def relu_backward(dout, cache):
        locs = cache[0]
        return dout * locs

    @staticmethod
    def softmax(scores, y):
        N, C = scores.shape
        scores = (scores - scores.max(axis=1, keepdims=True)[0]).exp()
        probabilities = scores / scores.sum(axis=1, keepdim=True)

        # compute loss
        predictions = probabilities.argmax(axis=1)
        loss = -torch.log(probabilities[torch.arange(N), y]).mean()

        dout = probabilities
        dout[torch.arange(N), y] -= 1
        dout /= N
        return loss, dout, predictions


class Neural_Network:
    def __init__(self, input_size=784, output_size=10, hidden_size=100):
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size

        # weights
        W1 = torch.randn(self.inputSize, self.hiddenSize).to(device) * ((self.inputSize + self.hiddenSize) ** -0.5)
        b1 = torch.zeros(self.hiddenSize).to(device)

        W2 = torch.randn(self.hiddenSize, self.hiddenSize).to(device) * ((self.hiddenSize + self.hiddenSize) ** -0.5)
        b2 = torch.zeros(self.hiddenSize).to(device)

        W3 = torch.randn(self.hiddenSize, self.outputSize).to(device) * ((self.hiddenSize + self.outputSize) ** -0.5)
        b3 = torch.zeros(self.outputSize).to(device)
        self.params = {
            'W1': W1, 'b1': b1,
            'W2': W2, 'b2': b2,
            'W3': W3, 'b3': b3,
        }

    def forward(self, X):
        X, cache1 = Layers.linear_forward(X, self.params['W1'], self.params['b1'])
        X, cache_relu1 = Layers.relu_forward(X)
        X, cache2 = Layers.linear_forward(X, self.params['W2'], self.params['b2'])
        X, cache_relu2 = Layers.relu_forward(X)
        X, cache3 = Layers.linear_forward(X, self.params['W3'], self.params['b3'])
        self.caches = [cache1, cache_relu1, cache2, cache_relu2, cache3]
        return X

    def backward(self, dout, lr=.1):
        cache1, cache_relu1, cache2, cache_relu2, cache3 = self.caches

        dout, dW3, db3 = Layers.linear_backward(dout, cache3)
        dout = Layers.relu_backward(dout, cache_relu2)
        dout, dW2, db2 = Layers.linear_backward(dout, cache2)
        dout = Layers.relu_backward(dout, cache_relu1)
        dout, dW1, db1 = Layers.linear_backward(dout, cache1)
        self.params['W1'] -= lr * dW1
        self.params['b1'] -= lr * db1
        self.params['W2'] -= lr * dW2
        self.params['b2'] -= lr * db2
        self.params['W3'] -= lr * dW3
        self.params['b3'] -= lr * db3

    def train(self, X, y, lr=1e-1):
        # forward + backward pass for training
        scores = self.forward(X)
        loss, dout, pred = Layers.softmax(scores, y)
        self.backward(dout, lr)
        return loss, pred

    def pred(self, X):
        scores = self.forward(X)
        y = torch.zeros(X.shape[0]).to(torch.long)
        loss, dout, pred = Layers.softmax(scores, y)
        return pred


def check_accuracy(loader, model):
    num_correct, num_samples = 0, 0
    for x, y in loader:
        x = x.view(-1, 28 * 28).to(device=device)
        y = y.to(device=device, dtype=torch.long)
        preds = model.pred(x)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    return acc


def make_plots(loss_list, train_acc, test_acc):
    plt.plot(loss_list, label='Train Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss convergence')
    plt.legend()
    plt.show()

    plt.figure()

    plt.plot(train_acc, '-o', label='Train Accuracy')
    plt.plot(test_acc, '-o', label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Convergence')

    plt.legend()
    plt.show()


def main():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = dsets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = Neural_Network()
    loss_list, train_acc, test_acc = [], [check_accuracy(train_loader, model)], [check_accuracy(test_loader, model)]
    num_epochs = 8
    lr, lr_decay = 0.5, 0.9
    for epoch in tqdm(range(num_epochs)):
        for i, (images, labels) in enumerate(train_loader):
            x = images.view(-1, 28 * 28).to(device=device)
            y = labels.to(device=device, dtype=torch.long)
            loss, pred = model.train(x, y, lr)
            loss_list.append(loss)
        lr *= lr_decay
        train_acc.append(check_accuracy(train_loader, model))
        test_acc.append(check_accuracy(test_loader, model))
    make_plots(loss_list, train_acc, test_acc)
    # torch.save(model.params, 'model_weights.pkl')


if __name__ == '__main__':
    main()
