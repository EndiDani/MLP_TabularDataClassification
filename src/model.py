import torch
from torch import nn
from torch import optim


class TabularNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim): 
        super(TabularNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512), # Architettura ancora da definire
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x): 
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def training(dataloader, model, loss_fn, optimizer): 
        size = len(dataloader.dataset)
        mdel.train()

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to("cpu"), y.to("cpu") # TODO: cambia questa cosa

            # Calcolo errore di predizione
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropragation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0: 
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:7f} [{current:>5d}/{size:>5d}]")
            
    # Per controllare la performance del modello e controllare che stia imparando
    def test(dataloader, model, loss_fn): 
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader: 
                X, y = X.to("cpu"), y.to("cpu")
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test error:\nAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")





model = TabularNeuralNetwork(input_dim=10, output_dim=1).to("cpu")
print(model)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "../results/models/")
print("Saved PyTorch Model State to ../results/models/")
model.load_state_dict(torch.load("../results/models/", weights_only=True))
