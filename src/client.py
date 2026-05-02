import torch
import flwr as fl
from collections import OrderedDict
from shared_model import get_mobilevit_64bit
from utils import get_dataloader # We'll build this next

# 1. Device and Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_mobilevit_64bit().to(DEVICE)

# 2. Local Training Loop
def train(model, trainloader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            # Ensure images/labels match the 64-bit precision
            images, labels = images.to(DEVICE).to(torch.float64), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

# 3. Flower Client Wrapper
class MalwareClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trainloader = get_dataloader(partition="train")
        train(model, trainloader, epochs=2)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

if __name__ == "__main__":
    print("Starting 64-bit Federated Client (Host OS)...")
    # REPLACE WITH YOUR KALI VM's IP ADDRESS!
    fl.client.start_numpy_client(server_address="192.168.159.130:8080", client=MalwareClient())