import torch
import numpy as np
from torchvision import datasets, transforms
from collections import OrderedDict
from shared_model import get_mobilevit_64bit

# --- CONFIGURATION ---
# UPDATE THIS NAME if your file is named something else!
NPZ_FILE = "global_model_round_1.npz" 
DATA_PATH = "/mnt/c/Users/vishn/Desktop/Malware_Project/DetectionDataset/DetectionDataset/splittedDataset/test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f" Loading 64-bit Federated Model from {NPZ_FILE}...")

# 1. Load the architecture
model = get_mobilevit_64bit().to(DEVICE)

# 2. Extract the weights from the .npz file
data = np.load(NPZ_FILE)
parameters = [data[f] for f in data.files]

# 3. Inject the weights into the model
params_dict = zip(model.state_dict().keys(), parameters)
state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
model.load_state_dict(state_dict, strict=True)
model.eval() # Set to testing mode

print("Loading Test Dataset...")
# 4. Load the Test Images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
test_dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

# 5. Run the Audit
print(f"Starting Evaluation on {len(test_dataset)} unseen malware/benign images...")
correct = 0
total = 0

with torch.no_grad(): # Disable training mechanisms to save memory
    for images, labels in test_loader:
        # Convert images to 64-bit to match the model
        images, labels = images.to(DEVICE).to(torch.float64), labels.to(DEVICE)
        
        # Predict
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 6. Calculate Final Accuracy
accuracy = 100 * correct / total

print(f"FINAL FEDERATED ACCURACY: {accuracy:.2f}%")
