import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=15):
        super(ConvNet, self).__init__()
        
        self.layer11 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01)
        )
        
        self.layer21 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01)
        )
        
        self.layer31 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01)
        )
        
        self.layer41 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01)
        )
        
        self.layer51 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01)
        )

        self.fc0 = nn.Linear(9216, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, X):
        x = self.layer11(X)
        print("After layer11:", x.shape)
        x = self.layer21(x)
        print("After layer21:", x.shape)
        x = self.layer31(x)
        print("After layer31:", x.shape)
        x = self.layer41(x)
        print("After layer41:", x.shape)
        x = self.layer51(x)
        print("After layer51:", x.shape)
        
        x = x.reshape(x.size(0), -1)
        print("Flattened size:", x.shape)
        x = self.fc0(x)
        x = F.relu(x)
        x = self.fc1(x)   
        x = F.relu(x)
        x = self.fc2(x)          
        return x

device = 'cpu'  
model = ConvNet(num_classes=15)
model.load_state_dict(torch.load('fine_tuned_model_epoch_1000.pth', map_location=device))
model.eval()

dummy_input = torch.randn(1, 1, 192, 192).to(device)

onnx_model_path = 'model.onnx'
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_model_path, 
    opset_version=11, 
    input_names=['input'], 
    output_names=['output'], 
    dynamic_axes={
        'input': {2: 'height', 3: 'width'},  
        'output': {0: 'batch_size'}   
    }
)
print(f"Model exported to ONNX format at '{onnx_model_path}'.")
