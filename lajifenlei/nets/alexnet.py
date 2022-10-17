import torch.nn as nn
import torchvision.models as models
import torch
from torch.hub import load_state_dict_from_url
model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
}


def alexnet(pretrained=False, progress=True, num_classes=2):
    alexnet = models.alexnet(pretrained=False)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], model_dir='./model_data',
                                              progress=progress)
        alexnet.load_state_dict(state_dict)

    if num_classes != 1000:
        alexnet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )
    return alexnet

def alexnet_trained(pretrained=True,num_classes=2):
    alexnet = models.alexnet(pretrained=True)
    alexnet.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 2),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alexnet.load_state_dict(torch.load('logs//alexnet//ep100-loss0.055-val_loss0.067.pth', map_location=device))
    alexnet = alexnet.eval()
    return alexnet
