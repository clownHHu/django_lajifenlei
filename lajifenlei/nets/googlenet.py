import torch.nn as nn
import torchvision.models as models
import torch


def googlenet(pretrained=False, progress=True, num_classes=1000):
    googlenet = models.googlenet(pretrained=True)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['googlenet'], model_dir='./model_data',
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)

    if num_classes != 1000:
        googlenet.fc = nn.Linear(1024, num_classes)
    return googlenet


def googlenet_trained():
    googlenet = models.googlenet(pretrained=True)
    googlenet.fc = nn.Linear(1024, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    googlenet.load_state_dict(torch.load('logs/googlenet/ep100-loss0.032-val_loss0.041.pth', map_location=device))
    googlenet = googlenet.eval().cuda()
    return googlenet
