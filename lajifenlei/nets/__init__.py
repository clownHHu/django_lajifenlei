from .alexnet import alexnet, alexnet_trained
from .mobilenet import mobilenet_v2
from .resnet50 import resnet50
from .vgg16 import vgg16
from .densenet import densenet_121
from .googlenet import googlenet

get_model_from_name = {
    "mobilenet": mobilenet_v2,
    "resnet50": resnet50,
    "vgg16": vgg16,
    "densenet_121": densenet_121,
    'googlenet': googlenet,
    'alexnet': alexnet,
    'alexnet_trained': alexnet_trained,
}
