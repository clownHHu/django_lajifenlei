import numpy as np
import torch
from PIL import Image

from classification import (Classification, cvtColor,
                            letterbox_image, preprocess_input)
from utils.utils import letterbox_image
from django.http import HttpResponse

class top1_Classification(Classification):
    def detect_image(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------#
        #   对图片进行不失真的resize
        # ---------------------------------------------------#
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        # ---------------------------------------------------------#
        #   归一化+添加上batch_size维度+转置
        # ---------------------------------------------------------#
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo = torch.from_numpy(image_data).type(torch.FloatTensor)

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        arg_pred = np.argmax(preds)
        return arg_pred


def predict(classfication, img_path):
    x = Image.open(img_path)
    pred = classfication.detect_image(x)
    return pred


def detect(request):
    res=['厨余垃圾','可回收垃圾','不可回收垃圾']
    if request.method == "POST":
        File = request.FILES.get("image", None)
        with open("temp.jpg", 'wb+') as f:
            for chunk in File.chunks():
                f.write(chunk)
    classfication = top1_Classification()
    img_path = 'temp.jpg'
    result=predict(classfication, img_path)
    return HttpResponse(res[int(result)])