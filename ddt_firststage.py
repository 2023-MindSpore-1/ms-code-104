from pca_project import PCAProjectNet
import os
from vgg import *
from PIL import Image
import cv2
import pandas as pd
import shutil
import numpy as np
from PIL import ImageFile
import mindspore
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.dataset import vision, transforms
from mindspore.dataset import GeneratorDataset
import numpy
import os
from PIL import Image
from mindspore.dataset import MindDataset
from mindspore import dtype as mstype
from data import load_data


ImageFile.LOAD_TRUNCATED_IMAGES = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root = '/disk/8T/xuyy/rateddt/'
train_set, _ = load_data(root, 'bird', batch_size=32)
train_loader = train_set.create_tuple_iterator(num_epochs=1)

savecount = []
model = vgg19(pretrained=True).cuda()
maxv = 0
imgs = []
flag = 0
features = []
savelabel = []
saveimage_id = []

for it, (img, indexlabel) in enumerate(train_loader):
    img = img.cuda()

    feature, _ = model(img)
    pca = PCAProjectNet()

    project_map = ops.clip_by_value(pca(feature), clip_value_min=Tensor(0, mindspore.int32), )
    print('============', project_map.shape)
    maxv = project_map.view(project_map.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
    epochi = img.shape[0]

    for i in range(0, epochi):
        if maxv[i] > 0 :
            savelabel.append(indexlabel[i].item())
            saveimage_id.append(i+i*32)
mytrain = pd.DataFrame()
mytrain['label'] = savelabel
mytrain['image_id'] = saveimage_id
mytrain.to_csv('webbirdtrain.csv', index=False) 


