import mindspore
import mindspore.ops as ops
from mindspore import Tensor, nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import GeneratorDataset
import os
import presets
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from resnetrecall import ResNet50 as resnet50
from ecoc import Rand
import mindspore.numpy as np
import random
import time
from libsvm.svmutil import *
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from ecoc.PreKNN import PreKNN
from mindspore import load_checkpoint, load_param_into_net

CFG = {
    'root_dir': '/dataset/ICCV2021/',
    'seed': 68,  # 719,42,68
    'resize_size': 526,
    'crop_size': 448,
    'epochs': 100,#20+40+80+160+320=620
    'warmup_epochs': 5,
    'train_bs': 96,
    'valid_bs': 96,
    'lr': 0.0002,
    'weight_decay': 2e-5,
    'lr_warmup_decay': 0.01,
    'num_workers': 16,
    'accum_iter': 1,
    'verbose_step': 1,
    'device': 'cuda:0',
    'cutmix_prob': 0.8,
}
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
# handler = logging.FileHandler(f"{CFG['model_arch']}_5flod.log")
handler = logging.FileHandler("./stage3try1.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def get_img(path):
    # im_bgr = cv2.imread(path)
    # im_rgb = im_bgr[:, :, ::-1]
    # return im_rgb
    img = Image.open(path).convert('RGB')
    return img


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class ImageNetDataset:
    def __init__(self, root, part='train', transforms=None):
        self.part = part
        self.transforms = transforms
        self.images = []
        self.labels = []
        self.labelset = []
        if part == 'train':
            # mycsv = pd.read_csv('./webbird_train.csv')
            mycsv = pd.read_csv('/disk/8T/xuyy/Tempclassifier/baseline_test/cleanwebbird_train.csv')
        else:
            mycsv = pd.read_csv('/disk/8T/xuyy/Tempclassifier/baseline_test/webbird_val.csv')
        for i in range(len(mycsv['image_id'])):
            self.images.append(mycsv['image_id'][i])
            self.labels.append(int(mycsv['label'][i]))

    def __len__(self):
        return len(self.labels)
        # return 5120

    def __getitem__(self, index):
        # print(self.labelset[index])
        try:
            image = get_img(self.images[index])
        except:
            print(self.images[index])
        if self.transforms is not None:
            try:
                image = self.transforms(image)
            except:
                print(self.images[index])
            # image = self.transforms(image=image)['image']
        return image, self.labels[index], self.images[index]


        
seed_everything(CFG['seed'])

model = resnet50(pretrained=True, n_classes=200)
# model = timm.create_model('resnet50',pretrained=True, num_classes=200)


# model.load_state_dict(torch.load('1219topklosstrain.pth'),strict=False)
load_param_into_net(model, load_checkpoint('stage2.pth')) #根据recall 损失训练出来的权重

train_dataset = ImageNetDataset(CFG['root_dir'], 'train', presets.ClassificationPresetEval(
            crop_size=CFG['crop_size'], resize_size=CFG['resize_size']
        ))
train_dataset = GeneratorDataset(source=train_dataset)

train_loader = train_dataset.create_tuple_iterator(num_epoch=6)
print('down!')

val_dataset = ImageNetDataset(CFG['root_dir'], 'train', presets.ClassificationPresetEval(
            crop_size=CFG['crop_size'], resize_size=CFG['resize_size']
        ))
val_dataset = GeneratorDataset(source=val_dataset)
val_loader = val_dataset.create_tuple_iterator(num_epoch=6)

tr_alldata = None
test_alldata = None
init = None
testlabelpath = []
testpartial = None
trainpartial = None
real_trainlabel = None
real_testlabel = None
for epoch in range(1):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels, img_patha) in pbar:
        # print(image_labels.shape)
        imgs = imgs
        image_labels = image_labels

        image_preds, featx = model(imgs)
        asoft = ops.softmax(image_preds, dim=1)
        bsoftv, bsoft = ops.top_k(asoft, 5)
        bsz = imgs.size(0)
        # print(bsz)
        for i in range(bsz):
            entrop = ops.zeros((200))
            if image_labels[i] in bsoft[i]:
                init = True
            else:
                init = False
                bsoft[i][4] = image_labels[i]
            aasoft = asoft[i]
            entropt = ops.log2(aasoft)
            entrop -= aasoft * entropt
            entroptt = entrop.sum()
            templabel_onehot = np.zeros((200,1))
            reallabel_onehot = np.zeros((200,1))

            # if entroptt < 5: #不混乱的在这里
            if tr_alldata is None:#数据特征
                tr_alldata = featx[i].cpu().numpy()
            else:
                tr_alldata = np.vstack((tr_alldata, featx[i].cpu().numpy()))

            if image_labels[i] in bsoft[i][0]:#确定是一个标签还是partial标签
                templabel_onehot[bsoft[i][:1].cpu().tolist(),0] = 1
            else:
                templabel_onehot[bsoft[i].cpu().tolist(),0] = 1

            if trainpartial is None:
                trainpartial = templabel_onehot
            else:
                # trainpartial = np.vstack((trainpartial,templabel_onehot))
                trainpartial = np.concatenate((trainpartial, templabel_onehot),axis=1) #200class*样本数
            reallabel_onehot[image_labels[i].cpu().numpy(),0] = 1
            if real_trainlabel is None:#按照partial label产生的真实标签
                real_trainlabel = reallabel_onehot
            else:
                real_trainlabel = np.concatenate((real_trainlabel, reallabel_onehot),axis=1)

            # else:#混乱的全放在test集里面
    pbarv = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels, img_patha) in pbarv:
        imgs = imgs.float()
        image_labels = image_labels#.long()

        image_preds, featx = model(imgs)
        asoft = ops.softmax(image_preds, dim=1)
        bsoftv, bsoft = ops.top_k(asoft, 5)
        bsz = imgs.size(0)
        # print(bsz)
        for i in range(bsz):
            entrop = ops.zeros((200))
            if image_labels[i] in bsoft[i]:
                init = True
            else:
                init = False
                bsoft[i][4] = image_labels[i]
            aasoft = asoft[i]
            entropt = ops.log2(aasoft)
            entrop -= aasoft * entropt
            entroptt = entrop.sum()
            templabel_onehot = np.zeros((200,1))
            reallabel_onehot = np.zeros((200,1))

            if test_alldata is None:
                test_alldata = featx[i].cpu().numpy()
            else:
                test_alldata = np.vstack((test_alldata, featx[i].cpu().numpy())) #一行一行的增加的OK的
            testlabelpath.append(img_patha[i])
            if image_labels[i] in bsoft[i][0]:#确定是一个标签还是partial标签
                templabel_onehot[image_labels[i].cpu().numpy(),0] = 1
            else:
                templabel_onehot[bsoft[i].cpu().tolist(),0] = 1
            if testpartial is None:
                testpartial = templabel_onehot
            else:
                testpartial = np.concatenate((testpartial, templabel_onehot),axis=1) #200class*样本数
            reallabel_onehot[image_labels[i].cpu().numpy(),0] = 1
            if real_testlabel is None:
                real_testlabel = reallabel_onehot
            else:
                real_testlabel = np.concatenate((real_testlabel, reallabel_onehot),axis=1)

                # print(tr_alldata.shape)
print('===================train=======================')
filename = open('path.txt', 'a')
for value in testlabelpath:
    filename.write(str(value) + '\n')
filename.close()
print('tr_alldata', tr_alldata.shape)
np.save('tr_alldata', tr_alldata)
print('trainpartial', trainpartial.shape)
np.save('trainpartial', trainpartial)
print('real_trainlabel', real_trainlabel.shape)
np.save('real_trainlabel', real_trainlabel)
print('===================test=======================')
print('test_alldata', test_alldata.shape)
np.save('test_alldata', test_alldata)
print('testpartial', testpartial.shape)
np.save('testpartial',testpartial)
print('real_testlabel', real_testlabel.shape)
np.save('real_testlabel',real_testlabel)
start = time.time()
accuracies = None
ite = 10
i = 0
skf = StratifiedKFold(n_splits=4)
tr_alldata = preprocessing.MinMaxScaler().fit_transform(tr_alldata)#对数据归一化
tr_alldata = preprocessing.StandardScaler().fit_transform(tr_alldata) #对数据标准化
tr_data = tr_alldata
tr_labels = trainpartial
tr_labels = tr_labels.astype(np.int32)

true_labels = real_trainlabel
true_labels = true_labels.astype(np.int32)

test_alldata = preprocessing.MinMaxScaler().fit_transform(test_alldata)#对数据归一化
test_alldata = preprocessing.StandardScaler().fit_transform(test_alldata) #对数据标准化


split_ts_data = test_alldata
testpartial = testpartial
testpartial = testpartial.astype(np.int32)

test_labels = real_testlabel
split_ts_labels = test_labels.astype(np.int32)

X = tr_data
cntwrite = 0
y=np.zeros(true_labels.shape[1])#4998
for index in range(true_labels.shape[1]):
    y[index]=np.argmax(true_labels[:,index])
for tr_idx, ts_idx in skf.split(X, y):#img y都分成了1:9
    # tr_data=preprocessing.scale().fit_transform(tr_data)
    # tr_idx, ts_idx, tv_idx = Tools.tr_ts_split_idx(tr_data)
    split_tr_data, split_tv_data = tr_data[tr_idx], tr_data[ts_idx]
    split_tr_labels = y[tr_idx]#split img class  split_tr_labels 训练集的真实类  测试集0/1真实标签
    tmp_true_labels=true_labels[:,tr_idx]#true_target 训练集真实集的标签
    split_tr_labelsa=tr_labels[:,tr_idx] #partial_target 13, 4498训练集的partiallabel标签
    split_tv_labels = tr_labels[:,ts_idx]
    '''
    训练集:split_tr_data,真实值:
    验证集:split_tv_data,
    测试集:split_ts_data,
    
    '''


    pre_knn = PreKNN(split_tr_labelsa, split_tv_data, split_tv_labels)
    #split_tr_labelsa:训练集的partial label, split_tv_data:验证集的数据, split_tv_labels:验证集的partial label
    pre_knn.fit(split_tr_data, split_tr_labelsa)
    pl_ecoc = Rand.RandPLECOC(libsvm, svm_param='-t 2 -c 1')
    result = pl_ecoc.fit_predict(
                split_tr_data, split_tr_labelsa, split_ts_data, split_ts_labels, split_tv_data, split_tv_labels, pre_knn, cntwrite)
    result = np.array(result).T
    print('result', result)
    accuracies = result if accuracies is None else np.vstack(
                (accuracies, result))
    cntwrite = cntwrite + 1
    del pl_ecoc
    break
# file_name = item+"_mean"
# accuracies=np.array(accuracies)
# for index in range(accuracies.shape[0]):
#      print(str(index+1)+": "+str(accuracies[index, :]))
# for index in range(accuracies.shape[1]):
#     print(str(index+1)+"列mean: "+str(np.mean(accuracies[:, index]))+" max:"+str(np.max(accuracies[:, index]))+" min:"+str(np.min(accuracies[:, index])))
# draw_hist(file_name,accuracies,item+"_mean:"+str(np.mean(accuracies))," ","Accuracy",0,1,0,1)
# print(name + '_ECOC finish')
print('耗时: {:>10.2f} minutes'.format((time.time()-start)/60))


    # result = pl_ecoc.fit_predict(
    #             split_tr_data, split_tr_labels, split_ts_data, split_ts_labels, split_tv_data, split_tv_labels, pre_knn)

        
