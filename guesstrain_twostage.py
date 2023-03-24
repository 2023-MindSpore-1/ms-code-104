import mindspore
from mindspore import nn, ops, Tensor, Parameter
import copy
import os
import presets
import transforms
import losses.Recall_at_K as Recall_at_K
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import time
from collections import defaultdict
import pandas as pd
from resnetrecall import ResNet50 as resnet50
from mindspore.dataset import GeneratorDataset
from mindspore import load_checkpoint, load_param_into_net

CFG = {
    'root_dir': '/dataset/ICCV2021/',
    'seed': 68,  # 719,42,68
    'resize_size': 526,
    'crop_size': 448,
    'epochs': 100,#20+40+80+160+320=620
    'warmup_epochs': 5,
    'train_bs': 50,
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
handler = logging.FileHandler("./1209.txt")
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
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class OurSampler:
    def __init__(self, data_source, batch_size,method,n_pos,N):
        self.data_source = data_source
        self.index_dic = data_source.Index
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)
        self.batch_size = (batch_size//2) *2
        self.method = method
        self.classes = len(data_source.Index)   
        self.times = self.data_source.sample_nums//self.batch_size
        self.n_pos = n_pos
        self.N = N

    def __len__(self):
        return self.times*self.batch_size

    def __iter__(self):
        ret = []
        iterindex_dic = copy.deepcopy(self.index_dic)
        for _ in range(self.times):
            cnt = 0
            sample_classes = 200
            slices = random.sample([i for i in range(self.classes)], sample_classes)
            # print('slices', slices)
            break_flag=False
            for i in range(sample_classes):
                category_num=len(iterindex_dic[int(slices[i])])
                candidatea=iterindex_dic[slices[i]][:category_num]
                if category_num>=self.n_pos+1:
                    candidate=random.sample([candidatea[i] for i in range(len(candidatea))],self.n_pos+1)
                    for j in range(len(candidate)):
                        cnt+=1
                        ret.append(candidate[j])
                        for deli in range(category_num):
                            if candidate[j] == iterindex_dic[int(slices[i])][deli]:
                                del iterindex_dic[int(slices[i])][deli]
                                break
                        if cnt==self.batch_size:
                            break_flag=True
                            break
                    if break_flag==True:
                        break
        return iter(ret)


class ImageNetDataset:
    def __init__(self, root, part='train', transforms=None):
        self.part = part
        self.transforms = transforms
        self.images = []
        self.labels = []
        if part == 'train':
            # mycsv = pd.read_csv('./webbird_train.csv')
            mycsv = pd.read_csv('./cleanwebbird_train.csv')
        else:
            mycsv = pd.read_csv('./webbird_val.csv')
        for i in range(len(mycsv['image_id'])):
            self.images.append(mycsv['image_id'][i])
            self.labels.append(int(mycsv['label'][i]))
        Index = defaultdict(list)
        for i in range(len(self.labels)):
            Index[self.labels[i]].append(i)
        self.Index = Index
        self.sample_nums = len(self.labels)
        # print('self.sample_nums', self.sample_nums)

    def __len__(self):
        return len(self.labels)
        # return 5120

    def __getitem__(self, index):
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


class ImageNetDatasetval:
    def __init__(self, root, part='train', transforms=None):
        self.part = part
        self.transforms = transforms
        self.images = []
        self.labels = []
        if part == 'train':
            # mycsv = pd.read_csv('./webbird_train.csv')
            mycsv = pd.read_csv('./cleanwebbird_train.csv')
        else:
            mycsv = pd.read_csv('./webbird_val.csv')
        for i in range(len(mycsv['image_id'])):
            self.images.append(mycsv['image_id'][i])
            self.labels.append(int(mycsv['label'][i]))
        Index = defaultdict(list)
        for i in range(len(self.labels)):
            Index[self.labels[i]].append(i)
        self.Index = Index
        self.sample_nums = len(self.labels)
        # print('self.sample_nums', self.sample_nums)

    def __len__(self):
        return len(self.labels)
        # return 5120

    def __getitem__(self, index):
        try:
            image = get_img(self.images[index])
        except:
            print(self.images[index])
        if self.transforms is not None:
            try:
                image = self.transforms(image)
            except:
                print(self.images[index])
        return image, self.labels[index]


def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, loss_recall, device, scheduler=None, schd_batch_update=False):
    model.train()
    
    t = time.time()
    running_loss = None
    image_preds_all = []
    image_targets_all = []
    img_path = []
    image_idsave = [] 
    img_set=[] 
    class_save=[]

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    if epoch != 50:
        for step, (imgs, image_labels, img_paths) in pbar:

            imgs = imgs.to(device).float()
            image_labels = image_labels.to(device)#.long()
            bzs = image_labels.shape[0]
   
            image_preds, featpre = model(imgs)  # output = model(input)
            # print(image_preds.shape, exam_pred.shape)
            image_preds_all += [ops.argmax(image_preds, 1).detach().cpu().numpy()]
            
            loss_metric, err_pos, sparsity = loss_recall(featpre, image_labels,img_path, image_idsave, img_set, class_save)
            loss = loss_fn(image_preds, image_labels)
            loss1 = loss + 0.5* loss_metric

            image_targets_all += [image_labels.detach().cpu().numpy()]

            
            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01
            
            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):

                optimizer.zero_grad()
                
                if scheduler is not None and schd_batch_update:
                    scheduler.step()
            
            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                
                pbar.set_description(description)
    
        image_preds_all = np.concatenate(image_preds_all)
        image_targets_all = np.concatenate(image_targets_all)

        ans = (image_preds_all == image_targets_all).mean()
        print('Train multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))
        logger.info(' Epoch: ' + str(epoch) + ' Train multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))
        if scheduler is not None and not schd_batch_update:
            scheduler.step() 
    else:
        for step, (imgs, image_labels, img_paths) in pbar:

            imgs = imgs.to(device).float()
            image_labels = image_labels.to(device)#.long()
            bzs = image_labels.shape[0]
   
            image_preds, featpre = model(imgs)  
            image_preds_all += [ops.argmax(image_preds, 1).detach().cpu().numpy()]
            aaa = ops.argmax(image_preds, 1)
            for i in range(0, bzs):
                image_idsave.append(img_paths[i])
                img_set.append(aaa[i].item())
                class_save.append(image_labels[i].item())
                print(image_labels[i].item())
        mytrain = pd.DataFrame()
        print(class_save)
        mytrain['label'] = class_save
        mytrain['image_id'] = image_idsave
        mytrain['img_set'] = img_set
        mytrain.to_csv('guess5.csv', index=False) 


def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()
    
    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device)#.float()
        
        image_preds, featp = model(imgs)  # output = model(input)
        # print(image_preds.shape, exam_pred.shape)
        image_preds_all += [ops.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
        loss = loss_fn(image_preds, image_labels)
        
        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]
        
        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    ans = (image_preds_all == image_targets_all).mean()
    print('validation multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))
    logger.info(' Epoch: ' + str(epoch) + 'validation multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()
    return ans


# from typing import List, Optional, Tuple, Union
def split_normalization_params(model, norm_classes=None):
    # Adapted from https://github.com/facebookresearch/ClassyVision/blob/659d7f78/classy_vision/generic/util.py#L501
    if not norm_classes:
        norm_classes = [nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm]

    for t in norm_classes:
        if not issubclass(t, nn.Cell):
            raise ValueError(f"Class {t} is not a subclass of nn.Module.")

    classes = tuple(norm_classes)

    norm_params = []
    other_params = []
    for module in model.modules():
        if next(module.children(), None):
            other_params.extend(p for p in module.parameters(recurse=False) if p.requires_grad)
        elif isinstance(module, classes):
            norm_params.extend(p for p in module.parameters() if p.requires_grad)
        else:
            other_params.extend(p for p in module.parameters() if p.requires_grad)
    return norm_params, other_params


        
seed_everything(CFG['seed'])

model = resnet50(pretrained=True, n_classes=200)
# model = timm.create_model('resnet50',pretrained=True, num_classes=200)

load_param_into_net(model, load_checkpoint('stage2.pth')) #根据recall 损失训练出来的权重

train_dataset = ImageNetDataset(CFG['root_dir'], 'train', presets.ClassificationPresetEval(
            crop_size=CFG['crop_size'], resize_size=CFG['resize_size']
        ))
train_dataset = GeneratorDataset(source=train_dataset)

train_loader = train_dataset.create_tuple_iterator(num_epoch=6)
print('down!')

val_dataset = ImageNetDatasetval(CFG['root_dir'], 'train', presets.ClassificationPresetEval(
            crop_size=CFG['crop_size'], resize_size=CFG['resize_size']
        ))
val_dataset = GeneratorDataset(source=val_dataset)
val_loader = val_dataset.create_tuple_iterator(num_epoch=6)

param_groups = split_normalization_params(model)
wd_groups = [0.0, CFG['weight_decay']]
parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

optimizer = nn.Adam(parameters, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
# optimizer = torch.optim.AdamW(parameters, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=2,
#                                                                         eta_min=CFG['min_lr'], last_epoch=-1)
# scheduler = SchedulerCosineDecayWarmup(optimizer, CFG['lr'], 5, CFG['epochs'])
step_size_train = train_dataset.get_dataset_size()
lr = nn.cosine_decay_lr(min_lr=1e-5, max_lr=1e-2, total_step=step_size_train*6,
                            step_per_epoch=step_size_train, decay_epoch=5)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2,
#                                                                         eta_min=1e-6, last_epoch=-1)

loss_tr = nn.CrossEntropyLoss(label_smoothing=0.1)  # MyCrossEntropyLoss().to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn1 = nn.CrossEntropyLoss()
loss_recall = Recall_at_K.Recall_at_K(n_input=55, k=3, tau=0, margin=0.05,n_pos=3)


best_answer = 0.0

for epoch in range(6):
    # with open('./labelcheck.txt', 'a') as f:
    # print(optimizer.param_groups[0]['lr'])

    train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, loss_recall, schd_batch_update=False)

    answer = 0.0

    # if (epoch<100 and epoch%10==9) or (epoch<130 and epoch>=100 and epoch%5==4) or epoch>=130:
    if epoch%1==0:
        answer = valid_one_epoch(epoch, model, loss_fn1, val_loader, scheduler=None, schd_loss_update=False)
    # if answer > best_answer:
    #     torch.save(model.state_dict(), 'stage2.pth'.format(epoch))
    if answer > best_answer:
        best_answer = answer
del model, optimizer, train_loader, val_loader
print(best_answer)
logger.info('BEST-TEST-ACC: ' + str(best_answer))
