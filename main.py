import mindspore
import mindspore as ms
from mindspore import nn, train, Tensor

import argparse
import random
import os
import logging
import numpy as np
from tqdm import tqdm

from data import load_data
from resnet import resnet50
# from losses.centerloss import CenterLoss


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)


def _logging(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    fhandler = logging.FileHandler('./logs/' + args.dataset + '_train.log')
    fhandler.setLevel(logging.INFO)
    chandler = logging.StreamHandler()
    chandler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    chandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(chandler)

    logger.info(args)
    return logger


def run():
    seed_everything(68)
    args = load_args()
    logger = _logging(args)

    train_set, test_set = load_data(args.root, args.dataset, args.batch_size)
    logger.info('Data Loaded!')
    num_train, num_test = 18388, 5794
    logger.info('[train set size: {}][test set size: {}]'.format(num_train, num_test))
    train_loop(train_set, test_set, args, logger)


def train_loop(train_set, test_set, args, logger):
    network = resnet50(pretrained=True)
    in_channle = network.fc.in_channels
    fc = nn.Dense(in_channels=in_channle, out_channels=200)
    network.fc = fc
    for param in network.get_parameters():
        param.requires_grad = True
    step_size_train = train_set.get_dataset_size()
    step_size_val = test_set.get_dataset_size()

    lr = nn.cosine_decay_lr(min_lr=1e-5, max_lr=1e-2, total_step=step_size_train*args.epochs,
                            step_per_epoch=step_size_train, decay_epoch=5)

    opt = nn.Momentum(params=network.trainable_params(), learning_rate=args.lr, momentum=args.wd)
    loss_tr = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn1 = nn.CrossEntropyLoss()
    # lossc = CenterLoss(num_classes=200, feat_dim=200)
    # opt_centerloss = nn.Momentum(params=lossc.trainable_params(), learning_rate=0.5, momentum=args.wd)

    def forward_fn(inputs, targets):
        logits = network(inputs)
        loss = loss_fn(logits, targets)
        return loss
    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, opt.parameters)
    def train_step(inputs, targets):
        loss, grads = grad_fn(inputs, targets)
        opt(grads)
        return loss

    model = mindspore.Model(network, loss_fn, opt, metrics={'accuracy'})
    train_dataloader = train_set.create_tuple_iterator(num_epochs=args.epochs)

    best_answer = 0.
    logger.info('Start Trainging.....')
    for epoch in range(args.epochs):
        loss_sum = 0.
        network.set_train()
        pbar = tqdm(enumerate(train_dataloader), total=step_size_train)
        for it, (image, label) in pbar:
            loss = train_step(image, label)
            loss_sum += loss
        logger.info('[Train_Stage]][epoch:{}/{}][train_loss:{:.3f}]'.format(epoch, args.epochs, loss_sum/step_size_train))

        acc = model.eval(test_set)['accuracy']
        logger.info('[Test Stage][Test Accuracy:{:.3f}][Best Accuract untill:{:.3f}'.format(epoch, acc, best_answer))
        if best_answer < acc:
            best_answer = acc
    logger.info("[Done!][Bird][Best Accuracy:{:.3f}]".format(best_answer))


def load_args():
    parser = argparse.ArgumentParser(description='Webly Supervised Fine Grained Recongnition')
    parser.add_argument('--dataset', default='bird',
                        help='data set name')
    parser.add_argument('--root', default='../../dataset/web-bird')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--wd', default=2e-5, type=float)
    parser.add_argument('--optim', default='SGD', type=str)
    parser.add_argument('--momen', default=0.9, type=float)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--backbone', default='resnet50', type=str)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    ms.set_context(device_target="CPU")
    return args


if __name__ == '__main__':
    run()