import mindspore
from mindspore.dataset import vision, transforms
from mindspore.dataset import GeneratorDataset
import mindspore.numpy as np
import numpy
import os
from PIL import Image
from mindspore.dataset import MindDataset
from mindspore import dtype as mstype


def load_data(root, dataset, batch_size):
    data_transform = transforms.Compose([
        vision.Decode(),
        vision.Resize([224, 224]),
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        vision.HWC2CHW()
    ])
    target_trans = transforms.TypeCast(mstype.int32)

    if dataset == 'bird':
        train_data_path = 'web-bird-train.mindrecord'
        test_data_path = 'web-bird-test.mindrecord'
        train_data_set = MindDataset(dataset_files=train_data_path, columns_list=['data', 'label'], shuffle=True)
        train_data_set = train_data_set.map(
            operations=data_transform,
            input_columns='data',
            num_parallel_workers=4)
        train_data_set = train_data_set.map(
            operations=target_trans,
            input_columns='label',
            num_parallel_workers=4)
        test_data_set = MindDataset(dataset_files=test_data_path, columns_list=['data', 'label'], shuffle=False)
        test_data_set = test_data_set.map(
            operations=data_transform,
            input_columns='data',
            num_parallel_workers=4)
        test_data_set = test_data_set.map(
            operations=target_trans,
            input_columns='label',
            num_parallel_workers=4)
        train_data_set = train_data_set.batch(batch_size)
        test_data_set = test_data_set.batch(batch_size)
    return train_data_set, test_data_set


class Iterable:
    def __init__(self, root, mode, data_transform=None, label_transform=None):
        self.root = root
        self.mode = mode
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.data = []
        self.label = []
        if mode == 'train':
            self.list_file = os.path.join(root, 'train-list.txt')
        elif mode == 'test':
            self.list_file = os.path.join(root, 'val-list.txt')
        self.data, self.label = self.get_data()

    def get_data(self):
        with open(self.list_file, 'r') as f:
            strs = f.readlines()
            for s in strs:
                s = s.split(' ')
                self.data.append(os.path.join(self.root, s[0][:-1]))
                self.label.append(int(s[1][:-1]))
        return self.data, self.label

    def __getitem__(self, index):
        data_file = self.data[index]
        img = Image.open(data_file).convert('RGB')
        data = numpy.array(img)
        label = self.label[index]
        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return data, label

    def __len__(self):
        # train 18388
        # test 5794
        return len(self.label)


if __name__ == '__main__':
    from PIL import Image
    from io import BytesIO
    from mindspore.mindrecord import FileWriter
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    root = '../../dataset/web-bird'
    file_name = 'web-bird-test.mindrecord'
    cv_schema = {"file_name": {"type": "string"},
                 "label": {"type": "int32"},
                 "data": {"type": "bytes"}}
    writer = FileWriter(file_name, shard_num=1, overwrite=True)
    writer.add_schema(cv_schema, "it is a cv dataset")
    writer.add_index(['file_name', 'label'])

    train_set = Iterable(root, 'train')
    test_set = Iterable(root, 'test')
    train_data = train_set.data
    train_label = train_set.label

    test_data = test_set.data
    test_label = test_set.label

    n_data = len(test_data)
    data = []
    for i in range(n_data):
        sample = {}
        white_io = BytesIO()
        Image.open(test_data[i]).convert('RGB').save(white_io, 'JPEG')
        image_bytes = white_io.getvalue()
        sample['file_name'] = str(i)+'.jpg'
        sample['label'] = test_label[i]
        sample['data'] = white_io.getvalue()
        data.append(sample)
        if i % 100 == 0:
            writer.write_raw_data(data)
            data = []
    if data:
        writer.write_raw_data(data)

    writer.commit()

    # from mindspore.dataset import MindDataset
    # import tqdm
    # file_name = 'web-bird-train.mindrecord'
    # data_set = MindDataset(dataset_files=file_name, columns_list=['data', 'label'], shuffle=False)
    #
    # data_transform = transforms.Compose([
    #     vision.Decode(),
    #     vision.Resize([224, 224]),
    #     vision.Rescale(1.0 / 255.0, 0),
    #     vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     vision.HWC2CHW()
    # ])
    # data_set = data_set.map(
    #     operations=data_transform,
    #     input_columns='data',
    #     num_parallel_workers=4)
    #
    # data_set = data_set.batch(32)
    # data_loader_train = data_set.create_tuple_iterator(num_epochs=110)
    # for i, (images, labels) in enumerate(data_loader_train):
    #     print(images.shape)
    #     print(i)

