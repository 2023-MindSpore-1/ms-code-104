import mindspore
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.dataset import vision, transforms
from mindspore.dataset import GeneratorDataset


class ClassificationPresetTrain:
    def __init__(
        self,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
    ):
        trans = [vision.RandomResizedCrop(crop_size)]
        if hflip_prob > 0:
            trans.append(vision.RandomHorizontalFlip(hflip_prob))

        trans.extend(
            [
                vision.Decode(),
                transforms.TypeCast(mindspore.float32),
                vision.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(vision.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):

        self.transforms = transforms.Compose(
            [
                vision.Resize(resize_size),
                vision.CenterCrop(crop_size),
                vision.Decode(),
                transforms.TypeCast(mindspore.float32),
                vision.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)