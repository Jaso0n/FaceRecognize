from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision as tv
from config import config as conf
from torchvision import transforms
import time

def load_data(conf, training=True):
    if training:
        dataroot = conf.train_root
        transform = conf.train_transform
        batch_size = conf.train_batch_size
    else:
        dataroot = conf.test_root
        transform = conf.lfw_test_transform
        batch_size = conf.lfw_test_batch_size

    data = ImageFolder(dataroot, transform=transform)
    class_num = len(data.classes)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, 
        pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader, class_num

if __name__ == '__main__':
    #check dataset
    loader, class_num = load_data(conf)
    to_pil_image = transforms.ToPILImage()
    for image, label in loader:
        img = to_pil_image(image[0])
        img.show()
        time.sleep(1)
    # OK