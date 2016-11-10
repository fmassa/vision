import torch.utils.data as data
from PIL import Image
import os
import os.path

import xml.etree.ElementTree



class VOCDetection(data.Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        dataset_name = 'VOC2007'
        self.annopath = os.path.join(self.root,dataset_name,'Annotations','%s.xml')
        self.imgpath = os.path.join(self.root,dataset_name,'JPEGImages','%s.jpg')
        self.imgsetpath = os.path.join(self.root,dataset_name,'ImageSets','Main','%s.txt')
 
        with open(self.imgsetpath%self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = xml.etree.ElementTree.parse(self.annopath % img_id).getroot()

        img = Image.open(self.imgpath % img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    ds = VOCDetection('/home/francisco/work/datasets/VOCdevkit/', 'train')
    print(len(ds))
    img, target = ds[0]
    img.show()
    print(target)
