from torch.utils import data
import os
import torchvision.transforms as transforms
import numpy as np
import cv2


class Dataset(data.Dataset):

    def __init__(self, path,folder_list, max_images = None):
        self.path = path
        self.images_folder = folder_list   #图片文件夹，每个小文件夹是一个样本（包含五张）
        if max_images != None:
            self.images_folder = self.images_folder[:max_images]
        self.size = 160
        self.burst_length = 4

    def __len__(self):
        return len(self.images_folder)

    def load_image(self,path):
        # image = cv2.cvtColor(cv2.imread(path))
        image=cv2.imread(path)
        image = cv2.resize(image, (self.size, self.size))
        # 图像归一化
        transform_GY = transforms.ToTensor()#将PIL.Image转化为tensor，即归一化。
        # 图像标准化
        transform_BZ= transforms.Normalize(
        mean=[0.5, 0.5, 0.5],# 取决于数据集
        std=[0.5, 0.5, 0.5]
        )
        # transform_compose
        transform_compose= transforms.Compose([
        # 先归一化再标准化
            transform_GY,
            transform_BZ
        ])
        # (H, W, C)变为(C, H, W)
        img_normalize = transform_compose(image)
        img_normalize=img_normalize.numpy()
        return img_normalize

    def __getitem__(self, index):
        images_path=os.path.join(self.path,self.images_folder[index])
        images=os.listdir(images_path)
        #找出target
        target=None   #此时的target是路径
        images_without_target=[]
        for image in images:
            if image.endswith('F.png'):
                target=image
            else:
                images_without_target.append(image)
        images=images_without_target

        #加载图片为numpy数组
        target=self.load_image(os.path.join(images_path,target))
        tmp=[]

        for i in range(self.burst_length):
            burst_i=self.load_image(os.path.join(images_path,images[i]))
            tmp.append(burst_i)

        burst=np.array(tmp) #转换成numpy格式

        #return两个，一个是x，另一个是label标签
        return burst,target


