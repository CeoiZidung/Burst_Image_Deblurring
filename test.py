import torchvision.transforms as transforms
import numpy as np
import cv2

img_path = "./dataSet/001/multi_595_1.png"
img=cv2.imread(img_path,cv2.COLOR_BGR2RGB)
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
    transform_GY ,
    transform_BZ
])
# (H, W, C)变为(C, H, W)
img_transform = transform_compose(img)
# 输出变换后图像，需要将图像格式调整为PIL.Image形式
img = img_transform .numpy()   #

if img.dtype == np.float32:
    img=(img-img.min()) / (img.max()-img.min())
    img=img * np.iinfo(np.uint8).max
    img=img.astype(np.uint8)
    # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

cv2.imwrite('./1.png',img)










