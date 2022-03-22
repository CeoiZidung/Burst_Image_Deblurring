import os
import numpy as np

def get_newest_model(path):
    key_files = [(float(file.replace('model_', '').replace('.pt', '')), file) for file in os.listdir(path) if
                 'model_' in file]
    key_files = sorted(key_files, key=lambda x: x[0], reverse=True)

    paths = [os.path.join(path, basename[1]) for basename in key_files]

    for path in paths:
        if '.pt' in path:
            return path

    print('Could not find any model')
    return None


# def save_img(img,save_path,img_name):
#     if img.dtype == np.float:
#         img = (img-img.min()) / (img.max()-img.min())
#         img = img * np.iinfo(np.uint8).max
#         img = img.astype(np.uint8)
#         # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#
#     cv2.imwrite(os.path.join(save_path,img_name), img)

def anti_normalize(img):
    img=(img - img.min()) / (img.max() - img.min())
    img=img*np.iinfo(np.uint8).max
    return img

def make_compared_im(pred, X_batch, target):
    b, im, c, w, h = X_batch.size()

    image = np.zeros((3, h, (w + 5) * (im + 2)))
    image[:, :, :w] = anti_normalize(target[0, :, :, :].detach().cpu().numpy())

    image[:, :, (w+5):(2*w + 5)] = anti_normalize(pred[0, :, :, :].detach().cpu().numpy())

    for i in range(im):
        image[:, :, (2 + i) * w + (2 + i) * 5:(3 + i) * w + (2 + i) * 5] = anti_normalize(X_batch[0, i, :, :, :].detach().cpu().numpy())

    image=image.astype(np.uint8)
    image=np.transpose(image,(1,2,0))
    return image


def load_namespace(file):
    file = open(file)
    namespace = file.read()
    file.close()

    namespace_dict = {}
    namespace = namespace.split('Namespace')[1].replace("(", "").replace(")", "").replace(" ", "").replace('"','').replace("'","").split(',')
    for attr in namespace:
        att, val = attr.split("=")
        namespace_dict[att] = val

    return namespace_dict
