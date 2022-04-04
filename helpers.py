import os
import numpy as np
import torchvision.transforms as transforms

# def get_newest_model(path):  #得到最新的model76
#     key_files = [(float(file.replace('best_model_at_epoch', '').replace('.pt', '')), file) for file in os.listdir(path) if
#                  'best_model_at_epoch' in file]
#
#     key_files = sorted(key_files, key=lambda x: x[0], reverse=True)
#     path = [os.path.join(path, basename[1]) for basename in key_files][0]
#     if '.pt' in path:
#         return path
#
#     print('Could not find any model')
#     return None

def get_newest_model_path(results_path):
    newest_path=None
    paths_list=os.listdir(results_path)
    k=len(paths_list)
    for path in paths_list:
        if path.startswith(str(k)):
            newest_path=path
            break

    if not newest_path:
        print('Could nont find any model!')
        return None

    return results_path+'/'+newest_path


def anti_normalize(img):
    #与normalize相反
    img=(img+1)/2
    img=img*np.iinfo(np.uint8).max
    img=np.clip(img,0,255)
    return img

def make_single_pred_img(pred_patch):
    imgs=[]
    for i in range(pred_patch.shape[0]):
        img=anti_normalize(pred_patch[i, :, :, :].detach().cpu().numpy())
        img=np.uint8(img)
        img=np.transpose(img,(1,2,0))
        imgs.append(img)

    return imgs

def make_compared_im(pred, X_batch, target):
    #无target则是测试集
    if target is not None:
        batch, im, c, w, h = X_batch.size()
        batch_image=[]
        for b in range(batch):
            image = np.zeros((3, h, (w + 5) * (im + 2)))
            image[:, :, :w] = anti_normalize(target[b, :, :, :].detach().cpu().numpy())

            image[:, :, (w+5):(2*w + 5)] = anti_normalize(pred[b, :, :, :].detach().cpu().numpy())

            for i in range(im):
                image[:, :, (2 + i) * w + (2 + i) * 5:(3 + i) * w + (2 + i) * 5] = anti_normalize(X_batch[b, i, :, :, :].detach().cpu().numpy())

            # image=image.astype(np.uint8)
            image=np.uint8(image)
            image=np.transpose(image,(1,2,0))
            batch_image.append(image)
        return batch_image
    else:
        batch, im, c, w, h = X_batch.size()
        batch_image=[]
        for b in range(batch):
            image = np.zeros((3, h, (w + 5) * (im + 1)))
            image[:, :, :w] = anti_normalize(pred[b, :, :, :].detach().cpu().numpy())
            for i in range(im):
                image[:, :, (1 + i) * w + (1 + i) * 5 : (2 + i) * w + (1 + i) * 5] = anti_normalize(X_batch[b, i, :, :, :].detach().cpu().numpy())

            # image=image.astype(np.uint8)
            image=np.uint8(image)
            image=np.transpose(image,(1,2,0))
            batch_image.append(image)
        return batch_image



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
