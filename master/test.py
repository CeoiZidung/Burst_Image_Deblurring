import torch
import os
import time



device='cuda:0' if torch.cuda.is_available() else 'cpu'

tensor_4=torch.randn(120,3,512,512).float().to(device)
tensor_5=torch.randn(80,3,512,512).float().to(device)
tensor_44=torch.randn(120,3,512,512).float().to(device)
tensor_55=torch.randn(80,3,512,512).float().to(device)

time.sleep(1)
print('nvidia-smi start')
print(os.system('nvidia-smi -i 0'))

gpu_memory_bf=torch.cuda.memory_allocated(device=device)
print('the gpu memory before tensor to cpu: {}'.format(gpu_memory_bf/1024/1024))


gpu_memory_re=torch.cuda.memory_reserved(device=device)
print('gpu_memory_re: {}'.format(gpu_memory_re/1024/1024))

tensor_4=tensor_4.cpu()
tensor_5=tensor_5.cpu()
gpu_memory_bf=torch.cuda.memory_allocated(device=device)
print('the gpu memory after tensor to cpu: {}'.format(gpu_memory_bf/1024/1024))


gpu_memory_re=torch.cuda.memory_reserved(device=device)
print('gpu_memory_re: {}'.format(gpu_memory_re/1024/1024))
print(os.system('nvidia-smi -i 0'))


torch.cuda.empty_cache()
gpu_memory_re=torch.cuda.memory_reserved(device=device)
print('After clearing cache: {}'.format(gpu_memory_re/1024/1024))
print(os.system('nvidia-smi -i 0'))
