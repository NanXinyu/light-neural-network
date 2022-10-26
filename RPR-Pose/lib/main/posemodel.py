import torch
import torch.nn as nn
import numpy as np

def generate_token_target(joints):
    sigma = 2
    num_joints = 1
    num = 256
    num_h = 16
    patch = [16, 16]
    target = np.zeros((num_joints, num), dtype = np.float32)
    
       
    div = np.full((num), num_h, dtype = np.float32)
    print("div={}".format(div))
    offset_x = np.full((num), patch[1] * 0.5, dtype = np.float32)
    offset_y = np.full((num), patch[0] * 0.5, dtype = np.float32) 
    print("offset={}".format(offset_x))

        
    for joint_id in range(num_joints):
        mu_x = np.full((256), joints[joint_id][0], dtype = np.float32)
        mu_y = np.full((256), joints[joint_id][1], dtype = np.float32)
        
        x = (np.arange(0, num, 1, np.float32) // div) * patch[1] + offset_x
        y = (np.arange(0, num, 1, np.float32) % div) * patch[0] + offset_y
        
        target[joint_id] = (np.exp(-0.5 * (((x - mu_x)**2 + (y - mu_y)**2) / sigma**2)))/(np.pi*2*(sigma**2))
    
    print("x={}".format(x))
    print("y={}".format(y))
    return target
if __name__ == "__main__":
    x = torch.rand(1, 2)*255
    x = x.detach().numpy()
    print(x)
    target = generate_token_target(x)
    print(target)