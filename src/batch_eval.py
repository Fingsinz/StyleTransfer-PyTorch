import os
import time
import sys
from PIL import Image
import numpy as np

from utils.utils import calculate_psnr, calculate_ssim


if __name__ == "__main__":
    content_path = sys.argv[1]
    transform_path = sys.argv[2]
    
    if not os.path.exists(content_path) or not os.path.exists(transform_path):
        print(f"[ERROR] {content_path} or {transform_path} not exist")
        exit()
    
    now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    output = f"metrics{now_time}.csv"
    
    total_psnr = 0
    total_ssim = 0
    files = 0
    
    with open(output, "w") as f:
        f.write("content,transform,psnr,ssim\n")
        for content_img_path in os.listdir(content_path):
            for transform_img_path in os.listdir(transform_path):
                content_name = content_img_path.split('.')[0]
                transform_name = transform_img_path.split('.')[0]
                
                if transform_name.find(content_name) != -1:
                    content = np.array(Image.open(f"{content_path}/{content_img_path}").convert('RGB'))
                    transform = np.array(Image.open(f"{transform_path}/{transform_img_path}").convert('RGB'))
                    transform.resize(content.shape)
                
                    psnr = calculate_psnr(content, transform)
                    ssim = calculate_ssim(content, transform)
                    
                    total_psnr += psnr
                    total_ssim += ssim
                    files += 1
                    
                    f.write(f"{content_img_path},{transform_img_path},{psnr:.6f},{ssim:.6f}\n")
        f.write(f"Average,,{(total_psnr / files):.6f},{(total_ssim /files):.6f}\n")    
    