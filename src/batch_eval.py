import os
import time
import sys
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

from utils.utils import calculate_psnr, calculate_ssim, calculate_ms_ssim, gram_matrix, check_dir
from models.networks import VGG19_Pretrained

from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm

from tqdm import tqdm

def cal_batch_ssim_psnr(content_path, transform_path, output_name=""):
    out_dir = check_dir("../output")
    if output_name != "":
        output = f"{out_dir}/statistics_{output_name}_psnr_ssim.csv"
    else:
        now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        output = f"{out_dir}/statistics_psnr_ssim_{now_time}.csv"
    
    total_psnr = 0
    total_ssim = 0
    total_ms_ssim = 0
    files = 0
    
    bar = tqdm(total=len(os.listdir(transform_path)), ncols=100)
    bar.set_description("Calculating PSNR, SSIM, MS-SSIM...")
    with open(output, "w") as f:
        f.write("content,transform,psnr,ssim,ms_ssim\n")
        for content_img_path in os.listdir(content_path):
            for transform_img_path in os.listdir(transform_path):
                content_name = content_img_path.split('.')[0]
                transform_name = transform_img_path.split('.')[0]
                
                if transform_name.find(content_name) != -1:
                    content = np.array(Image.open(f"{content_path}/{content_img_path}").convert('RGB'))
                    transform = np.array(Image.open(f"{transform_path}/{transform_img_path}").convert('RGB'))
                    if content.shape != transform.shape:
                        transform.resize(content.shape)
                
                    psnr = calculate_psnr(content, transform)
                    ssim = calculate_ssim(content, transform)
                    ms_ssim = calculate_ms_ssim(content, transform)
                    
                    total_psnr += psnr
                    total_ssim += ssim
                    total_ms_ssim += ms_ssim
                    files += 1
                    
                    f.write(f"{content_img_path},{transform_img_path},{psnr:.6f},{ssim:.6f},{ms_ssim:.6f}\n")
                    bar.update(1)
        f.write(f"Average,,{(total_psnr / files):.6f},{(total_ssim /files):.6f}, {(total_ms_ssim /files):.6f}\n")
    print(f"\n[INFO] Average PSNR: {(total_psnr / files):.6f}, Average SSIM: {(total_ssim /files):.6f}, "
          f"Average MS-SSIM: {(total_ms_ssim /files):.6f}")
    print(f"[INFO] Saved to {output}")
    
def cal_gram_cosine_similarity(style_path, transform_path, output_name=""):
    out_dir = check_dir("../output")
    if output_name != "":
        output = f"{out_dir}/statistics_{output_name}_cosine_similarity.csv"
    else:
        now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        output = f"{out_dir}/statistics_cosine_similarity_{now_time}.csv"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vgg19 = VGG19_Pretrained([26]).eval()
    vgg19 = vgg19.to(device)
    
    total = 0
    files = 0
    
    bar = tqdm(total=len(os.listdir(transform_path)), ncols=100)
    bar.set_description("Calculating cosine similarity...")
    with open(output, "w") as f:     
        f.write("content,transform,cosine_similarity\n")
        for style_img_path in os.listdir(style_path):
            for transform_img_path in os.listdir(transform_path):
                style_name = style_img_path.split('.')[0]
                transform_name = transform_img_path.split('.')[0]
                
                if transform_name.find(style_name) != -1:
                    style = Image.open(f"{style_path}/{style_img_path}").convert('RGB')
                    transform = Image.open(f"{transform_path}/{transform_img_path}").convert('RGB')
                    
                    height = style.height
                    width = style.width
                    
                    style = transforms.Compose([
                        transforms.ToTensor(), 
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])(style)
                    transform = transforms.Compose([
                        transforms.Resize((height, width)),
                        transforms.ToTensor(), 
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])(transform)
                    
                    style = style.unsqueeze(0).to(device)
                    transform = transform.unsqueeze(0).to(device)
                    style_features = vgg19(style)
                    transform_features = vgg19(transform)
                    
                    gram1 = gram_matrix(style_features[0])
                    gram2 = gram_matrix(transform_features[0])
                    gram1 = gram1.flatten()
                    gram2 = gram2.flatten()

                    norm1 = torch.norm(gram1)
                    norm2 = torch.norm(gram2)
                    cosine_sim = torch.dot(gram1, gram2) / (norm1 * norm2 + 1e-8)
                    
                    total += cosine_sim.item()

                    files += 1
                    
                    f.write(f"{style_img_path},{transform_img_path},{cosine_sim.item():.6f}\n")
                    bar.update(1)
        f.write(f"Average,,{(total /files):.6f}\n")
    print(f"\n[INFO] Average cosine similarity: {(total /files):.6f}")
    print(f"[INFO] Saved to {output}")

def cal_fid(content_path, style_path, transform_path, output_name=""):
    out_dir = check_dir("../output")
    if output_name != "":
        output = f"{out_dir}/statistics_{output_name}_fid.csv"
    else:
        now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        output = f"{out_dir}/statistics_fid_{now_time}.csv"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True).eval()
    model.aux_logits = False
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def get_features1(path: str):
        feats = []
        for img in os.listdir(path):
            img_path = f"{path}/{img}"
            img = Image.open(img_path).convert('RGB')
            tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.inference_mode():
                feat = model(tensor).cpu().numpy().reshape(-1)
            feats.append(feat)
        return np.array(feats)
    
    def get_features2(paths: list[str]):
        feats = []
        for img in paths:
            img = Image.open(img).convert('RGB')
            tensor = preprocess(img).unsqueeze(0).to(device)
            
            with torch.inference_mode():
                feat = model(tensor).cpu().numpy().reshape(-1)
            feats.append(feat)
        return np.array(feats)
    
    def get_batch_fid(content_path, transforms: list[str]):
        content_feats = get_features1(content_path)
        transform_feats = get_features2(transforms)
        
        mu1, sigma1 = np.mean(content_feats, axis=0), np.cov(content_feats, rowvar=False)
        mu2, sigma2 = np.mean(transform_feats, axis=0), np.cov(transform_feats, rowvar=False)
        
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid
    
    avg_fid = 0
    
    bar = tqdm(os.listdir(style_path), ncols=100)
    bar.set_description("Calculating FID...")
    with open(output, "w") as f:
        f.write("style,fid\n")

        for style in bar:
            style = style.split('.')[0]
            this_trans = []
            for transform in os.listdir(transform_path):
                if transform.find(style) != -1:
                    this_trans.append(f"{transform_path}/{transform}")
            fid = get_batch_fid(content_path, this_trans)
            avg_fid += fid
            
            f.write(f"{style},{fid:.6f}\n")
        
        avg_fid /= len(os.listdir(style_path))
    
        f.write(f"Average,{avg_fid:.6f}\n")
        
    print(f"\n[INFO] FID: {avg_fid}")
    print(f"[INFO] Saved to {output}")
    
if __name__ == "__main__":
    if len(sys.argv) < 5 or sys.argv[4] not in ["1", "2", "3"]:
        print(f"[ERROR] Usage: python {sys.argv[0]} <content_path> <style_path> <transform_path> <mode> [output_name]")
        print(f"[ERROR] mode: 1 -> psnr and ssim, 2 -> Gram 余弦相似度, 3 -> FID")
        exit()
    
    content_path = sys.argv[1]
    style_path = sys.argv[2]
    transform_path = sys.argv[3]
    mode = sys.argv[4]
    
    if len(sys.argv) == 6:
        output_name = sys.argv[5]
    else:
        output_name = ""
    
    if not os.path.exists(content_path) or not os.path.exists(style_path) or not os.path.exists(transform_path):
        print(f"[ERROR] Path {content_path} or {style_path} or {transform_path} not exist")
        exit()

    if mode == "1":
        cal_batch_ssim_psnr(content_path, transform_path, output_name)
    elif mode == "2":
        cal_gram_cosine_similarity(style_path, transform_path, output_name)
    elif mode == "3":
        cal_fid(content_path, style_path, transform_path, output_name)
