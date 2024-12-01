import math
import time
from pathlib import Path
import argparse
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import cv2
from pathlib import Path
import os

class LoadGaussians:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        num_points,
        image,
        device,
        Model = None,
        args = None,
    ):
        self.device = device
        self.gt_image = image_to_tensor(image).to(self.device)
        self.num_points=num_points
        self.data_name=args.data_name
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        
        from GaussianSplats_Compress import GaussianVideo_frame
        self.gaussian_model = GaussianVideo_frame(num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
        device=self.device,quantize=True).to(self.device)
        if Model is not None:
            checkpoint = Model
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)
 
    def test(self,):
        self.gaussian_model.eval()
        with torch.no_grad():
            encoding_dict = self.gaussian_model.compress_wo_ec()
            out = self.gaussian_model.decompress_wo_ec(encoding_dict)
            start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model.decompress_wo_ec(encoding_dict)
            end_time = (time.time() - start_time)/100
        data_dict = self.gaussian_model.analysis_wo_ec(encoding_dict)
    
        out_img = out["render"].float()
        mse_loss = F.mse_loss(out_img, self.gt_image)
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out_img, self.gt_image, data_range=1, size_average=True).item()
        
        data_dict["psnr"] = psnr
        data_dict["ms-ssim"] = ms_ssim_value
        data_dict["rendering_time"] = end_time
        data_dict["rendering_fps"] = 1/end_time
        # np.save(self.log_dir / "test.npy", data_dict)
        # self.logwriter.write("Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, 1/end_time))
        # self.logwriter.write("PSNR:{:.4f}, MS_SSIM:{:.6f}, bpp:{:.4f}".format(psnr, ms_ssim_value, data_dict["bpp"]))
        # self.logwriter.write("position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(data_dict["position_bpp"], data_dict["cholesky_bpp"], data_dict["feature_dc_bpp"]))
        return data_dict
    
    def test_o(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
            out_image = out["render"]
        mse_loss = F.mse_loss(out_image.float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out_image.float(), self.gt_image.float(), data_range=1, size_average=True).item()
        return psnr, ms_ssim_value

def image_to_tensor(img: Image.Image):
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv', help="Dataset Path"
    )
    parser.add_argument(
         "--model_path", type=str, default='/home/e/e1344641/GaussianVideo/models/Models/Beauty/GaussianImage_Cholesky_100000_30000/gmodels_state_dict.pth', help="Model Path"
    )
    parser.add_argument(
        "--data_name", type=str, default='Beauty', help="Training dataset"
    )
    parser.add_argument(
        "--fps", type=int, default=120, help="number of frames per second (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=4000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument(
        "--width", type=int, default=1920, help="width (default: %(default)s)"
    )
    parser.add_argument(
        "--height", type=int, default=1080, help="height (default: %(default)s)"
    )
    parser.add_argument("--quantize", action="store_true", help="Quantize")
    parser.add_argument("--savdir", type=str, default="result", help="Path to results")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    args = parser.parse_args(argv)
    return args

def main(argv):
    step=10
    args = parse_args(argv)
    savdir=args.savdir
    fps=5
    width = args.width
    height = args.height
    model_path=args.model_path
    num_points=args.num_points
    device=torch.device("cuda:0")
    video_frames = process_yuv_video(args.dataset, width, height)
    image_length,start=len(video_frames),0
    # image_length=50
    logwriter = LogWriter(Path(f"./checkpoints/{savdir}/{args.data_name}/{args.num_points}"))
    psnrs, ms_ssims, eval_times, eval_fpses, bpps = [], [], [], [], []
    position_bpps, cholesky_bpps, feature_dc_bpps = [], [], []
    print(f"loading model path:{model_path}")
    gmodels_state_dict = torch.load(model_path,map_location=device)
    image_h, image_w = 0, 0
    # for i in tqdm(range(start, start + image_length), desc="Processing Frames"):
    for i in range(start, start + image_length):
        frame_num=i+1
        modelid=f"frame_{i + 1}"
        Model = gmodels_state_dict[modelid]
        Gaussianframe = LoadGaussians(num_points=num_points,image=video_frames[i], Model=Model,device=device,args=args)
        #data_dict = Gaussianframe.test()
        psnr, ms_ssim_value = Gaussianframe.test_o()
        logwriter.write("Frame_{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}".format(
            frame_num, Gaussianframe.H, Gaussianframe.W, psnr,  ms_ssim_value))




    #     psnrs.append(data_dict["psnr"])
    #     ms_ssims.append(data_dict["ms-ssim"])
    #     eval_times.append(data_dict["rendering_time"])
    #     eval_fpses.append(data_dict["rendering_fps"])
    #     bpps.append(data_dict["bpp"])
    #     position_bpps.append(data_dict["position_bpp"])
    #     cholesky_bpps.append(data_dict["cholesky_bpp"])
    #     feature_dc_bpps.append(data_dict["feature_dc_bpp"])
    #     image_h += Gaussianframe.H
    #     image_w += Gaussianframe.W
    #     logwriter.write("Frame_{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, Eval:{:.8f}s, FPS:{:.4f}, position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(
    #         frame_num, Gaussianframe.H, Gaussianframe.W, data_dict["psnr"],  data_dict["ms-ssim"], data_dict["bpp"], 
    #         data_dict["rendering_time"], data_dict["rendering_fps"], 
    #         data_dict["position_bpp"], data_dict["cholesky_bpp"], data_dict["feature_dc_bpp"]))

    # avg_psnr = torch.tensor(psnrs).mean().item()
    # avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    # avg_eval_time = torch.tensor(eval_times).mean().item()
    # avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    # avg_bpp = torch.tensor(bpps).mean().item()
    # avg_position_bpp = torch.tensor(position_bpps).mean().item()
    # avg_cholesky_bpp = torch.tensor(cholesky_bpps).mean().item()
    # avg_feature_dc_bpp = torch.tensor(feature_dc_bpps).mean().item()
    # avg_h = image_h//image_length
    # avg_w = image_w//image_length

    # logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, Eval:{:.8f}s, FPS:{:.4f}, position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(
    #     avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_bpp, avg_eval_time, avg_eval_fps, 
    #     avg_position_bpp, avg_cholesky_bpp, avg_feature_dc_bpp))    

if __name__ == "__main__":
    
    main(sys.argv[1:])


