from gsvc_runtime import configure_runtime_environment
configure_runtime_environment()
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
from utils import (
    LogWriter,
    generate_video_decoder,
    log_run_config,
    process_yuv_video_frame_start_end,
    save_parameter_distribution_plots,
)
import random
import torchvision.transforms as transforms
from quantize import FakeQuantizationFP16
from GSVC_decoder import GaussianVideo_frame, GaussianVideo_delta
class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image,
        frame_num,
        savdir,
        model_name:str = "GaussianVideo",
        quantized_model = None,
        previous_quantized_model=None,
        args = None,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_to_tensor(image).to(self.device)
        self.frame_num=frame_num
        self.model_name=model_name
        self.data_name=args.data_name
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.save_imgs = args.save_imgs
        self.m_bit, self.s_bit, self.r_bit, self.c_bit = 0, 0, 0, 0
        self.p_trained_model=previous_quantized_model
        self.point_ids = None
        if previous_quantized_model is not None:
            checkpoint = quantized_model
            p_checkpoint = previous_quantized_model
            cur_len = checkpoint['_features_dc'].shape[0]
            self.num_points = cur_len
            self.log_dir = Path(f"./checkpoints_quant/{savdir}/{args.data_name}/{args.model_name}_{self.num_points}")
            self.gaussian_model = GaussianVideo_delta(
                num_points=self.num_points,
                H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, 
                quantize=True,
                add_num_Gaussians=0,
                xyz_format=checkpoint.get('xyz_format')).to(self.device)
            model_dict = self.gaussian_model.state_dict()
            delta_model_dict = self.gaussian_model.state_dict()
            has_ids = 'point_ids' in checkpoint
            if has_ids:
                self.cur_ids = checkpoint['point_ids'].detach().cpu().numpy()
                self.point_ids = torch.as_tensor(self.cur_ids, dtype=torch.long)
                prev_ids = p_checkpoint['point_ids'].detach().cpu().numpy()
                prev_index = {int(i): idx for idx, i in enumerate(prev_ids.tolist())}
                device = self.device


                def _count_unmatched(prev_index_map, cur_ids_list):
                    unmatched = 0
                    for pid in cur_ids_list:
                        if prev_index_map.get(int(pid), -1) == -1:
                            unmatched += 1
                    return unmatched
                self.add_num_Gaussians = _count_unmatched(prev_index, self.cur_ids)
                self.gaussian_model.add_num_Gaussians = self.add_num_Gaussians
                with torch.no_grad():
                    if checkpoint.get('xyz_format') != 'fp12_e5m6_raw':
                        self.gaussian_model.xyz_quantizer.scale.data.copy_(
                            checkpoint['xyz_scale'].to(self.device).float().view_as(self.gaussian_model.xyz_quantizer.scale)
                        )
                        self.gaussian_model.xyz_quantizer.beta.data.copy_(
                            checkpoint['xyz_beta'].to(self.device).float().view_as(self.gaussian_model.xyz_quantizer.beta)
                        )
                    self.gaussian_model.cholesky_quantizer.scale.data.copy_(
                        checkpoint['cholesky_scale'].to(self.device).float().view_as(self.gaussian_model.cholesky_quantizer.scale)
                    )
                    self.gaussian_model.cholesky_quantizer.beta.data.copy_(
                        checkpoint['cholesky_beta'].to(self.device).float().view_as(self.gaussian_model.cholesky_quantizer.beta)
                    )
                    codebooks = checkpoint.get('features_codebooks', None)
                    if codebooks is not None:
                        if getattr(self.gaussian_model.features_dc_quantizer, 'num_quantizers', 1) == 1:
                            with torch.no_grad():
                                self.gaussian_model.features_dc_quantizer.quantizer._codebook.embed.data.copy_(
                                    codebooks[0].to(self.device).view_as(self.gaussian_model.features_dc_quantizer.quantizer._codebook.embed)
                                )
                        else:
                            for li, layer in enumerate(self.gaussian_model.features_dc_quantizer.quantizer.layers):
                                with torch.no_grad():
                                    layer._codebook.embed.data.copy_(
                                        codebooks[li].to(self.device).view_as(layer._codebook.embed)
                                    )
                    if self.add_num_Gaussians > 0:
                        self.gaussian_model.cholesky_quantizer_I.scale.data.copy_(
                            checkpoint['cholesky_scale_I'].to(self.device).float().view_as(self.gaussian_model.cholesky_quantizer.scale)
                        )
                        self.gaussian_model.cholesky_quantizer_I.beta.data.copy_(
                            checkpoint['cholesky_beta_I'].to(self.device).float().view_as(self.gaussian_model.cholesky_quantizer.beta)
                        )
                        codebooks_I = checkpoint.get('features_codebooks_I', None)
                        if codebooks_I is not None:
                            if getattr(self.gaussian_model.features_dc_quantizer_I, 'num_quantizers', 1) == 1:
                                with torch.no_grad():
                                    self.gaussian_model.features_dc_quantizer_I.quantizer._codebook.embed.data.copy_(
                                        codebooks_I[0].to(self.device).view_as(self.gaussian_model.features_dc_quantizer_I.quantizer._codebook.embed)
                                    )
                            else:
                                for li, layer in enumerate(self.gaussian_model.features_dc_quantizer_I.quantizer.layers):
                                    with torch.no_grad():
                                        layer._codebook.embed.data.copy_(
                                            codebooks_I[li].to(self.device).view_as(layer._codebook.embed)
                                        )
                        
                if self.add_num_Gaussians > 0:
                    num_delta_Gaussians = self.num_points-self.add_num_Gaussians
                    # Create empty _xyz tensor first
                    checkpoint['_xyz'] = torch.zeros(self.num_points, 2, device=self.device)
                    q_xyz_front = checkpoint['_xyz_front'].to(self.device)
                    if checkpoint.get('xyz_format') == 'fp12_e5m6_raw':
                        decoded_front = self.gaussian_model.xyz_quantizer.decompress(q_xyz_front, num_delta_Gaussians * 2).reshape(num_delta_Gaussians, 2)
                        m_bit_front = num_delta_Gaussians * 2 * 12
                    else:
                        q_xyz_front = q_xyz_front[:num_delta_Gaussians].float()
                        decoded_front = self.gaussian_model.xyz_quantizer.decompress(q_xyz_front)
                        m_bit_front = self.gaussian_model.xyz_quantizer.size(q_xyz_front)
                    m_bit_back = FakeQuantizationFP16.size(checkpoint['_xyz_back'])
                    checkpoint['_xyz'][:num_delta_Gaussians] = decoded_front
                    checkpoint['_xyz'][num_delta_Gaussians:] = checkpoint['_xyz_back'].to(self.device).float() 
                    self.m_bit = m_bit_front + m_bit_back

                    q_ch_front = checkpoint['_cholesky'][:num_delta_Gaussians].to(self.device).float()
                    q_ch_back = checkpoint['_cholesky'][num_delta_Gaussians:].to(self.device).float()
                    s_bit_front= self.gaussian_model.cholesky_quantizer.size(q_ch_front)
                    s_bit_back= self.gaussian_model.cholesky_quantizer_I.size(q_ch_back)
                    checkpoint['_cholesky'][:num_delta_Gaussians] = self.gaussian_model.cholesky_quantizer.decompress(q_ch_front)
                    checkpoint['_cholesky'][num_delta_Gaussians:] = self.gaussian_model.cholesky_quantizer_I.decompress(q_ch_back)
                    
                    self.s_bit = s_bit_front+s_bit_back
                    
                    feat_idx_front = checkpoint['_features_dc'][:num_delta_Gaussians].to(self.device).long()
                    feat_idx_back = checkpoint['_features_dc'][num_delta_Gaussians:].to(self.device).long()
                    c_bit_front= self.gaussian_model.features_dc_quantizer.size(feat_idx_front)
                    c_bit_back= self.gaussian_model.features_dc_quantizer_I.size(feat_idx_back)
                    color_front = self.gaussian_model.features_dc_quantizer.decompress(feat_idx_front)
                    color_back = self.gaussian_model.features_dc_quantizer_I.decompress(feat_idx_back)
                    color = torch.cat([color_front, color_back], dim=0)
                    
                    self.c_bit = c_bit_front + c_bit_back

                    checkpoint['_features_dc'] = color

                else:
                    q_xyz = checkpoint['_xyz'].to(self.device)
                    if checkpoint.get('xyz_format') == 'fp12_e5m6_raw':
                        checkpoint['_xyz'] = self.gaussian_model.xyz_quantizer.decompress(q_xyz, self.num_points * 2).reshape(self.num_points, 2)
                        self.m_bit = self.num_points * 2 * 12
                    else:
                        q_xyz = q_xyz.float()
                        checkpoint['_xyz'] = self.gaussian_model.xyz_quantizer.decompress(q_xyz)
                        self.m_bit = self.gaussian_model.xyz_quantizer.size(q_xyz)

                    q_ch = checkpoint['_cholesky'].to(self.device).float()
                    checkpoint['_cholesky'] = self.gaussian_model.cholesky_quantizer.decompress(q_ch)
                    self.s_bit= self.gaussian_model.cholesky_quantizer.size(q_ch)

                    feat_idx = checkpoint['_features_dc'].to(self.device).long()
                    checkpoint['_features_dc'] = self.gaussian_model.features_dc_quantizer.decompress(feat_idx)
                    self.c_bit= self.gaussian_model.features_dc_quantizer.size(feat_idx)

                # Build aligned tensors: current params in order, previous-frame params matched by point_id (zeros for new)
                def align_param(name):
                    cur = checkpoint[name].to(device)
                    dtype = cur.dtype
                    shape_tail = cur.shape[1:]
                    aligned_prev = torch.zeros((cur_len,)+shape_tail, dtype=dtype, device=device)
                    for ci, pid in enumerate(self.cur_ids):
                        pi = prev_index.get(int(pid), -1)
                        if pi != -1:
                            aligned_prev[ci] = p_checkpoint[name][pi].to(device)
                    return cur, aligned_prev

                cur_xyz, prev_xyz = align_param('_xyz')
                cur_ch, prev_ch = align_param('_cholesky')
                cur_feat, prev_feat = align_param('_features_dc')

                
                # Set deltas (current - prev) as learnable params; set previous as buffers p_*
                delta_model_dict['_xyz'] = cur_xyz
                delta_model_dict['_cholesky'] = cur_ch
                delta_model_dict['_features_dc'] = cur_feat
                delta_model_dict['p_xyz'] = prev_xyz
                delta_model_dict['p_cholesky'] = prev_ch
                delta_model_dict['p_features_dc'] = prev_feat
                self.gaussian_model.load_state_dict(delta_model_dict, strict=False)
            else:
                # Require stable point IDs for alignment in P-frames
                raise ValueError("no points id in checkpoints")
        else:
            checkpoint = quantized_model
            self.num_points = checkpoint['_xyz'].shape[0]
            self.log_dir = Path(f"./checkpoints_quant/{savdir}/{args.data_name}/{args.model_name}_{self.num_points}")
            if 'point_ids' in checkpoint:
                self.point_ids = checkpoint['point_ids'].detach().cpu().long()
            else:
                self.point_ids = torch.arange(self.num_points, dtype=torch.long)
            self.gaussian_model = GaussianVideo_frame(
                num_points=self.num_points,
                H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, 
                quantize=True).to(self.device)

            with torch.no_grad():
                self.gaussian_model.cholesky_quantizer.scale.data.copy_(
                    checkpoint['cholesky_scale'].to(self.device).float().view_as(self.gaussian_model.cholesky_quantizer.scale)
                )
                self.gaussian_model.cholesky_quantizer.beta.data.copy_(
                    checkpoint['cholesky_beta'].to(self.device).float().view_as(self.gaussian_model.cholesky_quantizer.beta)
                )
            codebooks = checkpoint.get('features_codebooks', None)
            if codebooks is not None:
                if getattr(self.gaussian_model.features_dc_quantizer, 'num_quantizers', 1) == 1:
                    with torch.no_grad():
                        self.gaussian_model.features_dc_quantizer.quantizer._codebook.embed.data.copy_(
                            codebooks[0].to(self.device).view_as(self.gaussian_model.features_dc_quantizer.quantizer._codebook.embed)
                        )
                else:
                    for li, layer in enumerate(self.gaussian_model.features_dc_quantizer.quantizer.layers):
                        with torch.no_grad():
                            layer._codebook.embed.data.copy_(
                                codebooks[li].to(self.device).view_as(layer._codebook.embed)
                            )
            xyz_q = checkpoint['_xyz'].to(self.device).float() 
            self.m_bit = FakeQuantizationFP16.size(xyz_q)

            q_ch = checkpoint['_cholesky'].to(self.device).float()
            ch_pre = self.gaussian_model.cholesky_quantizer.decompress(q_ch)
            self.s_bit= self.gaussian_model.cholesky_quantizer.size(q_ch)

            feat_idx = checkpoint['_features_dc'].to(self.device).long()
            cols = self.gaussian_model.features_dc_quantizer.decompress(feat_idx)
            self.c_bit= self.gaussian_model.features_dc_quantizer.size(feat_idx)
            
            model_dict = self.gaussian_model.state_dict()
            model_dict['_xyz'] = xyz_q
            model_dict['_cholesky'] = ch_pre
            model_dict['_features_dc'] = cols
            self.gaussian_model.load_state_dict(model_dict, strict=False)
    
    def render(self): 
        bpp=0
        bpp_xy = self.m_bit/self.H/self.W
        bpp_cholesky = self.s_bit/self.H/self.W
        bpp_color = self.c_bit/self.H/self.W
        
        bpp = (self.m_bit + self.s_bit + self.r_bit + self.c_bit)/self.H/self.W
        psnr_value, ms_ssim_value, img = self.test()
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100  
            fps = 1/test_end_time
            Gmodel = self.gaussian_model.state_dict()
            if self.p_trained_model is not None:
                q_xyz = self.gaussian_model._xyz + self.gaussian_model.p_xyz
                q_ch = self.gaussian_model._cholesky + self.gaussian_model.p_cholesky
                q_feat = self.gaussian_model._features_dc + self.gaussian_model.p_features_dc
                distribution_model = {
                    '_xyz': self.gaussian_model._xyz.detach().clone(),
                    '_cholesky': self.gaussian_model._cholesky.detach().clone(),
                    '_features_dc': self.gaussian_model._features_dc.detach().clone(),
                    'point_ids': self.point_ids.detach().clone() if torch.is_tensor(self.point_ids) else torch.as_tensor(self.point_ids, dtype=torch.long),
                }
                decoded_Gmodel = {
                    '_xyz': q_xyz.detach().clone(),
                    '_cholesky': q_ch.detach().clone(),
                    '_features_dc': q_feat.detach().clone(),
                    'point_ids': self.point_ids.detach().clone() if torch.is_tensor(self.point_ids) else torch.as_tensor(self.point_ids, dtype=torch.long),
                }
            else:
                decoded_Gmodel = {
                    k: v for k, v in Gmodel.items()
                    if k in ['_xyz', '_cholesky', '_features_dc','p_xyz', 'p_cholesky', 'p_features_dc']
                }
                decoded_Gmodel['point_ids'] = self.point_ids.detach().clone() if torch.is_tensor(self.point_ids) else torch.as_tensor(self.point_ids, dtype=torch.long)
                distribution_model = decoded_Gmodel

        return psnr_value, ms_ssim_value, fps, img, decoded_Gmodel, self.num_points, distribution_model, bpp, bpp_xy, bpp_cholesky, bpp_color
        
    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model.forward()
        out_image = out["render"]  
        mse_loss = F.mse_loss(out_image.float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out_image.float(), self.gt_image.float(), data_range=1, size_average=True).item()
        transform = transforms.ToPILImage()
        img = transform(out_image.float().squeeze(0))
        if self.save_imgs:
            save_path_img = self.log_dir / "img"
            save_path_img.mkdir(parents=True, exist_ok=True)
            name =str(self.frame_num) + "_fitting_quantized.png" 
            img.save(str(save_path_img / name))
        return psnr, ms_ssim_value, img


def image_to_tensor(img: Image.Image):
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return img_tensor




def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='Beauty', help="Training dataset"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianVideo"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to a checkpoint"
    )
    parser.add_argument(
        "--savdir", type=str, default="result", help="Path to save results"
    )
    parser.add_argument(
        "--fps", type=int, default=24, help="number of frames per second (default: %(default)s)"
    )
    parser.add_argument(
        "--image_length", type=int, default=50, help="number of input frames (default: %(default)s)"
    )
    parser.add_argument(
        "--width", type=int, default=1920, help="width (default: %(default)s)"
    )
    parser.add_argument(
        "--height", type=int, default=1080, help="height (default: %(default)s)"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--seed", type=float, default=1, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--save_imgs", action="store_true", help="Save image"
    )
    parser.add_argument(
        "--save_param_dist",
        action="store_true",
        help="Save parameter distribution plots"
    )
    args = parser.parse_args(argv)
    # Kept for log_run_config(); underscore keeps it out of the argument dump.
    args._parser = parser
    return args

def main(argv):
    args = parse_args(argv)
    savdir=args.savdir
    width = args.width
    height = args.height
    img_list = []
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
    psnrs, ms_ssims, eval_fpses = [], [], []
    image_h, image_w = 0, 0
    video_frames = process_yuv_video_frame_start_end(args.dataset, width, height, 0, args.image_length)
    frame_length,start=len(video_frames),0
    if args.image_length <= frame_length:
        image_length=args.image_length
    else:
        image_length=frame_length

    
    quantized_gmodels_state_dict = torch.load(args.model_path, map_location=torch.device("cuda:0"))
    # Build K-frame list from encoded state dict
    def _parse_frame_idx(key: str):
        try:
            return int(str(key).split('_')[-1])
        except Exception:
            return None
    K_frames = []
    for k, v in quantized_gmodels_state_dict.items():
        idx = _parse_frame_idx(k)
        if idx is None:
            continue
        is_k = False
        if isinstance(v, dict) and ('is_k_frame' in v):
            val = v['is_k_frame']
            try:
                is_k = bool(val.item()) if hasattr(val, 'item') else bool(val)
            except Exception:
                is_k = bool(val)
        if is_k:
            K_frames.append(idx)
    K_frames = sorted([k for k in K_frames if 1 <= k <= image_length])
    print(f"K_frames(from model): {K_frames}")
    Gmodel=None
    P_Gomdel=None
    has_log_writter=False
    Number_of_Gaussians=0
    for i in range(start, start+image_length):
        frame_num=i+1
        modelid=f"frame_{i + 1}"
        Model = quantized_gmodels_state_dict[modelid]
        if frame_num in K_frames:
            # if frame_num ==1:
            k_frame_num=frame_num
            #if_render_previous=False
            print(f"modelid:frame_{i + 1};")
            render = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,
                model_name=args.model_name, args=args, quantized_model=Model)
        else:
            #if_render_previous=True
            p_modelid = f"frame_{i}"
            if (P_Gomdel is not None) and ('point_ids' in quantized_gmodels_state_dict[p_modelid]):
                P_Gomdel = dict(P_Gomdel)
                P_Gomdel['point_ids'] = quantized_gmodels_state_dict[p_modelid]['point_ids']
            print(f"modelid:{modelid}; k_modelid:{p_modelid}")
            render = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,
                model_name=args.model_name, args=args, previous_quantized_model =P_Gomdel, quantized_model=Model)

        psnr, ms_ssim, eval_fps, img, Gmodel, number_Gaussinas, distribution_model, bpp, bpp_xy, bpp_cholesky, bpp_color = render.render()
        if not has_log_writter:
            has_log_writter=True
            Number_of_Gaussians=number_Gaussinas
            logwriter = LogWriter(Path(f"./checkpoints_quant/{savdir}/{args.data_name}/{args.model_name}_{Number_of_Gaussians}"))
            save_root = Path(f"./checkpoints_quant/{savdir}/{args.data_name}/{args.model_name}_{Number_of_Gaussians}")
            log_run_config(logwriter, args, argv=argv, parser=getattr(args, "_parser", None),
                           stage="stage 3: decoding (test_GSVC_decoder.py)",
                           extra={"K_frames": K_frames,
                                  "Number_of_Gaussians (frame 1)": Number_of_Gaussians})

            

        if args.save_param_dist:
            is_p_frame = frame_num not in K_frames
            save_parameter_distribution_plots(
                distribution_model,
                frame_num,
                save_root,
                is_p_frame=is_p_frame,
            )
            
        # if frame_num in K_frames:
        P_Gomdel=Gmodel

        img_list.append(img)
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        eval_fpses.append(eval_fps)
        image_h += render.H
        image_w += render.W
        torch.cuda.empty_cache()
        if i==0 or (i+1)%1==0:
            logwriter.write("Frame_{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f},Bpp:{:.4f}, Bpp_xy:{:.4f}, Bpp_cholesky:{:.4f}, Bpp_color:{:.4f}, FPS:{:.4f}, Gaussians:{}".format(
            frame_num, render.H, render.W, psnr, ms_ssim, bpp, bpp_xy, bpp_cholesky, bpp_color, eval_fps, number_Gaussinas))
    
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length
    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, FPS:{:.4f}".format(
        avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_eval_fps))    
    generate_video_decoder(savdir, img_list, args.data_name, args.model_name, args.fps, Number_of_Gaussians)
if __name__ == "__main__":
    
    main(sys.argv[1:])


