from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
import torch
import torch.nn as nn
from quantize import (
    FakeQuantizationFP12,
    FakeQuantizationFP16,
    UniformQuantizer,
    UniformQuantizer_one,
    VectorQuantizer,
)

class GaussianVideo_frame(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = kwargs["device"]
        self.register_buffer('_xyz', torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self.register_buffer('_cholesky', torch.rand(self.init_num_points, 3))
        self.register_buffer('_features_dc', torch.rand(self.init_num_points, 3))

        

        self.last_size = (self.H, self.W)
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self.register_buffer('background', torch.ones(3))
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        self.quantize = kwargs["quantize"]

        if self.quantize:
            self.xyz_quantizer = FakeQuantizationFP16.apply 
            self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            self.cholesky_quantizer = UniformQuantizer_one(signed=[False,True,False], bits=6, learned=True, num_channels=3)
            

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound

    
    def forward(self):
        _opacity = torch.ones(self._xyz.shape[0], 1).to(self.device)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, _opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}
    
    

class GaussianVideo_delta(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = kwargs["device"]
        self.register_buffer('_xyz', torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self.register_buffer('_cholesky', torch.rand(self.init_num_points, 3))
        self.register_buffer('_features_dc', torch.rand(self.init_num_points, 3))


        self.last_size = (self.H, self.W)

        self.register_buffer('p_xyz', torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self.register_buffer('p_cholesky', torch.rand(self.init_num_points, 3))
        self.register_buffer('p_features_dc', torch.rand(self.init_num_points, 3))

        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self.register_buffer('background', torch.ones(3))
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        self.quantize = kwargs["quantize"]

        self.add_num_Gaussians=kwargs["add_num_Gaussians"]

        if self.quantize:
            self.xyz_quantizer_I = FakeQuantizationFP16.apply 
            self.features_dc_quantizer_I = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            self.cholesky_quantizer_I = UniformQuantizer_one(signed=[False,True, False], bits=6, learned=True, num_channels=3)

            self.xyz_quantizer = (
                FakeQuantizationFP12()
                if kwargs.get("xyz_format") == "fp12_e5m6_raw"
                else UniformQuantizer(signed=[True, True], bits=8, learned=True, num_channels=2)
            )
            self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=7, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            self.cholesky_quantizer = UniformQuantizer_one(signed=[True,True, True], bits=5, learned=True, num_channels=3)

       
    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz+self.p_xyz)
    
    @property
    def get_features(self):
        return self._features_dc+self.p_features_dc
    
    @property
    def get_cholesky_elements(self):
        # Match the encoder's operation order; float addition is not associative.
        return self._cholesky+self.cholesky_bound+self.p_cholesky

    @property
    def num_delta_Gaussians(self):
        return self.init_num_points - self.add_num_Gaussians
 
    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    
    



   
