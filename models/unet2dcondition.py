from diffusers import DDPMScheduler, UNet2DConditionModel, UNet2DModel
import torch
from tqdm.auto import tqdm
from torch import nn

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


class UNet2DCondition(nn.Module):
    def __init__(self, ci_dim=120, model_choice=0, timesteps=1000, encoder_hid_dim=0):
        super().__init__()

        self.scheduler = DDPMScheduler(num_train_timesteps=timesteps, beta_schedule="squaredcos_cap_v2")

        self.encoder_cis = False
        if encoder_hid_dim > 0:
            self.encoder_hid_dim = encoder_hid_dim
            self.encoder_cis = True
        else:
            self.encoder_hid_dim = ci_dim

        self.model_choice = model_choice

        if model_choice == 'unet2d':
            self.model = UNet2DModel(
                sample_size=64,  # the target image resolution
                in_channels=1,  # the number of input channels
                out_channels=1,  # the number of output channels
                block_out_channels=(32, 64, 64),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D", 
                    "AttnDownBlock2D", 
                ),
                up_block_types=(
                    "AttnUpBlock2D",  
                    "UpBlock2D",  
                    "UpBlock2D",
                ),
                # num_class_embeds = 512,
                # class_embed_type = 'identity',
            )

        # Self.model is an conditional UNet
        elif model_choice == 0:
            self.model = UNet2DConditionModel(
                sample_size=16,  # the target image resolution
                in_channels=4, # Additional input channels for class cond.
                out_channels=4,  # the number of output channels
                layers_per_block=2,  # how many ResNet layers to use per UNet block
                block_out_channels=(32, 64, 64),
                down_block_types=(
                    "CrossAttnDownBlock2D",  # a ResNet downsampling block with cross-attention
                    "CrossAttnDownBlock2D", 
                    "DownBlock2D",  # a regular ResNet downsampling block
                ),
                up_block_types=(
                    "UpBlock2D",  # a regular ResNet upsampling block
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",  # a ResNet upsampling block with cross-attention
                ),
                encoder_hid_dim=self.encoder_hid_dim,
                time_cond_proj_dim=512,
            )
        elif model_choice == 1:
            self.model = UNet2DConditionModel(
                sample_size=16,  # the target image resolution
                in_channels=4, # Additional input channels for class cond.
                out_channels=4,  # the number of output channels
                layers_per_block=2,  # how many ResNet layers to use per UNet block
                block_out_channels=(32, 64, 64),
                down_block_types=(
                    "DownBlock2D",
                    "CrossAttnDownBlock2D",  # a ResNet downsampling block with cross-attention
                    "CrossAttnDownBlock2D", 
                ),
                up_block_types=(
                    "AttnUpBlock2D",  
                    "AttnUpBlock2D",
                    "UpBlock2D",  # a regular ResNet upsampling block
                ),
                encoder_hid_dim=self.encoder_hid_dim,
                time_cond_proj_dim=512,
            )
        elif model_choice == 2:
            self.model = UNet2DConditionModel(
                sample_size=16,  # the target image resolution
                in_channels=4, # Additional input channels for class cond.
                out_channels=4,  # the number of output channels
                layers_per_block=2,  # how many ResNet layers to use per UNet block
                block_out_channels=(32, 64, 64),
                down_block_types=(
                    "DownBlock2D",
                    "AttnDownBlock2D",  # a ResNet downsampling block with cross-attention
                    "CrossAttnDownBlock2D", 
                ),
                up_block_types=(
                    "CrossAttnUpBlock2D",  
                    "AttnUpBlock2D",
                    "UpBlock2D",  # a regular ResNet upsampling block
                ),
                encoder_hid_dim=self.encoder_hid_dim,
                time_cond_proj_dim=512,
            )
        elif model_choice == 3:
            self.model = UNet2DConditionModel(
                sample_size=16,  # the target image resolution
                in_channels=4, # Additional input channels for class cond.
                out_channels=4,  # the number of output channels
                layers_per_block=2,  # how many ResNet layers to use per UNet block
                block_out_channels=(32, 64, 64),
                down_block_types=(
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",  # a ResNet downsampling block with cross-attention
                    "CrossAttnDownBlock2D", 
                ),
                up_block_types=(
                    "CrossAttnUpBlock2D",  
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",  # a regular ResNet upsampling block
                ),
                encoder_hid_dim=self.encoder_hid_dim,
                time_cond_proj_dim=512,
            )
        elif model_choice == 4:
            self.model = UNet2DConditionModel(
                sample_size=16,  # the target image resolution
                in_channels=4, # Additional input channels for class cond.
                out_channels=4,  # the number of output channels
                layers_per_block=2,  # how many ResNet layers to use per UNet block
                block_out_channels=(32, 64, 64),
                down_block_types=(
                    "DownBlock2D",  
                    "CrossAttnDownBlock2D", 
                    "CrossAttnDownBlock2D",  
                ),
                up_block_types=(
                    "CrossAttnUpBlock2D",  
                    "CrossAttnUpBlock2D",
                    "UpBlock2D",  
                ),
                encoder_hid_dim=self.encoder_hid_dim,
                time_cond_proj_dim=512,
            )
        else:
            self.model = UNet2DConditionModel(
                sample_size=16,  # the target image resolution
                in_channels=4, # Additional input channels for class cond.
                out_channels=4,  # the number of output channels
                layers_per_block=2,  # how many ResNet layers to use per UNet block
                block_out_channels=(32, 64, 64),
                down_block_types=(
                    "DownBlock2D", 
                    "AttnDownBlock2D", 
                    "AttnDownBlock2D", 
                ),
                up_block_types=(
                    "AttnUpBlock2D",  
                    "AttnUpBlock2D",
                    "UpBlock2D",  
                ),
                encoder_hid_dim=self.encoder_hid_dim,
                time_cond_proj_dim=512,
            )
        if encoder_hid_dim > 0:
            self.ci_encoder = nn.Sequential(
                nn.Linear(ci_dim, ci_dim),
                nn.LeakyReLU(),
                nn.Linear(ci_dim, ci_dim),
                nn.LeakyReLU(),
                nn.Linear(ci_dim, encoder_hid_dim),
                nn.LeakyReLU(),
                nn.Linear(encoder_hid_dim, encoder_hid_dim),
                nn.LeakyReLU(),
            )



        

    # forward method
    def forward(self, x, t, ci, class_labels=None, guidance_scale=None):
        # Shape of x:
        bs, ch, w, h = x.shape

        if self.model_choice == 'unet2d':
            return self.model(x, t).sample
        else:
            if guidance_scale is not None:
                guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(bs)
                timestep_cond = self.get_guidance_scale_embedding(guidance_scale_tensor, embedding_dim=self.model.config.time_cond_proj_dim).to(device=device, dtype=x.dtype)
            else:
                timestep_cond = None

            if self.encoder_cis:
                ci = self.ci_encoder(ci)

            if class_labels != None:
                return self.model(x, t, ci, class_labels, timestep_cond=timestep_cond).sample
            else:
                return self.model(x,t,ci, timestep_cond=timestep_cond).sample

    def get_guidance_scale_embedding(self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb
    
    def runUnet(self, x, ci, init_t=0, class_labels=None, guidance_scale=None, return_allx = False):
        # Sampling loop
        allx = []
        for i, t in tqdm(enumerate(self.scheduler.timesteps[init_t:])):

            # Get model pred
            with torch.no_grad():
                residual = self.forward(x, t, ci, class_labels, guidance_scale=guidance_scale)

            # Update sample with step
            x = self.scheduler.step(residual, t, x).prev_sample

            if return_allx:
                allx.append(x)

        if return_allx:
            return torch.stack(allx)
        
        # swap axes back
        x = torch.swapaxes(x, -1, -2)

        return x