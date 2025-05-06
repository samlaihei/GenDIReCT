from torch import nn
from models.encoder import Encoder
from models.decoder import Decoder


class AE(nn.Module):
    def __init__(self, latent_size:int=4, n_res_layers=2, res_h_dim=8):
        super(AE, self).__init__()

        self.latent_size = latent_size

        # Encoder # input img, output n latent features
        self.encoder = Encoder(1, self.latent_size, n_res_layers=n_res_layers, res_h_dim=res_h_dim)

        # Decoder # input n, output img
        self.decoder = Decoder(self.latent_size, self.latent_size, n_res_layers=n_res_layers, res_h_dim=res_h_dim)

    
    def forward(self, imgs): 
        features = self.encoder(imgs)
        recon_img = self.decoder(features)
        return features, recon_img
    
    def encoder_to_img(self, imgs):
        features = self.encoder(imgs)
        recon_img = self.decoder(features)
        return features, None, recon_img
