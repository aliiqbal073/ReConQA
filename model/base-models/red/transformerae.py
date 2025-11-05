import torch
import torch.nn as nn

class TransformerAE(nn.Module):
    """
    Lightweight Transformer-based Autoencoder for feature-space reconstruction.
    Input: feature map of shape (B, C, H, W)
    Output: reconstructed feature map of same shape
    """
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:

        B, C, H, W = feat.shape       
        r = 8
        H_ds, W_ds = max(1, H // r), max(1, W // r)
        feat_ds = nn.functional.interpolate(feat, size=(H_ds, W_ds),
                                            mode="bilinear", align_corners=False)
 
        src = feat_ds.flatten(2).permute(2, 0, 1)
    
        mem = self.encoder(src)
        out = self.decoder(src, mem)
      
        recon_ds = out.permute(1, 2, 0).view(B, C, H_ds, W_ds)
      
        recon = nn.functional.interpolate(recon_ds, size=(H, W),
                                          mode="bilinear", align_corners=False)
        return recon
