# This file is originally developed by https://github.com/mohsenh17 in:
# https://github.com/mohsenh17/aminoClust/blob/main/models/amino_clust.py
# It is reused and further developed as part of this project.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import yaml
import numpy as np
from typing import Tuple, Optional
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Safely load configuration with error handling."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using default configuration.")
        return get_default_config()
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        return get_default_config()

def get_default_config() -> dict:
    """Default configuration for VQ-VAE."""
    return {
        'model': {
            'input_dim': 100,
            'latent_dim': 64,
            'num_embeddings': 512,
            'commitment_cost': 0.25,
            'hidden_dims': [256, 128]
        },
        'training': {
            'learning_rate': 1e-3,
            'batch_size': 32,
            'num_epochs': 100
        }
    }

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        """
        Args:
            num_embeddings (int): Number of vectors in the codebook.
            embedding_dim (int): Dimensionality of each codebook vector.
            commitment_cost (float): Weighting factor for commitment loss.
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialization using Xavier initialization
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)
        
        # Track codebook usage for better analysis
        self.register_buffer('cluster_usage', torch.zeros(num_embeddings))
        self.register_buffer('_need_init', torch.tensor(True))

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e (Tensor): Encoder output, shape (batch_size, embedding_dim)
        Returns:
            z_q (Tensor): Quantized output, same shape as z_e.
            loss (Tensor): The VQ loss including commitment loss.
            encoding_indices (Tensor): The codebook index for each latent vector.
        """

        input_shape = z_e.shape
        flat_input = z_e.view(-1, self.embedding_dim)
        
        distances = torch.cdist(flat_input, self.embeddings.weight, p=2)
        
        encoding_indices = torch.argmin(distances, dim=-1, keepdim=True)
        
        if self.training:
            unique, counts = torch.unique(encoding_indices, return_counts=True)
            self.cluster_usage.zero_()
            self.cluster_usage.index_add_(0, unique.flatten(), counts.float())
        
        z_q = self.embeddings(encoding_indices).view(input_shape)
        
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        z_q = z_e + (z_q - z_e).detach()
        
        return z_q, vq_loss, encoding_indices.view(input_shape[:-1])
    
    def get_codebook_usage(self) -> torch.Tensor:
        """Get codebook usage statistics."""
        return self.cluster_usage / (self.cluster_usage.sum() + 1e-8)
    
    def get_perplexity(self) -> torch.Tensor:
        """Calculate codebook perplexity."""
        usage = self.get_codebook_usage()
        perplexity = torch.exp(-torch.sum(usage * torch.log(usage + 1e-8)))
        return perplexity

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Optional[list] = None):
        """
        Args:
            input_dim (int): Dimensionality of input data.
            latent_dim (int): Dimensionality of the latent representation.
            hidden_dims (list, optional): Hidden layer dimensions.
        """
        super(Encoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        layers = []
        prev_dim = input_dim
        
        # Build encoder layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final layer to latent dimension
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input data, shape (batch_size, input_dim)
        Returns:
            Tensor: Latent representation of shape (batch_size, latent_dim)
        """
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: Optional[list] = None):
        """
        Args:
            latent_dim (int): Dimensionality of the latent representation.
            output_dim (int): Dimensionality of the reconstructed output.
            hidden_dims (list, optional): Hidden layer dimensions (reversed from encoder).
        """
        super(Decoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256]
        
        layers = []
        prev_dim = latent_dim
        
        # Build decoder layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final layer to output dimension
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Latent code, shape (batch_size, latent_dim)
        Returns:
            Tensor: Reconstructed input, shape (batch_size, output_dim)
        """
        return self.decoder(x)

class VQVAE(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 latent_dim: int, 
                 num_embeddings: int,
                 commitment_cost: float = 0.25,
                 hidden_dims: Optional[list] = None):
        """
        Args:
            input_dim (int): Dimensionality of the input.
            latent_dim (int): Dimensionality of the latent space.
            num_embeddings (int): Number of discrete latent codes in the codebook.
            commitment_cost (float): Weighting factor for commitment loss.
            hidden_dims (list, optional): Hidden layer dimensions.
        """
        super(VQVAE, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims)
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dims[::-1])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x (Tensor): Input data, shape (batch_size, input_dim)
        Returns:
            x_recon (Tensor): Reconstructed input.
            vq_loss (Tensor): Vector quantization loss.
            encoding_indices (Tensor): Discrete code indices representing clusters.
            metrics (dict): Additional metrics for monitoring.
        """
        # Validate input
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[-1]}")
        
        z_e = self.encoder(x)
        z_q, vq_loss, encoding_indices = self.vq(z_e)
        x_recon = self.decoder(z_q)
        
        metrics = {
            'perplexity': self.vq.get_perplexity(),
            'codebook_usage': self.vq.get_codebook_usage().mean(),
            'reconstruction_error': F.mse_loss(x_recon, x),
        }
        
        return x_recon, vq_loss, encoding_indices, metrics
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        self.eval()
        with torch.no_grad():
            z_e = self.encoder(x)
            z_q, _, encoding_indices = self.vq(z_e)
            return z_q
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        self.eval()
        with torch.no_grad():
            return self.decoder(z_q)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given input data, output the discrete cluster assignments (codebook indices).
        
        Args:
            x (Tensor): Input data.
        Returns:
            encoding_indices (Tensor): Discrete code indices.
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            x = x.to(device)
            z_e = self.encoder(x)
            _, _, encoding_indices = self.vq(z_e)
        return encoding_indices
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input data."""
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            x = x.to(device)
            x_recon, _, _, _ = self.forward(x)
        return x_recon
    
    def get_codebook_embeddings(self) -> torch.Tensor:
        """Get the learned codebook embeddings."""
        return self.vq.embeddings.weight.data
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'num_embeddings': self.num_embeddings,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }

class VQVAETrainer:
    """Training utility class for VQ-VAE."""
    
    def __init__(self, model: VQVAE, config: dict):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.get('learning_rate', 1e-3)
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
        
    def train_step(self, batch: torch.Tensor) -> dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        x_recon, vq_loss, _, metrics = self.model(batch)
        recon_loss = F.mse_loss(x_recon, batch)
        
        total_loss = recon_loss + vq_loss
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'vq_loss': vq_loss.item(),
            **{k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import torch
    import pandas as pd
    
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    
    from config.load_config import load_config
    from data.data_loader import load_raw_data
    from data.data_preprocess import classify_capacity_bins, prepare_features_for_modeling
    from data.feature_engineer import process_impedance_features


    def train_vqvae(
        trainer,
        data_df: pd.DataFrame,
        batch_size: int = 32,
        epochs: int = 10,
        device: torch.device = torch.device('cuda')
    ):
        from torch.utils.data import DataLoader, TensorDataset
        
        data_tensor = torch.tensor(data_df.values, dtype=torch.float32).to(device)
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_losses = []
            for batch_tuple in dataloader:
                batch = batch_tuple[0].to(device)
                losses = trainer.train_step(batch)
                epoch_losses.append(losses['total_loss'])
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_config(config_path)
    else:
        config = load_config()
    
    # Load and preprocess data
    data_path = config['paths']['dataset']
    raw_data = load_raw_data(data_path)
    
    # Feature engineering pipeline
    processed_data = classify_capacity_bins(raw_data)
    processed_data = process_impedance_features(processed_data)
    
    # Prepare features for modeling (train/val/test split optional)
    test_size = config['training']['test_size']
    val_size = config['training']['val_size']
    random_state = config['training']['random_state']
    
    prepared_data = prepare_features_for_modeling(
        processed_data, val_size=val_size, test_size=test_size, random_state=random_state
    )
    print(prepared_data['X_train_scaled'].shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize VQVAE model (update args as your model requires)
    model = VQVAE(input_dim=prepared_data['X_train_scaled'].shape[1],
                  latent_dim=config['training']['latent_dim'],
                  num_embeddings=config['training']['num_embeddings'],
                  commitment_cost=config['training']['commitment_cost'],
                  hidden_dims=config['training']['hidden_dims'])
    model.to(device)
    
    # Initialize trainer
    trainer = VQVAETrainer(model, config['training'])
    
    # Train VQVAE on training data
    train_vqvae(
        trainer,
        pd.DataFrame(prepared_data['X_train_scaled'], columns=prepared_data['feature_cols']),
        batch_size=config['training'].get('batch_size', 32),
        epochs=config['training'].get('epochs', 10),
        device=device
    )
