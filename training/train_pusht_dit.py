import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import wandb
from pathlib import Path
from types import SimpleNamespace

from policy.diffusion_dit_pusht_policy import DiffusionDiTPushTPolicy
from dataset.pusht_dataset_zip import PushTImageDataset
from model.pusht_encoder import create_pusht_encoder


class PushTTrainer:
    def __init__(self, config_dict):
        self.config = self.dict_to_namespace(config_dict)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create dataset
        self.create_datasets()
        self.best_loss = float('inf')
        
        # Create policy
        self.policy = DiffusionDiTPushTPolicy(config_dict).to(self.device)
        
        # Create optimizer
        optimizer_config = self.config.optimizer
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=float(optimizer_config.lr),
            betas=tuple(map(float, optimizer_config.betas)),
            eps=float(optimizer_config.eps),
            weight_decay=float(optimizer_config.weight_decay)
        )
        
        # Create EMA model
        ema_config = self.config.ema
        from diffusers.training_utils import EMAModel
        self.ema = EMAModel(
            parameters=self.policy.parameters(),
            decay=ema_config.power,  # 使用power作为decay
            max_value=ema_config.max_value,
            min_value=ema_config.min_value,
            update_after_step=ema_config.update_after_step,
            inv_gamma=ema_config.inv_gamma
        )
        
        # Learning rate scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.lr_scheduler = CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=int(self.config.training.num_epochs),
            eta_min=float(self.config.training.lr_final)
        )
        
        # Initialize wandb
        if self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.wandb_project,
                name=self.config.logging.run_name,
                config=config_dict
            )
    
    def dict_to_namespace(self, config_dict):
        if isinstance(config_dict, dict):
            namespace = SimpleNamespace()
            for key, value in config_dict.items():
                setattr(namespace, key, self.dict_to_namespace(value))
            return namespace
        elif isinstance(config_dict, list):
            return [self.dict_to_namespace(item) for item in config_dict]
        else:
            return config_dict
            
    def create_datasets(self):
        """Create train and validation datasets"""
        dataset_path = self.config.training.dataset_path
        
        self.train_dataset = PushTImageDataset(
            dataset_path=dataset_path,
            pred_horizon=self.config.pred_horizon,
            obs_horizon=self.config.obs_horizon,
            action_horizon=self.config.action_horizon
        )
        
        dataloader_config = self.config.dataloader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=dataloader_config.batch_size,
            num_workers=dataloader_config.num_workers,
            shuffle=dataloader_config.shuffle,
            pin_memory=dataloader_config.pin_memory,
            persistent_workers=dataloader_config.persistent_workers
        )
        
        print(f"Dataset size: {len(self.train_dataset)}")
        print(f"Number of batches: {len(self.train_dataloader)}")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.policy.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
 
        epoch_losses = []

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            loss = self.policy.compute_loss(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    self.config.training.max_grad_norm
                )
            
            self.optimizer.step()
            
            # Update EMA
            if self.ema is not None:
                self.ema.step(self.policy.parameters())
            
            # 
            epoch_losses.append(loss.item())
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.6f}',
                'best': f'{self.best_loss:.4f}'
            })
        
        # Update learning rate scheduler
        self.lr_scheduler.step()
    
        epoch_avg_loss = total_loss / num_batches
        epoch_min_loss = min(epoch_losses)
        epoch_max_loss = max(epoch_losses)
        current_lr = self.optimizer.param_groups[0]['lr']
        
        if self.config.logging.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss_avg': epoch_avg_loss,
                'train_loss_min': epoch_min_loss,
                'train_loss_max': epoch_max_loss,
                'learning_rate': current_lr,
                'best_loss': self.best_loss,
                'epoch_step': epoch
            })
            
        return epoch_avg_loss
        
    def save_checkpoint(self, epoch, avg_loss):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.logging.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': avg_loss,
            'config': self.config.__dict__
        }
        
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
            
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
            
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            torch.save(checkpoint, checkpoint_dir / f'{epoch}_{avg_loss:.4f}.pth')
            print(f"🎉 New best model saved! Epoch {epoch}, Loss: {avg_loss:.4f}")
            
            # Log best model to wandb
            if self.config.logging.use_wandb:
                wandb.log({
                    'best_epoch': epoch,
                    'best_loss': avg_loss
                })
        
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Total epochs: {self.config.training.num_epochs}")
        print(f"Batch size: {self.config.dataloader.batch_size}")
        print(f"Learning rate: {self.config.optimizer.lr}")
        
        for epoch in range(1, self.config.training.num_epochs + 1):
            avg_loss = self.train_epoch(epoch)
            
            print(f"Epoch {epoch}/{self.config.training.num_epochs} - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if epoch % self.config.logging.save_every == 0:
                self.save_checkpoint(epoch, avg_loss)
                
        # Save final checkpoint
        self.save_checkpoint(self.config.training.num_epochs, avg_loss)
        
        if self.ema is not None:
            print("Copying EMA weights to main model...")
            self.ema.copy_to(self.policy.parameters())
            
        print("Training completed!")


if __name__ == "__main__":
    # load config
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, "config/pusht_dit.yaml")
    
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Create trainer and start training
    trainer = PushTTrainer(config_dict)
    trainer.train()