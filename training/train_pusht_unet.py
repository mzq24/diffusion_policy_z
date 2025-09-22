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

from policy.diffusion_unet_pusht_policy import DiffusionUnetPushTPolicy, create_pusht_policy_config
from dataset.pusht_dataset_zip import PushTImageDataset
from model.pusht_encoder import create_pusht_encoder


class PushTTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create dataset
        self.create_datasets()
        self.best_loss = float('inf')
        
        # Create policy
        self.policy = DiffusionUnetPushTPolicy(config.policy).to(self.device)
        
        # Set action normalizer
        # self.policy.set_normalizer(self.train_dataset.stats['action'])
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config.training.learning_rate,
            betas=config.training.betas,
            weight_decay=config.training.weight_decay
        )
        
        # Create EMA model
        if config.training.use_ema:
            from diffusers.training_utils import EMAModel
            self.ema = EMAModel(
                parameters=self.policy.parameters(),
                decay=config.training.ema_decay
            )
        else:
            self.ema = None
            
        # Learning rate scheduler - 修改为cosine调度到0
        if config.training.use_lr_scheduler:
            if config.training.lr_scheduler_type == "cosine":
                # 使用CosineAnnealingLR，在1000个epoch后学习率衰减到0
                from torch.optim.lr_scheduler import CosineAnnealingLR
                self.lr_scheduler = CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=config.training.num_epochs,  # 1000 epochs
                    eta_min=getattr(config.training, 'lr_final', 0.0)  # 最终学习率为0
                )
                self.lr_scheduler_type = "epoch"  # 每个epoch更新一次
            else:
                # 使用diffusers的scheduler（每个batch更新）
                from diffusers.optimization import get_scheduler
                self.lr_scheduler = get_scheduler(
                    config.training.lr_scheduler_type,
                    optimizer=self.optimizer,
                    num_warmup_steps=config.training.lr_warmup_steps,
                    num_training_steps=config.training.num_epochs * len(self.train_dataloader)
                )
                self.lr_scheduler_type = "batch"  # 每个batch更新一次
        else:
            self.lr_scheduler = None
            
        # Initialize wandb
        if config.logging.use_wandb:
            wandb.init(
                project=config.logging.wandb_project,
                name=config.logging.run_name,
                config=config.__dict__
            )
            
    def create_datasets(self):
        """Create train and validation datasets"""
        self.train_dataset = PushTImageDataset(
            dataset_path=self.config.data.dataset_path,
            pred_horizon=self.config.policy.pred_horizon,
            obs_horizon=self.config.policy.obs_horizon,
            action_horizon=self.config.policy.action_horizon
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.config.training.num_workers > 0 else False
        )
        
        print(f"Dataset size: {len(self.train_dataset)}")
        print(f"Number of batches: {len(self.train_dataloader)}")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.policy.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            loss = self.policy.compute_loss(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.training.max_grad_norm)
            self.optimizer.step()
            
            # Update learning rate scheduler 
            if self.lr_scheduler is not None and getattr(self, 'lr_scheduler_type', 'batch') == 'batch':
                self.lr_scheduler.step()
            
            # Update EMA
            if self.ema is not None:
                self.ema.step(self.policy.parameters())
                
            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.6f}',
                'best': f'{self.best_loss:.4f}'
            })
            
            # Log to wandb
            if self.config.logging.use_wandb:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': current_lr,
                    'epoch': epoch,
                    'step': epoch * num_batches + batch_idx,
                    'best_loss': self.best_loss
                })
        
        # Update learning rate scheduler 
        if self.lr_scheduler is not None and getattr(self, 'lr_scheduler_type', 'batch') == 'epoch':
            self.lr_scheduler.step()
            
        return total_loss / num_batches
        
    
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
            
        # Save latest checkpoint
        #torch.save(checkpoint, checkpoint_dir / 'latest.pth')
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
        # Save periodic checkpoint
        #if epoch % self.config.logging.save_every == 0:
        #    torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch:04d}.pth')
            
        #print(f"Checkpoint saved at epoch {epoch}")
        
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Total epochs: {self.config.training.num_epochs}")
        print(f"Batch size: {self.config.training.batch_size}")
        print(f"Learning rate: {self.config.training.learning_rate}")
        
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


def create_training_config():
    # Policy config 
    policy_config = SimpleNamespace(
        action_dim=2,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        global_cond_dim=1028,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        num_diffusion_steps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        imagenet_norm=False,
    )
    
    # Data config
    data_config = SimpleNamespace(
        dataset_path="/media/z/data/mzq/code/diffusion_policy_z/data/pusht/pusht_cchi_v7_replay.zarr"
    )

    # Training config
    training_config = SimpleNamespace(
        num_epochs=300,  
        batch_size=64,
        learning_rate=1e-4,  
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        max_grad_norm=1.0,
        num_workers=4,

        # EMA config
        use_ema=True,
        ema_decay=0.75,  
    
        # Learning rate scheduler
        use_lr_scheduler=True,
        lr_scheduler_type="cosine",
        lr_warmup_steps=0,  
        lr_final=1e-7,  
    )
    
    # Logging config
    logging_config = SimpleNamespace(
        use_wandb=True,
        wandb_project="diffusion_policy_pusht",
        run_name="pusht_diffusion_unet_cosine_1000ep",
        checkpoint_dir="/media/z/data/mzq/code/diffusion_policy_z/checkpoints/pusht",
        save_every=10,  
    )
    
    # Main config
    config = SimpleNamespace(
        policy=policy_config,
        data=data_config,
        training=training_config,
        logging=logging_config
    )
        
    return config



if __name__ == "__main__":
    # Create configuration
    config = create_training_config()
    
    # Create trainer and start training
    trainer = PushTTrainer(config)
    trainer.train()