#@markdown ### **PushT Vision Encoder**
#@markdown
#@markdown Based on MultiImageObsEncoder but simplified for PushT
#@markdown - Supports both RGB images and low-dim observations (agent position)
#@markdown - Uses ResNet backbone with GroupNorm replacement for EMA compatibility
#@markdown - Configurable resize and normalization

from typing import Dict, Tuple, Union, Callable
import copy
import torch
import torch.nn as nn
import torchvision

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


class PushTObsEncoder(nn.Module):
    """
    PushT Observation Encoder based on MultiImageObsEncoder
    Handles both RGB images and low-dimensional observations
    """
    
    def __init__(self,
                 shape_meta: dict,
                 vision_encoder: nn.Module,
                 resize_shape: Union[Tuple[int,int], None] = None,
                 use_group_norm: bool = True,
                 imagenet_norm: bool = False):
        super().__init__()
        
        # Parse shape metadata
        obs_shape_meta = shape_meta['obs']
        
        rgb_keys = []
        low_dim_keys = []
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = {}
        
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            obs_type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            
            if obs_type == 'rgb':
                rgb_keys.append(key)

                # Clone the vision encoder for this key
                this_model = copy.deepcopy(vision_encoder)

                # Replace BatchNorm with GroupNorm if requested
                if use_group_norm:
                    this_model = replace_bn_with_gn(this_model)
                
                key_model_map[key] = this_model
                
                # Configure transforms
                transforms = []
                
                # Resize if specified
                if resize_shape is not None:
                    h, w = resize_shape
                    transforms.append(torchvision.transforms.Resize(size=(h, w)))
                
                # ImageNet normalization if requested
                if imagenet_norm:
                    transforms.append(torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]))
                
                # Combine transforms
                if transforms:
                    key_transform_map[key] = nn.Sequential(*transforms)
                else:
                    key_transform_map[key] = nn.Identity()
                    
            elif obs_type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")
        
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.rgb_keys = sorted(rgb_keys)
        self.low_dim_keys = sorted(low_dim_keys)
        self.key_shape_map = key_shape_map
    
    def forward(self, obs_dict):
        """
        Forward pass through the encoder
        
        Args:
            obs_dict: Dictionary of observations
                - For RGB: [batch_size, obs_horizon, C, H, W] or [batch_size, C, H, W]
                - For low_dim: [batch_size, obs_horizon, D] or [batch_size, D]
        
        Returns:
            features: [batch_size, feature_dim] concatenated features
        """
        batch_size = None
        features = []
        
        # Process RGB observations
        for key in self.rgb_keys:
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            
            # Handle both [B, T, C, H, W] and [B, C, H, W] formats
            if len(img.shape) == 5:  # [B, T, C, H, W]
                obs_horizon = img.shape[1]
                img = img.view(-1, *img.shape[2:])  # [B*T, C, H, W]
            else:  # [B, C, H, W]
                obs_horizon = 1
            
            # Apply transforms
            img = self.key_transform_map[key](img)
            
            # Extract features
            feature = self.key_model_map[key](img)
            
            # Reshape back if needed
            if obs_horizon > 1:
                feature = feature.view(batch_size, obs_horizon, -1)
                feature = feature.flatten(start_dim=1)  # [B, T*D]
            
            features.append(feature)
        
        # Process low-dimensional observations
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            
            # Handle both [B, T, D] and [B, D] formats
            if len(data.shape) == 3:  # [B, T, D]
                data = data.flatten(start_dim=1)  # [B, T*D]
            
            features.append(data)
        
        # Concatenate all features
        if features:
            result = torch.cat(features, dim=-1)
        else:
            # If no features, return zero tensor
            result = torch.zeros(batch_size, 1, device=next(self.parameters()).device)
        
        return result
    
    @torch.no_grad()
    def output_shape(self):
        """Compute output shape by running a forward pass with dummy data"""
        example_obs_dict = {}
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=torch.float32,
                device=next(self.parameters()).device)
            example_obs_dict[key] = this_obs
            
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape


def create_pusht_encoder(obs_horizon: int = 2, backbone: str = "resnet18", shape_meta: dict = None) -> PushTObsEncoder:
    """
    Create a PushT encoder with standard configuration
    
    Args:
        obs_horizon: Number of observation frames
        backbone: ResNet backbone to use
    
    Returns:
        Configured PushTObsEncoder
    """
    # Define shape metadata for PushT
    if shape_meta is None:
        shape_meta = {
            'obs': {
                'image': {
                    'shape': [obs_horizon, 3, 96, 96],
                    'type': 'rgb'
                },
                'agent_pos': {
                    'shape': [obs_horizon, 2],
                    'type': 'low_dim'
                }
            }
        }
    
    # Create RGB model (ResNet backbone)
    vision_encoder = get_resnet(backbone)
    
    # Create encoder
    encoder = PushTObsEncoder(
        shape_meta=shape_meta,
        vision_encoder=vision_encoder,
        resize_shape=None,  # Keep original 96x96 size
        use_group_norm=True,  # Important for EMA!
        imagenet_norm=False   # PushT images are already normalized
    )
    
    return encoder


if __name__ == "__main__":
    # Test the new encoder
    print("Testing PushTObsEncoder...")
    
    obs_horizon = 2
    encoder = create_pusht_encoder(obs_horizon=obs_horizon)
    
    # Create dummy observation
    dummy_obs = {
        'image': torch.randn(1, obs_horizon, 3, 96, 96),  # [B, T, C, H, W]
        'agent_pos': torch.randn(1, obs_horizon, 2)       # [B, T, D]
    }
    
    # Forward pass
    with torch.no_grad():
        features = encoder(dummy_obs)
        output_shape = encoder.output_shape()
    
    print(f"Input image shape: {dummy_obs['image'].shape}")
    print(f"Input agent_pos shape: {dummy_obs['agent_pos'].shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Output shape (computed): {output_shape}")
    
    # Test with different input formats
    print("\nTesting with single frame input...")
    single_frame_obs = {
        'image': torch.randn(1, 3, 96, 96),     # [B, C, H, W]
        'agent_pos': torch.randn(1, 2)          # [B, D]
    }
    
    with torch.no_grad():
        features_single = encoder(single_frame_obs)
    
    print(f"Single frame output shape: {features_single.shape}")
 