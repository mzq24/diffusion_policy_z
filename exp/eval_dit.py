import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm
import time
import sys
import collections
sys.path.append("..")

from dataset.pusht_dataset_zip import PushTImageDataset
from pusht_dataset_zip import PushTImageDataset as PushTImageDataset_ori
from pusht_env import PushTEnv
from policy.diffusion_dit_pusht_policy import DiffusionDiTPushTPolicy  # 修正：使用正确的类名
from pusht_env import PushTImageEnv
from video_recorder import VideoRecorder

def namespace_to_dict(namespace_obj):
    if hasattr(namespace_obj, '__dict__'):
        result = {}
        for key, value in namespace_obj.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = namespace_to_dict(value)
            else:
                result[key] = value
        return result
    else:
        return namespace_obj
    
def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def load_policy_from_checkpoint(checkpoint_path, device='cuda'):
    print(f"Loading checkpoint from: {checkpoint_path}")
    dataset_path = "/media/z/data/mzq/code/diffusion_policy_z/data/pusht/pusht_cchi_v7_replay.zarr"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint['config']
    d = {}
    for k, v in config_dict.items():
        d[k] = namespace_to_dict(v)
    # full_config = d
    config_dict = d
    policy = DiffusionDiTPushTPolicy(config_dict).to(device)  

    if 'ema_state_dict' in checkpoint:
        from diffusers.training_utils import EMAModel
        ema = EMAModel(policy.parameters())
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.copy_to(policy.parameters())
        print("Loaded EMA weights")
    else:
        policy.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded regular weights")
    
    policy.eval()
    
    # dataset for stats
    dataset = PushTImageDataset_ori(
        dataset_path=dataset_path,
        pred_horizon=config_dict.get('pred_horizon', 16),
        obs_horizon=config_dict.get('obs_horizon', 2),
        action_horizon=config_dict.get('action_horizon', 8)
    )
    stats = dataset.stats
    
    # action normalization
    policy.set_normalizer(action_stats=stats['action'], pos_stats=stats['agent_pos'])
    
    print(f"Policy loaded successfully from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['loss']:.4f}")
    
    return policy, config_dict, stats

def evaluate_single_episode(policy, policy_config, stats, max_steps=200, render=False, save_video=False, seed=100000):
    """使用policy.predict_action进行推理的简化版评估"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = PushTImageEnv()
    env.seed(seed)

    obs, info = env.reset()
    
    obs_horizon = policy_config.get('obs_horizon', 2)
    pred_horizon = policy_config.get('pred_horizon', 16)
    action_horizon = policy_config.get('action_horizon', 8)
    action_dim = policy_config.get('action_dim', 2)
    
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    if save_video:
        video_recorder = VideoRecorder.create_h264(fps=10)
        video_path = f"eval_dit_episode_seed_{seed}.mp4"
        video_recorder.start(video_path)
        video_recorder.write_frame(imgs[0])
    
    with tqdm(total=max_steps, desc="Eval PushTImageEnv (DIT)") as pbar:
        while not done:
            images = np.stack([x['image'] for x in obs_deque])  # (obs_horizon, H, W, C)
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])  # (obs_horizon, 2)
         
            nimages = images.astype(np.float32)   # whether to normalize
            nagent_poses = normalize_data(agent_poses, stats['agent_pos'])

            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32) # (obs_horizon, C, H, W)
            nimages = nimages.unsqueeze(0)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            nagent_poses = nagent_poses.unsqueeze(0)    # (obs_horizon, 2) -> (1, obs_horizon, 2)
            
            obs_dict = {
                'image': nimages,      # (1, obs_horizon, C, H, W)
                'agent_pos': nagent_poses  # (1, obs_horizon, 2)
            }
            
            with torch.no_grad():
                action_pred = policy.predict_action(obs_dict)  # (1, pred_horizon, action_dim)
            
            # action_pred = action_pred.squeeze(0).cpu().numpy()  # (pred_horizon, action_dim)
            #start = obs_horizon - 1
            #end = start + action_horizon
            #action = action_pred[start:end, :]  # (action_horizon, action_dim)
            action = action_pred['action'][0]
            
            for i in range(len(action)):

                obs, reward, done, _, info = env.step(action[i])

                obs_deque.append(obs)
      
                rewards.append(reward)
                img = env.render(mode='rgb_array')
                imgs.append(img)

                if render:
                    cv2.imshow('PushT Evaluation (DIT)', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if save_video:
                    video_recorder.write_frame(img)

                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward, max_reward=max(rewards) if rewards else 0)
                
                if step_idx > max_steps:
                    done = True
                if done:
                    break
    
    if save_video:
        video_recorder.stop()
        print(f"Video saved to: {video_path}")
    
    env.close()
    if render:
        cv2.destroyAllWindows()
    
    max_score = max(rewards) if rewards else 0
    print(f'Score: {max_score:.4f}')
    
    return {
        'rewards': rewards,
        'max_score': max_score,
        'episode_length': step_idx,
        'images': imgs,
        'success': max_score > 0.95
    }

def evaluate_policy(policy, policy_config, stats, num_episodes=5, max_steps=200, render=False, save_video=False):
    print(f"start eval DIT Policy... total {num_episodes} episodes")

    all_results = []
    success_count = 0
    
    for episode in range(num_episodes):

        seed = 100000 + episode * 1  
        print(f"\nEpisode {episode + 1}/{num_episodes} (seed={seed})")
        
        result = evaluate_single_episode(
            policy=policy,
            policy_config=policy_config,
            stats=stats,
            max_steps=max_steps,
            render=render,
            save_video=save_video,
            seed=seed
        )
        
        all_results.append(result)
        if result['success']:
            success_count += 1
        
        print(f"Episode {episode + 1}: Score={result['max_score']:.4f}, Length={result['episode_length']}, Success={result['success']}")
    
    # statistics
    max_scores = [r['max_score'] for r in all_results]
    episode_lengths = [r['episode_length'] for r in all_results]
    
    mean_score = np.mean(max_scores)
    std_score = np.std(max_scores)
    mean_length = np.mean(episode_lengths)
    success_rate = success_count / num_episodes
    
    print("\n" + "="*50)
    print("DIT Policy evaluate result:")
    print(f"avg score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"avg episode length: {mean_length:.1f}")
    print(f"success rate: {success_rate:.1%} ({success_count}/{num_episodes})")
    print(f"score range: {min(max_scores):.4f} - {max(max_scores):.4f}")
    print("="*50)
    
    return {
        'all_results': all_results,
        'mean_score': mean_score,
        'std_score': std_score,
        'mean_length': mean_length,
        'success_rate': success_rate,
        'max_scores': max_scores,
        'episode_lengths': episode_lengths
    }

def main():
    import os
    checkpoint_path = "/media/z/data/mzq/code/diffusion_policy_z/checkpoints/pusht_dit/300_0.0151.pth"
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, "config/pusht_dit.yaml")
    with open(config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"device: {device}")
    print("eval DIT Policy")
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("可用的checkpoint文件:")
        checkpoint_dir = Path(checkpoint_path).parent
        if checkpoint_dir.exists():
            for f in checkpoint_dir.glob("*.pth"):
                print(f"  - {f}")
        return
    
    # load policy
    policy, policy_config, stats = load_policy_from_checkpoint(checkpoint_path, device)
    print("DIT Policy load success")
    
    # eval policy
    results = evaluate_policy(
        policy=policy,
        policy_config=policy_config,
        stats=stats,
        num_episodes=50,   
        max_steps=1000,    
        render=False,     
        save_video=False
    )

if __name__ == "__main__":
    main()