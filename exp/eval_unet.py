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
from policy.diffusion_unet_pusht_policy import DiffusionUnetPushTPolicy
from pusht_env import PushTImageEnv
from video_recorder import VideoRecorder


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
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint['config']
    
    # 重建policy配置
    from types import SimpleNamespace
    policy_config = config_dict['policy']
    
    # 创建policy
    policy = DiffusionUnetPushTPolicy(policy_config).to(device)

    # 加载权重
    if 'ema_state_dict' in checkpoint:
        from diffusers.training_utils import EMAModel
        ema = EMAModel(policy.parameters())
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.copy_to(policy.parameters())
        print("Loaded EMA weights")
    else:
        policy.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded regular weights")
    
    # 设置为评估模式
    policy.eval()
    
    # 加载数据集统计信息
    dataset = PushTImageDataset_ori(
        dataset_path=dataset_path,
        pred_horizon=policy_config.pred_horizon,
        obs_horizon=policy_config.obs_horizon,
        action_horizon=policy_config.action_horizon
    )
    stats = dataset.stats
    
    # 设置action归一化
    policy.set_normalizer(action_stats=stats['action'], pos_stats=stats['agent_pos'])
    
    print(f"Policy loaded successfully from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['loss']:.4f}")
    
    return policy, policy_config, stats


def evaluate_single_episode(policy, policy_config, stats, max_steps=200, render=False, save_video=False, seed=100000):
    """使用policy.predict_action进行推理的简化版评估"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建环境
    env = PushTImageEnv()
    env.seed(seed)
    
    # 获取第一个观测
    obs, info = env.reset()
    
    # 保持最后obs_horizon步的观测队列
    obs_horizon = policy_config.obs_horizon
    pred_horizon = policy_config.pred_horizon
    action_horizon = policy_config.action_horizon
    action_dim = policy_config.action_dim
    
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    
    # 保存可视化和奖励
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0
    
    # 视频录制
    if save_video:
        video_recorder = VideoRecorder.create_h264(fps=10)
        video_path = f"eval_episode_seed_{seed}.mp4"
        video_recorder.start(video_path)
        video_recorder.write_frame(imgs[0])
    
    with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
        while not done:
          
            images = np.stack([x['image'] for x in obs_deque])  # (obs_horizon, H, W, C)
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])  # (obs_horizon, 2)
         
            nimages = images.astype(np.float32) * 255.0  # to [0,255]
            nagent_poses = normalize_data(agent_poses, stats['agent_pos'])

            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
            nimages = nimages.unsqueeze(0)
            # (obs_horizon, H, W, C) -> (1, obs_horizon, C, H, W)
            #nimages = nimages.permute(0, 3, 1, 2).unsqueeze(0)
            
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (obs_horizon, 2) -> (1, obs_horizon, 2)
            nagent_poses = nagent_poses.unsqueeze(0)
            
            # 构建观测字典
            obs_dict = {
                'image': nimages,      # (1, obs_horizon, C, H, W)
                'agent_pos': nagent_poses  # (1, obs_horizon, 2)
            }
            
            # 使用policy的predict_action方法
            with torch.no_grad():
                action_pred = policy.predict_action(obs_dict)  # (1, pred_horizon, action_dim)
            
            # action_pred = action_pred[0].cpu().numpy()  # (pred_horizon, action_dim)
            action_pred = action_pred.squeeze(0)  # (pred_horizon, action_dim)
            # 只取action_horizon个动作
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end, :]  # (action_horizon, action_dim)
            
            # 执行action_horizon个步骤，不重新规划
            for i in range(len(action)):
                # 环境步进
                obs, reward, done, _, info = env.step(action[i])
                # 保存观测
                obs_deque.append(obs)
                # 保存奖励和可视化
                rewards.append(reward)
                img = env.render(mode='rgb_array')
                imgs.append(img)
                
                # 渲染
                if render:
                    cv2.imshow('PushT Evaluation', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 录制视频
                if save_video:
                    video_recorder.write_frame(img)
                
                # 更新进度条
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward, max_reward=max(rewards) if rewards else 0)
                
                if step_idx > max_steps:
                    done = True
                if done:
                    break
    
    # 停止录制
    if save_video:
        video_recorder.stop()
        print(f"Video saved to: {video_path}")
    
    # 关闭环境
    env.close()
    if render:
        cv2.destroyAllWindows()
    
    # 打印最大目标覆盖率
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
    """评估policy在多个episode中的性能"""
    
    print(f"开始评估... 共{num_episodes}个episode")
    
    all_results = []
    success_count = 0
    
    for episode in range(num_episodes):
        # 使用不同的seed避免重复初始状态
        seed = 100000 + episode * 10
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
    
    # 计算统计信息
    max_scores = [r['max_score'] for r in all_results]
    episode_lengths = [r['episode_length'] for r in all_results]
    
    mean_score = np.mean(max_scores)
    std_score = np.std(max_scores)
    mean_length = np.mean(episode_lengths)
    success_rate = success_count / num_episodes
    
    print("\n" + "="*50)
    print("评估结果:")
    print(f"平均得分: {mean_score:.4f} ± {std_score:.4f}")
    print(f"平均episode长度: {mean_length:.1f}")
    print(f"成功率: {success_rate:.1%} ({success_count}/{num_episodes})")
    print(f"得分范围: {min(max_scores):.4f} - {max(max_scores):.4f}")
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
    checkpoint_path = "/media/z/data/mzq/code/diffusion_policy_z/checkpoints/pusht/290_0.0035.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("可用的checkpoint文件:")
        checkpoint_dir = Path(checkpoint_path).parent
        if checkpoint_dir.exists():
            for f in checkpoint_dir.glob("*.pth"):
                print(f"  - {f}")
        return
    
    # 加载policy和统计信息
    policy, policy_config, stats = load_policy_from_checkpoint(checkpoint_path, device)
    print("Policy加载成功!")
    
    # 评估policy
    results = evaluate_policy(
        policy=policy,
        policy_config=policy_config,
        stats=stats,
        num_episodes=50,  # 可以调整episode数量
        max_steps=1000,   # 与参考代码一致
        render=False,    # 是否显示渲染画面
        save_video=False  # 是否保存视频
    )

if __name__ == "__main__":
    main()