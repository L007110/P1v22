import torch
import torch.optim as optim
import os
import shutil
import numpy as np
import time
import gc
import sys
import math
import copy
import random

# 引入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Parameters
import Main
import GNNModel
import Topology
from logger import global_logger
from Topology import formulate_global_list_dqn


# =======================================================================
# 🔧 1. 修复版：混合密度拦截器 (Stable Density Mixer)
# =======================================================================
class VehicleDensityMixer:
    def __init__(self, original_func):
        self.original_func = original_func
        self.current_high_level_n = 60
        self.mix_ratio = 0.2
        self.low_density_candidates = [20, 40]
        self.active = True

        # [Fix 1] 伪 Episode 控制
        self.episode_length = 50  # 每 50 步视为一个稳定的 Episode
        self.step_counter = 0  # 内部计数器
        self.current_target = 60  # 当前锁定的目标密度

    def set_level(self, n):
        """更新当前课程的主难度，并重置计数器"""
        self.current_high_level_n = n
        # 切换关卡时，强制刷新一次目标
        self._refresh_target()

    def _refresh_target(self):
        """掷骰子决定接下来的 Episode 密度"""
        if self.active and random.random() < self.mix_ratio:
            self.current_target = random.choice(self.low_density_candidates)
            # print(f"🎲 [Mix] 新 Episode 开始: 切换至低密度 N={self.current_target}")
        else:
            self.current_target = self.current_high_level_n
            # print(f"🎲 [Mix] 新 Episode 开始: 保持主难度 N={self.current_target}")

    def __call__(self, vehicle_id, vehicle_list, target_count=None, speed_kmh=60):
        # [Fix 1] 检查是否需要切换密度 (模拟 Episode Reset)
        # 如果是第一步，或者步数达到了 Episode 长度
        if self.step_counter % self.episode_length == 0:
            self._refresh_target()

        self.step_counter += 1
        real_target = self.current_target

        # [Fix 2] 稳定裁剪 (Stable Pruning)
        # 只要列表前面的车 (保留 ID 和历史信息)，不要随机抽样！
        if len(vehicle_list) > real_target:
            # vehicle_list = sorted(vehicle_list, key=lambda v: v.id)[:real_target] # 如果列表本来就是乱的，可以用这个
            # 但通常 vehicle_list 是 append 进去的，直接切片就是保留最老的车
            vehicle_list = vehicle_list[:real_target]

        # 调用原始函数
        return self.original_func(vehicle_id, vehicle_list, target_count=real_target, speed_kmh=speed_kmh)


# 🔥 应用拦截补丁
print("🛠️ 正在安装车辆密度拦截器 (V3 Stable)...")
original_movement_func = Topology.vehicle_movement
density_mixer = VehicleDensityMixer(original_movement_func)
Topology.vehicle_movement = density_mixer
print("✅ 拦截器安装完成！")


# =======================================================================
# 🔧 2. 修复版：缓冲区持久化 (Partial Inheritance)
# =======================================================================
class PersistentBufferWrapper:
    _instance_store = []

    @classmethod
    def save_buffer(cls, buffer_instance):
        if buffer_instance is not None and len(buffer_instance) > 0:
            cls._instance_store = [buffer_instance.buffer]
            print(f"   💾 [Buffer] 已保存本关卡 {len(buffer_instance)} 条经验")


class PatchedGNNReplayBuffer(Main.GNNReplayBuffer):
    current_instance = None

    def __init__(self, capacity):
        super().__init__(capacity)
        PatchedGNNReplayBuffer.current_instance = self

        # [Fix 3] 软继承 (Soft Inheritance)
        if PersistentBufferWrapper._instance_store:
            old_data = PersistentBufferWrapper._instance_store[0]
            inherit_ratio = 0.5  # 只继承 50%
            inherit_size = int(len(old_data) * inherit_ratio)

            if inherit_size > 0:
                # 这里可以用 random.sample，因为经验之间是独立的 (只要不打乱内部的时序元组)
                # Replay Buffer 的顺序通常不影响训练 (除了 PER，但这里是 GNNBuffer)
                injected_data = random.sample(old_data, inherit_size)
                self.buffer = copy.deepcopy(injected_data)
                print(f"   🔄 [Buffer] 软继承: 抽取上一关 {inherit_size}/{len(old_data)} 条经验")
            else:
                self.buffer = []


Main.GNNReplayBuffer = PatchedGNNReplayBuffer

# =======================================================================
# 3. 课程配置
# =======================================================================

LEVEL_CONFIGS = {
    # N : (LR, TotalEpochs, StartEpsilon)
    60: (0.0004, 400, 0.5),
    80: (0.0004, 300, 0.2),
    100: (0.0003, 300, 0.15),
    120: (0.0003, 300, 0.1),
    140: (0.0002, 300, 0.1)
}

CURRICULUM_LEVELS = sorted(LEVEL_CONFIGS.keys())
FINAL_EPSILON = 0.01
FINAL_MODEL_NAME = "model_Universal_LargeMap_MixV4.pt"


# =======================================================================
# 4. 主流程
# =======================================================================

def calculate_decay(start_eps, end_eps, total_epochs):
    target_step = int(total_epochs * 0.80)
    if target_step <= 0: return 0.9
    return math.pow(end_eps / start_eps, 1.0 / target_step)


def run_mixed_curriculum_v4():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 70}")
    print(f"🚀 启动混合密度课程学习 V4 (Stable Mix)")
    print(f"✨ Fix 1: 伪 Episode 机制 (每50步切换密度)")
    print(f"✨ Fix 2: 稳定车辆裁剪 (保留ID连续性)")
    print(f"✨ Fix 3: 50% 缓冲区软继承")
    print(f"📍 设备: {device}")
    print(f"{'=' * 70}\n")

    Parameters.USE_GNN_ENHANCEMENT = True
    Parameters.GNN_ARCH = "HYBRID"
    Parameters.SCENE_SCALE_X = 1200
    Parameters.SCENE_SCALE_Y = 1200

    last_passed_model_path = None
    current_level_idx = 0

    while current_level_idx < len(CURRICULUM_LEVELS):
        n_vehicles = CURRICULUM_LEVELS[current_level_idx]
        current_lr, total_epochs, start_epsilon = LEVEL_CONFIGS[n_vehicles]

        # 🚨 更新主难度
        density_mixer.set_level(n_vehicles)

        decay_rate = calculate_decay(start_epsilon, FINAL_EPSILON, total_epochs)

        print(f"\n" + "=" * 60)
        print(f"🔥 [LEVEL {current_level_idx + 1}] 主难度 N={n_vehicles} (Mix Enabled)")
        print(f"🎲 Epsilon: {start_epsilon} -> {FINAL_EPSILON}")
        print("=" * 60)

        # --- 环境准备 ---
        gc.collect()
        torch.cuda.empty_cache()

        Parameters.TRAINING_VEHICLE_TARGET = n_vehicles
        Parameters.NUM_VEHICLES = n_vehicles
        Parameters.RL_N_EPOCHS = total_epochs
        Parameters.ABLATION_SUFFIX = f"_MixV4_N{n_vehicles}"

        global_logger._init_metrics_storage()
        formulate_global_list_dqn(Parameters.global_dqn_list, device)

        for dqn in Parameters.global_dqn_list:
            dqn.epsilon = start_epsilon

        # --- 模型加载 ---
        GNNModel.global_gnn_model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(device)
        GNNModel.global_target_gnn_model = GNNModel.EnhancedHeteroGNN(node_feature_dim=12, hidden_dim=64).to(device)

        if last_passed_model_path and os.path.exists(last_passed_model_path):
            print(f"   📥 继承权重: {last_passed_model_path}")
            checkpoint = torch.load(last_passed_model_path, map_location=device)
            GNNModel.global_gnn_model.load_state_dict(checkpoint)
            GNNModel.global_target_gnn_model.load_state_dict(checkpoint)
        else:
            print("   🌱 [Cold Start] 初始化网络")
            GNNModel.update_target_gnn()

        gnn_optimizer = optim.Adam(GNNModel.global_gnn_model.parameters(), lr=current_lr)

        # --- 训练 ---
        try:
            Main.rl(gnn_optimizer=gnn_optimizer, device=device)

            # 保存 Buffer
            if hasattr(PatchedGNNReplayBuffer, 'current_instance'):
                active_buf = PatchedGNNReplayBuffer.current_instance
                PersistentBufferWrapper.save_buffer(active_buf)

            save_name = f"checkpoint_mixv4_passed_N{n_vehicles}.pt"
            torch.save(GNNModel.global_gnn_model.state_dict(), save_name)
            last_passed_model_path = save_name
            current_level_idx += 1

        except Exception as e:
            print(f"   ❌ 训练中断: {e}")
            import traceback
            traceback.print_exc()
            return

    print("\n" + "=" * 70)
    print("🏆 混合训练完成！")
    if last_passed_model_path:
        shutil.copy(last_passed_model_path, FINAL_MODEL_NAME)
        print(f"💾 最终模型: {FINAL_MODEL_NAME}")
    print("=" * 70)


if __name__ == "__main__":
    run_mixed_curriculum_v4()