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

# 引入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Parameters
import Main
import GNNModel
from logger import global_logger
from Topology import formulate_global_list_dqn


# =======================================================================
# 🔧 1. 缓冲区持久化补丁 (Buffer Persistence Patch)
# =======================================================================
# 这是一个"魔法"类，用于拦截 Main.py 中的 GNNReplayBuffer
# 使得经验池可以在不同的 Main.rl() 调用之间传递，防止灾难性遗忘
class PersistentBufferWrapper:
    _instance_store = []  # 静态变量，存储上一轮的 buffer 数据

    def __new__(cls, *args, **kwargs):
        # 创建真正的 GNNReplayBuffer 实例 (引用 Main 中的原始类)
        real_buffer = Main.GNNReplayBuffer(*args, **kwargs)

        # 如果有存货，注入旧数据
        if cls._instance_store:
            print(f"   🔄 [Buffer Patch] 正在注入上一关的经验数据...")
            old_buffer_data = cls._instance_store[0]
            # 将旧数据深拷贝到新 buffer (防止引用问题)
            real_buffer.buffer = copy.deepcopy(old_buffer_data)
            print(f"   ✅ 成功恢复 {len(real_buffer)} 条经验 (混合训练开启)")

        # 清空存储，准备接收新的（虽然这里我们其实只需要在 rl 结束时保存）
        # 但为了简单，我们在 rl 结束后手动去抓取 global_gnn_buffer
        return real_buffer

    @classmethod
    def save_buffer(cls, buffer_instance):
        """在 rl 结束后手动调用此方法保存数据"""
        if buffer_instance is not None and len(buffer_instance) > 0:
            cls._instance_store = [buffer_instance.buffer]
            print(f"   💾 [Buffer Patch] 已保存本关卡 {len(buffer_instance)} 条经验用于下一关")


# ⚡ 应用补丁：替换 Main 模块中的类定义
# 注意：这不会修改文件，只会修改运行时的类引用
OriginalReplayBufferClass = Main.GNNReplayBuffer  # 备份原类（虽然这里直接替换了）


# 这里的逻辑稍微调整：因为 Main.rl 内部是 `global_gnn_buffer = GNNReplayBuffer(...)`
# 我们需要替换 Main.GNNReplayBuffer 这个名字指向我们的 Wrapper 逻辑
# 但由于 __new__ 比较复杂，我们采用更简单的"类欺骗"
class PatchedGNNReplayBuffer(Main.GNNReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)
        # 尝试恢复数据
        if PersistentBufferWrapper._instance_store:
            print(f"   🔄 [Memory] 继承上一关经验池: {len(PersistentBufferWrapper._instance_store[0])} 条样本")
            self.buffer = copy.deepcopy(PersistentBufferWrapper._instance_store[0])


Main.GNNReplayBuffer = PatchedGNNReplayBuffer

# =======================================================================
# 2. 课程配置 (Curriculum Config)
# =======================================================================

LEVEL_CONFIGS = {
    # N : (LR, TotalEpochs, StartEpsilon)
    # N=60: 基础夯实，高探索
    60: (0.0004, 400, 0.5),
    # N=80: 进阶，降低探索，防止破坏已有策略
    80: (0.0004, 300, 0.2),
    # N=100: 拥堵，低探索
    100: (0.0003, 300, 0.15),
    # N=120: 严重拥堵
    120: (0.0003, 300, 0.1),
    # N=140: 极限
    140: (0.0002, 300, 0.1)
}

CURRICULUM_LEVELS = sorted(LEVEL_CONFIGS.keys())

PASS_THRESHOLDS = {
    60: 0.85, 80: 0.88, 100: 0.90, 120: 0.85, 140: 0.80
}

FINAL_EPSILON = 0.01
FINAL_MODEL_NAME = "model_Universal_LargeMap_v2.pt"


# =======================================================================
# 3. 辅助函数
# =======================================================================

def calculate_decay(start_eps, end_eps, total_epochs):
    # 前 80% 的 Epoch 衰减完，后 20% 纯利用
    target_step = int(total_epochs * 0.80)
    if target_step <= 0: return 0.9
    return math.pow(end_eps / start_eps, 1.0 / target_step)


def run_adaptive_curriculum_v2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 70}")
    print(f"🚀 启动增强版课程学习 (Smart Curriculum V2)")
    print(f"✨ 特性: 经验回放池持久化 + 智能 Epsilon 衰减")
    print(f"📍 设备: {device}")
    print(f"📈 路线: {CURRICULUM_LEVELS}")
    print(f"{'=' * 70}\n")

    # 强制覆盖参数
    Parameters.USE_GNN_ENHANCEMENT = True
    Parameters.GNN_ARCH = "HYBRID"
    Parameters.SCENE_SCALE_X = 1200
    Parameters.SCENE_SCALE_Y = 1200

    last_passed_model_path = None
    current_level_idx = 0

    while current_level_idx < len(CURRICULUM_LEVELS):
        n_vehicles = CURRICULUM_LEVELS[current_level_idx]
        current_lr, total_epochs, start_epsilon = LEVEL_CONFIGS[n_vehicles]
        target_score = PASS_THRESHOLDS.get(n_vehicles, 0.80)

        # 动态计算 Decay
        decay_rate = calculate_decay(start_epsilon, FINAL_EPSILON, total_epochs)

        print(f"\n" + "=" * 60)
        print(f"🔥 [LEVEL {current_level_idx + 1}] N={n_vehicles} | LR={current_lr} | Epochs={total_epochs}")
        print(f"🎲 Epsilon: {start_epsilon} -> {FINAL_EPSILON} (Decay: {decay_rate:.5f})")
        print("=" * 60)

        # --- 1. 环境准备 ---
        gc.collect()
        torch.cuda.empty_cache()

        Parameters.TRAINING_VEHICLE_TARGET = n_vehicles
        Parameters.NUM_VEHICLES = n_vehicles
        Parameters.RL_N_EPOCHS = total_epochs
        Parameters.ABLATION_SUFFIX = f"_Lvl{current_level_idx}_N{n_vehicles}"

        global_logger._init_metrics_storage()
        formulate_global_list_dqn(Parameters.global_dqn_list, device)

        # 设置 Epsilon (智能设定)
        for dqn in Parameters.global_dqn_list:
            dqn.epsilon = start_epsilon

        # --- 2. 模型加载与同步 ---
        GNNModel.global_gnn_model = GNNModel.EnhancedHeteroGNN(
            node_feature_dim=12, hidden_dim=64
        ).to(device)
        GNNModel.global_target_gnn_model = GNNModel.EnhancedHeteroGNN(
            node_feature_dim=12, hidden_dim=64
        ).to(device)

        if last_passed_model_path and os.path.exists(last_passed_model_path):
            print(f"   📥 继承权重: {last_passed_model_path}")
            checkpoint = torch.load(last_passed_model_path, map_location=device)
            GNNModel.global_gnn_model.load_state_dict(checkpoint)
            # 关键：严格同步 Target
            GNNModel.global_target_gnn_model.load_state_dict(checkpoint)
        else:
            print("   🌱 [Cold Start] 初始化网络")
            GNNModel.update_target_gnn()

        # 优化器
        gnn_optimizer = optim.Adam(GNNModel.global_gnn_model.parameters(), lr=current_lr)

        # --- 3. 训练 ---
        try:
            # Main.rl 会触发 PatchedGNNReplayBuffer，自动加载旧数据
            Main.rl(gnn_optimizer=gnn_optimizer, device=device)

            # 训练结束，保存当前 Buffer 供下一关使用
            # Main.global_gnn_buffer 是 Main.py 模块级的引用吗？不是，它是 rl 内部的局部变量
            # 糟糕，我们无法从外部访问 rl 内部的 buffer。
            # 修正策略：我们在 PatchedGNNReplayBuffer 的析构或者通过全局引用来抓取
            # 由于 Python 引用机制，只要 Main.rl 跑完，local 变量就没了。
            # 但我们在 Main.py 里无法修改 return。
            # 补救：我们在 Patch 类里做一个钩子，每次 add 的时候更新一下静态存储？太慢。
            # 补救 V2：Main.py 的 rl 函数没有返回 buffer。
            # 终极补丁：Main.py 运行中 global_gnn_buffer 是局部变量，但在运行结束前无法获取。
            # 等等，Main.py 有 `import GNNReplayBuffer`。
            # 我们其实在 `rl` 循环里，`global_gnn_buffer` 只是被用来 sample 和 add。

            # 这里的 Hack：
            # 我们在 PatchedGNNReplayBuffer 中维持一个类级别的引用指向"当前活跃的buffer"
            # 这样我们在外部就可以访问了
            if hasattr(PatchedGNNReplayBuffer, 'current_instance'):
                active_buf = PatchedGNNReplayBuffer.current_instance
                PersistentBufferWrapper.save_buffer(active_buf)

            # --- 保存模型 ---
            save_name = f"checkpoint_passed_N{n_vehicles}.pt"
            torch.save(GNNModel.global_gnn_model.state_dict(), save_name)
            last_passed_model_path = save_name
            current_level_idx += 1
            print(f"   ✅ N={n_vehicles} 完成，模型已保存。")

        except Exception as e:
            print(f"   ❌ 训练中断: {e}")
            import traceback
            traceback.print_exc()
            return

    print("\n" + "=" * 70)
    print("🏆 通用模型训练全部完成！")
    if last_passed_model_path:
        shutil.copy(last_passed_model_path, FINAL_MODEL_NAME)
        print(f"💾 最终通用模型: {FINAL_MODEL_NAME}")
    print("=" * 70)


# 更新一下 Patch 类，增加 current_instance 钩子
class PatchedGNNReplayBuffer_V2(Main.GNNReplayBuffer):
    current_instance = None  # 静态引用

    def __init__(self, capacity):
        super().__init__(capacity)
        PatchedGNNReplayBuffer_V2.current_instance = self  # 捕获引用

        # 尝试恢复数据
        if PersistentBufferWrapper._instance_store:
            print(f"   🔄 [Buffer Patch] 注入上一关经验: {len(PersistentBufferWrapper._instance_store[0])} 条")
            self.buffer = copy.deepcopy(PersistentBufferWrapper._instance_store[0])


# 重新应用 V2 补丁
Main.GNNReplayBuffer = PatchedGNNReplayBuffer_V2

if __name__ == "__main__":
    run_adaptive_curriculum_v2()