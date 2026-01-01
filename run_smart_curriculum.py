import torch
import torch.optim as optim
import os
import shutil
import numpy as np
import random
import time
import gc
import sys
import math

# 引入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Parameters
import Main
import GNNModel
from logger import global_logger
from Topology import formulate_global_list_dqn

# =======================================================================
# 1. 课程配置 (针对 1200x1200 大地图)
# =======================================================================

# [训练关卡设置]
# 我们跳过 N=20/40，因为在大地图上它们太稀疏，训练效率极低。
# 我们从 N=60 (密度~42) 开始，这是建立有效连接的"甜蜜点"。
LEVEL_CONFIGS = {
    # N : (LR, TotalEpochs)
    60: (0.0004, 400),  # Level 1: 基础夯实 (多跑几轮)
    80: (0.0004, 300),  # Level 2: 进阶
    100: (0.0003, 300),  # Level 3: 拥堵 (Paper常见高点)
    120: (0.0003, 300),  # Level 4: 严重拥堵
    140: (0.0002, 300)  # Level 5: 极限施压 (SOTA)
}

CURRICULUM_LEVELS = sorted(LEVEL_CONFIGS.keys())

# 考核及格线 (Soft Feasible Score)
# 大地图上干扰源距离较远，物理信道条件较好，要求可以高一点
PASS_THRESHOLDS = {
    60: 0.85,
    80: 0.88,
    100: 0.90,
    120: 0.85,
    140: 0.80
}

# 基础参数
WARMUP_EPOCHS = 100
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
STABILITY_THRESHOLD = 0.03
SCORE_WINDOW_SIZE = 30
MAX_RETRIES = 3
FINAL_MODEL_NAME = "model_Universal_LargeMap.pt"


# =======================================================================

def calculate_decay(start_eps, end_eps, total_epochs):
    """
    动态计算衰减率，确保在 85% 的进度处降到 FINAL_EPSILON
    剩下的 15% 用于纯利用 (Exploitation) 以冲刺高分
    """
    target_step = int(total_epochs * 0.85)
    if target_step <= 0: return 0.95
    # 公式: start * (decay ^ steps) = end
    return math.pow(end_eps / start_eps, 1.0 / target_step)


def run_adaptive_curriculum():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 70}")
    print(f"🚀 启动自适应课程学习 (Large Map 1200x1200)")
    print(f"📍 设备: {device}")
    print(f"🗺️  注意: 请确认 Parameters.py 中 SCENE_SCALE = 1200")
    print(f"📈 训练路线: {CURRICULUM_LEVELS}")
    print(f"💡 策略: 从 N=60 起步以保证样本质量，N=20/40 将在测试阶段验证")
    print(f"{'=' * 70}\n")

    Parameters.USE_GNN_ENHANCEMENT = True
    Parameters.GNN_ARCH = "HYBRID"

    last_passed_model_path = None
    current_level_idx = 0

    while current_level_idx < len(CURRICULUM_LEVELS):
        n_vehicles = CURRICULUM_LEVELS[current_level_idx]
        target_score = PASS_THRESHOLDS.get(n_vehicles, 0.80)
        current_lr, total_epochs = LEVEL_CONFIGS[n_vehicles]

        # 动态计算 Decay
        decay_rate = calculate_decay(INITIAL_EPSILON, FINAL_EPSILON, total_epochs)

        print(f"\n" + "=" * 60)
        print(f"🔥 [LEVEL {current_level_idx + 1}] 挑战关卡: N={n_vehicles} (Large Map)")
        print(f"💊 处方: LR={current_lr}, Epochs={total_epochs}")
        print(f"📉 Epsilon Decay: {decay_rate:.6f} (将在第 {int(total_epochs * 0.85)} 轮归零)")
        print(f"🎯 目标: Feasible V2V >= {target_score * 100:.1f}%")
        print("=" * 60)

        passed = False
        attempt = 0

        while not passed and attempt < MAX_RETRIES:
            attempt += 1

            # --- 步骤 0: 内存清洗 ---
            gc.collect()
            torch.cuda.empty_cache()
            if hasattr(Main, 'global_gnn_buffer'):
                Main.global_gnn_buffer = None

            # --- 步骤 A: 注入参数 ---
            Parameters.TRAINING_VEHICLE_TARGET = n_vehicles
            Parameters.NUM_VEHICLES = n_vehicles
            Parameters.RL_N_EPOCHS = total_epochs
            Parameters.RL_EPSILON_DECAY = decay_rate
            Parameters.ABLATION_SUFFIX = f"_LargeMap_N{n_vehicles}_Try{attempt}"

            global_logger._init_metrics_storage()

            # --- 步骤 B: 网络重置 ---
            formulate_global_list_dqn(Parameters.global_dqn_list, device)
            # 暴力重置 Epsilon
            for dqn_agent in Parameters.global_dqn_list:
                dqn_agent.epsilon = INITIAL_EPSILON
                if hasattr(dqn_agent, 'epsilon_decay'):
                    dqn_agent.epsilon_decay = decay_rate

            GNNModel.global_gnn_model = GNNModel.EnhancedHeteroGNN(
                node_feature_dim=12, hidden_dim=64
            ).to(device)
            GNNModel.global_target_gnn_model = GNNModel.EnhancedHeteroGNN(
                node_feature_dim=12, hidden_dim=64
            ).to(device)

            # --- 步骤 C: 接力存档 ---
            if last_passed_model_path and os.path.exists(last_passed_model_path):
                print(f"   📥 继承权重: {last_passed_model_path}")
                checkpoint = torch.load(last_passed_model_path, map_location=device)
                GNNModel.global_gnn_model.load_state_dict(checkpoint)
                GNNModel.global_target_gnn_model.load_state_dict(checkpoint)

                if attempt > 1:
                    print("   ⚠️ [补考] Epsilon=0.6")
                    for dqn in Parameters.global_dqn_list: dqn.epsilon = 0.6
            else:
                if current_level_idx == 0:
                    print("   🌱 [Cold Start] 从零开始 (N=60)")
                    GNNModel.update_target_gnn()

            # 优化器
            gnn_optimizer = optim.Adam(GNNModel.global_gnn_model.parameters(), lr=current_lr)

            # --- 步骤 D: 训练 ---
            try:
                Main.rl(gnn_optimizer=gnn_optimizer, device=device)

                # 保存中间结果
                attempt_save_name = f"checkpoint_attempt_LargeMap_N{n_vehicles}_Try{attempt}.pt"
                torch.save(GNNModel.global_gnn_model.state_dict(), attempt_save_name)

                # --- 步骤 E: 判卷 ---
                raw_v2v = np.array(global_logger.metrics['v2v_success_rate'])
                raw_v2i = np.array(global_logger.metrics['v2i_sum_capacity'])

                if len(raw_v2v) > WARMUP_EPOCHS + 5:
                    v2v_valid = raw_v2v[WARMUP_EPOCHS:]
                    v2i_valid = raw_v2i[WARMUP_EPOCHS:]
                else:
                    v2v_valid = raw_v2v
                    v2i_valid = raw_v2i

                eval_window = min(SCORE_WINDOW_SIZE, len(v2v_valid))
                v2v_tail = v2v_valid[-eval_window:]
                v2i_tail = v2i_valid[-eval_window:]

                # Soft Score
                penalty_mask = (v2i_tail >= Parameters.V2I_CAPACITY_THRESHOLD).astype(float)
                weighted_scores = v2v_tail * penalty_mask

                final_score = np.mean(weighted_scores)
                score_std = np.std(weighted_scores)

                # 诊断
                raw_avg = np.mean(v2v_tail)
                feasible_rate = np.mean(penalty_mask)

                print(f"   📊 结果: Soft Score={final_score * 100:.2f}% | Std={score_std:.4f}")
                print(f"      (Raw V2V: {raw_avg * 100:.1f}% | Feasible Rate: {feasible_rate * 100:.1f}%)")

                if final_score >= target_score and score_std <= STABILITY_THRESHOLD:
                    print(f"   ✅ 晋级！")
                    save_name = f"checkpoint_passed_LargeMap_N{n_vehicles}.pt"
                    torch.save(GNNModel.global_gnn_model.state_dict(), save_name)
                    last_passed_model_path = save_name
                    passed = True
                    current_level_idx += 1
                else:
                    print(f"   ❌ 挂科。")
                    if feasible_rate < 0.8:
                        print("      -> 警告: V2I 违规严重，请检查约束！")
                    if attempt >= MAX_RETRIES:
                        print("   ☠️ 课程终止。")
                        return

            except Exception as e:
                print(f"   [Error] {e}")
                import traceback
                traceback.print_exc()
                return

    print("\n" + "=" * 70)
    print("🏆 大地图训练全部完成！")
    if last_passed_model_path:
        shutil.copy(last_passed_model_path, FINAL_MODEL_NAME)
        print(f"💾 最终模型: {FINAL_MODEL_NAME}")
        print("💡 现在你可以用这个模型去测试 N=20, 40... 140 了！")
    print("=" * 70)


if __name__ == "__main__":
    run_adaptive_curriculum()