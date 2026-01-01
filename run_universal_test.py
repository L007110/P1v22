import torch
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import gc

# 引入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Parameters
import Main
import GNNModel
from logger import global_logger
from Topology import formulate_global_list_dqn

# ==================== 配置区域 ====================
MODEL_PATH = "model_Universal_Final_Mixed.pt"  # 你的最终模型路径
TEST_SCENARIOS = [20, 40, 60, 80, 100, 120, 140]  # 测试密度列表
EPISODES_PER_SCENARIO = 50  # 每个密度跑多少轮取平均
SCENE_SCALE = 1200  # 确保是大地图


# ================================================

def run_universal_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"🧪 启动通用模型全场景测试 (Universal Test)")
    print(f"📍 设备: {device}")
    print(f"🗺️  地图: {SCENE_SCALE}x{SCENE_SCALE}")
    print(f"📂 模型: {MODEL_PATH}")
    print(f"{'=' * 60}\n")

    # 1. 强制设置环境参数
    Parameters.SCENE_SCALE_X = SCENE_SCALE
    Parameters.SCENE_SCALE_Y = SCENE_SCALE
    Parameters.USE_GNN_ENHANCEMENT = True
    Parameters.GNN_ARCH = "HYBRID"

    # 2. 加载模型 (只加载一次)
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        return

    print("📥 正在加载 GNN 模型...")
    gnn_model = GNNModel.EnhancedHeteroGNN(
        node_feature_dim=12, hidden_dim=64
    ).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    gnn_model.load_state_dict(checkpoint)
    gnn_model.eval()  # 开启评估模式 (关闭 Dropout 等)

    # 注入全局
    GNNModel.global_gnn_model = gnn_model

    # 存储结果
    results = []

    # 3. 循环测试不同密度
    for n_vehicles in TEST_SCENARIOS:
        print(f"\n🚗 [Testing] 正在测试车辆数 N={n_vehicles} ...")

        # --- 这里的关键是重置环境 ---
        Parameters.NUM_VEHICLES = n_vehicles
        Parameters.TRAINING_VEHICLE_TARGET = n_vehicles

        # 重新生成 DQN 列表 (因为车辆数变了)
        formulate_global_list_dqn(Parameters.global_dqn_list, device)

        # 强制所有 Agent 关闭探索 (Epsilon = 0)
        for dqn in Parameters.global_dqn_list:
            dqn.epsilon = 0.0

        # 清空之前的 Metrics
        global_logger._init_metrics_storage()

        # 运行测试循环
        # 我们复用 Main.rl 但不传入 optimizer，这样就不会由 backward
        # 或者为了保险，我们可以直接调用 Main.test() 如果你有的话，
        # 这里我们模拟一个纯推理的 Loop

        episode_v2v_rates = []
        episode_v2i_caps = []

        start_time = time.time()

        # 这里我们利用 Main.rl 的逻辑，但为了避免它进行训练操作，
        # 我们需要在 Main.py 里确保没有 optimizer 就不会 step。
        # 如果 Main.rl 强制训练，我们这里可以使用 Main.run_episode (假设有) 或者直接跑 Main.rl
        # 但最简单的方法是：设置 Run Mode

        # 由于没法直接改 Main.py 的代码，我们这里调用 Main.rl
        # 但传入 None 作为 optimizer，通常这会跳过反向传播
        try:
            # 这里的 Hack: 传入 None optimizer
            # 同时将 Parameters.RL_N_EPOCHS 设为测试轮数
            Parameters.RL_N_EPOCHS = EPISODES_PER_SCENARIO

            # 临时静音 logger 以免刷屏
            # Main.rl 会运行 EPISODES_PER_SCENARIO 轮
            Main.rl(gnn_optimizer=None, device=device)

            # 收集数据
            raw_v2v = global_logger.metrics['v2v_success_rate']
            raw_v2i = global_logger.metrics['v2i_sum_capacity']

            # 计算平均值
            avg_v2v = np.mean(raw_v2v) * 100
            avg_v2i = np.mean(raw_v2i)
            feasible_rate = np.mean(np.array(raw_v2i) >= Parameters.V2I_CAPACITY_THRESHOLD) * 100

            print(f"   ✅ N={n_vehicles} 完成 | 耗时: {time.time() - start_time:.1f}s")
            print(f"      -> V2V 成功率: {avg_v2v:.2f}%")
            print(f"      -> V2I 满足率: {feasible_rate:.1f}%")

            results.append({
                "Density (N)": n_vehicles,
                "V2V Success Rate (%)": avg_v2v,
                "V2I Sum Capacity (Mbps)": avg_v2i,
                "Feasibility (%)": feasible_rate
            })

        except Exception as e:
            print(f"   ❌ N={n_vehicles} 测试出错: {e}")
            import traceback
            traceback.print_exc()

        # 内存清理
        gc.collect()

    # 4. 生成报告
    print("\n" + "=" * 60)
    print("📊 最终测试报告 (Final Report)")
    print("=" * 60)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # 保存 CSV
    df.to_csv("test_results_universal.csv", index=False)
    print(f"\n💾 数据已保存至: test_results_universal.csv")

    # 5. 简单绘图
    plot_results(df)


def plot_results(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df["Density (N)"], df["V2V Success Rate (%)"], marker='o', linewidth=2, label='Proposed GNN-DRL')

    # 装饰
    plt.title(f"Universal Model Performance ({SCENE_SCALE}x{SCENE_SCALE}m)", fontsize=14)
    plt.xlabel("Number of Vehicles (N)", fontsize=12)
    plt.ylabel("V2V Success Rate (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 105)
    plt.legend()

    plt.savefig("test_result_plot.png")
    print("📈 图表已保存至: test_result_plot.png")
    # plt.show() # 如果在服务器上跑，注释掉这行


if __name__ == "__main__":
    import time

    run_universal_test()