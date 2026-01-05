import torch
import sys
import os

# 引入 Main 模块，因为我们要复用 Main.test()
import Main
import Parameters

# ================= 配置区域 =================
# 1. 指定要测试的模型文件 (指向 V2 训练产出的文件)
TARGET_MODEL_PATH = "model_Universal_LargeMap_MixV4.pt"

# 2. 指定测试场景 (必须覆盖默认值，确保测试 N=20/40 等低密度场景)
TEST_SCENARIOS = [20, 40, 60, 80, 100, 120, 140]

# 3. 每个场景跑多少轮 (建议 50-100 以获得稳定平均值)
EPISODES_PER_SCENARIO = 50


# ===========================================

def run_evaluation():
    print(f"🚀 启动最终评估脚本")
    print(f"📂 加载模型: {TARGET_MODEL_PATH}")
    print(f"🚗 测试密度: {TEST_SCENARIOS}")

    # --- 关键步骤：参数覆盖 (Monkey Patching) ---
    # 我们直接修改 Main 模块和 Parameters 模块中的变量
    # 这样 Main.test() 运行时就会使用我们的配置

    # 1. 强制覆盖模型路径
    Parameters.MODEL_PATH_GNN = TARGET_MODEL_PATH
    Main.MODEL_PATH_GNN = TARGET_MODEL_PATH  # Main 模块里也有一份引用，必须覆盖

    # 2. 覆盖测试列表
    Parameters.TEST_VEHICLE_COUNTS = TEST_SCENARIOS
    Main.TEST_VEHICLE_COUNTS = TEST_SCENARIOS

    # 3. 覆盖测试轮数
    Parameters.TEST_EPISODES_PER_COUNT = EPISODES_PER_SCENARIO
    Main.TEST_EPISODES_PER_COUNT = EPISODES_PER_SCENARIO

    # 4. 强制环境参数 (确保是大地图模式)
    Parameters.SCENE_SCALE_X = 1200
    Parameters.SCENE_SCALE_Y = 1200
    Parameters.USE_GNN_ENHANCEMENT = True
    Parameters.GNN_ARCH = "HYBRID"  # 确保架构一致

    # 5. 确保 Main 模块中的 device 设置正确
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Main.device = device

    # 检查模型文件是否存在
    if not os.path.exists(TARGET_MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {TARGET_MODEL_PATH}")
        print("   请先运行 run_smart_curriculum_v2.py 完成训练！")
        return

    # --- 启动测试 ---
    # Main.test() 包含了完整的物理计算、推理和结果保存逻辑
    try:
        Main.test()
        print(f"\n✅ 测试完成！请查看 training_results 目录下的 CSV 文件。")
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_evaluation()