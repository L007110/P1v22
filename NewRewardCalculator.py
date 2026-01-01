import numpy as np
from ChannelModel import global_channel_model
from logger import debug
import Parameters
from Parameters import (
    V2V_DELAY_THRESHOLD, V2I_LINK_POSITIONS, TRANSMITTDE_POWER,
    V2V_CHANNEL_BANDWIDTH, V2V_PACKET_SIZE_BITS, V2V_MIN_SNR_DB,
    GAIN_ANTENNA_T, V2I_TX_POWER, V2I_CAPACITY_THRESHOLD
)


class NewRewardCalculator:
    def __init__(self):
        self.channel_model = global_channel_model
        self.BEAM_ROLLOFF_EXPONENT = 2
        self.ANGLE_PER_DIRECTION = 10

        # [IEEE FIX 1] 物理层处理时延 (Processing Latency)
        self.PHY_MAC_LATENCY_OFFSET = 0.001  # 1ms

        # [IEEE FIX 2] 动态范围校准 (Normalization Bounds)
        self.stats = {
            'snr': {'min': -5.0, 'max': 30.0},
            'delay': {'min': 0.0, 'max': 0.005},  # 5ms 上限
            'v2i': {'min': 0.0, 'max': 10.0},
            'power': {'min': 0.0, 'max': 1.0}
        }

    def normalize_value(self, key, value):
        s = self.stats[key]
        norm = (value - s['min']) / (s['max'] - s['min'])
        return np.clip(norm, 0.0, 1.0)

    def _calculate_directional_gain(self, horizontal_dir, vertical_dir):
        theta_h = (horizontal_dir - 1) * self.ANGLE_PER_DIRECTION
        theta_v = (1 - vertical_dir) * self.ANGLE_PER_DIRECTION
        theta_h_rad = np.deg2rad(theta_h)
        theta_v_rad = np.deg2rad(theta_v)
        gain_h = np.cos(theta_h_rad) ** self.BEAM_ROLLOFF_EXPONENT
        gain_v = np.cos(theta_v_rad) ** self.BEAM_ROLLOFF_EXPONENT
        return gain_h * gain_v

    def calculate_delay(self, distance_3d, dqn_action, directional_gain=1.0, snr_linear=None):
        try:
            propagation_delay = distance_3d / 3e8
            if snr_linear is None: return 1.0

            if snr_linear > 0:
                data_rate = V2V_CHANNEL_BANDWIDTH * np.log2(1 + snr_linear)
                transmission_delay = V2V_PACKET_SIZE_BITS / (data_rate + 1e-9)
            else:
                transmission_delay = 1.0

            delay = transmission_delay + propagation_delay + self.PHY_MAC_LATENCY_OFFSET
        except:
            delay = 1.0
        return delay

    def _record_communication_metrics(self, dqn, delay, snr):
        dqn.delay_list.append(delay)
        dqn.snr_list.append(snr)

        is_delay_ok = 1 if delay <= V2V_DELAY_THRESHOLD else 0
        is_snr_ok = 1 if snr >= V2V_MIN_SNR_DB else 0
        success = 1 if (is_delay_ok and is_snr_ok) else 0

        if not hasattr(dqn, 'v2v_success_list'): dqn.v2v_success_list = []
        dqn.v2v_success_list.append(success)

        if not hasattr(dqn, 'v2v_delay_ok_list'): dqn.v2v_delay_ok_list = []
        if not hasattr(dqn, 'v2v_snr_ok_list'): dqn.v2v_snr_ok_list = []
        dqn.v2v_delay_ok_list.append(is_delay_ok)
        dqn.v2v_snr_ok_list.append(is_snr_ok)

    def calculate_complete_reward(self, dqn, vehicles, action, active_v2v_interferers=None):
        """
        核心奖励计算函数 (集成平滑惩罚 + V2I修正)
        """
        if not vehicles:
            self._record_communication_metrics(dqn, 1.0, -100.0)
            return 0.0, {}

        try:
            # --- 物理计算部分 ---
            closest_vehicle = vehicles[0]
            vehicle_loc = closest_vehicle.curr_loc
            distance_3d = self.channel_model.calculate_3d_distance(
                (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle_loc)

            beam_count = action[0] + 1
            h_dir, v_dir = action[1], action[2]
            power_ratio = (action[3] + 1) / 10.0

            directional_gain = self._calculate_directional_gain(h_dir, v_dir)
            total_tx_power = Parameters.TRANSMITTDE_POWER * power_ratio * beam_count * directional_gain * Parameters.GAIN_ANTENNA_T

            # A. V2V Link
            total_v2v_interference_W = 0.0
            if active_v2v_interferers:
                for interferer in active_v2v_interferers:
                    if interferer['tx_pos'] == vehicle_loc: continue
                    d = self.channel_model.calculate_3d_distance(interferer['tx_pos'], vehicle_loc)
                    pl, _, _ = self.channel_model.calculate_path_loss(d)
                    total_v2v_interference_W += interferer['power_W'] * (10 ** (-pl / 10))

            pl_sig, _, _ = self.channel_model.calculate_path_loss(distance_3d)
            signal_W = total_tx_power * (10 ** (-pl_sig / 10))
            noise_W = self.channel_model._calculate_noise_power(V2V_CHANNEL_BANDWIDTH)

            sinr_lin = signal_W / (total_v2v_interference_W + noise_W + 1e-20)
            snr_db = 10 * np.log10(sinr_lin)
            delay = self.calculate_delay(distance_3d, action, directional_gain, sinr_lin)

            # B. V2I Constraint (With Background Interference)
            background_interference_on_v2i = {}
            for idx, link in enumerate(Parameters.V2I_LINK_POSITIONS):
                rx_pos = link['rx']
                interf_sum = 0.0
                if active_v2v_interferers:
                    for interferer in active_v2v_interferers:
                        if interferer['tx_pos'] == vehicle_loc: continue
                        d_other = self.channel_model.calculate_3d_distance(interferer['tx_pos'], rx_pos)
                        pl_other, _, _ = self.channel_model.calculate_path_loss(d_other)
                        interf_sum += interferer['power_W'] * (10 ** (-pl_other / 10))
                background_interference_on_v2i[idx] = interf_sum

            min_v2i_capacity = float('inf')
            for idx, link in enumerate(Parameters.V2I_LINK_POSITIONS):
                rx_pos = link['rx']
                d_sig = self.channel_model.calculate_3d_distance(link['tx'], rx_pos)
                _, _, v2i_sig_W = self.channel_model.calculate_snr(Parameters.V2I_TX_POWER, d_sig)

                d_int = self.channel_model.calculate_3d_distance(vehicle_loc, rx_pos)
                pl_int, _, _ = self.channel_model.calculate_path_loss(d_int)
                my_interference_W = total_tx_power * (10 ** (-pl_int / 10))

                total_interference = my_interference_W + background_interference_on_v2i[idx] + noise_W
                v2i_sinr = v2i_sig_W / (total_interference + 1e-20)
                cap = np.log2(1 + v2i_sinr)
                if cap < min_v2i_capacity: min_v2i_capacity = cap

            current_v2i_capacity = min_v2i_capacity

            dqn.prev_snr = snr_db
            self._record_communication_metrics(dqn, delay, snr_db)

            # ==========================================
            # C. Reward Calculation (IEEE Standard + Smoothing)
            # ==========================================

            n_snr = self.normalize_value('snr', snr_db)
            n_delay = self.normalize_value('delay', delay)
            n_v2i = self.normalize_value('v2i', current_v2i_capacity)
            n_power = np.clip(power_ratio, 0.0, 1.0)

            reward = 0.0
            SNR_THRESHOLD = Parameters.V2V_MIN_SNR_DB
            V2I_THRESHOLD = Parameters.V2I_CAPACITY_THRESHOLD

            # --- Level 1: Reliability (Survival) with Smoothing ---
            if snr_db < SNR_THRESHOLD:
                # [New Smooth Penalty Logic]
                # 计算差值 (diff > 0)
                diff = SNR_THRESHOLD - snr_db

                # 使用 tanh 函数构造平滑惩罚:
                # 当 diff = 0 时, penalty = 1.0 (保持硬截断的基准)
                # 当 diff 增大时, penalty 平滑增加，最大趋近于 2.0 (1 + 1)
                # 除以 5.0 是为了控制斜率，避免梯度过大
                # penalty_factor = 1.0 + np.tanh(diff / 5.0)
                # [修改为] >>> 更温和的惩罚曲线
                # 解释：
                # 1. diff/10.0: 让梯度更平缓，模型不会因为 SNR 差一点点就收到剧烈反馈
                # 2. 0.5 + 0.5*tanh: 将惩罚范围从 [1, 2] 降低到 [0.5, 1.0]
                penalty_factor = 0.5 + 0.5 * np.tanh(diff / 10.0)
                reward -= penalty_factor

                # 保留梯度引导项 (给予微弱的正向反馈，指引优化方向)
                reward += 0.1 * Parameters.SNR_MULTIPLIER * n_snr

                breakdown = {
                    'raw_snr': snr_db, 'raw_delay': delay, 'raw_v2i': current_v2i_capacity,
                    'norm_snr': n_snr, 'norm_delay': n_delay, 'norm_v2i': n_v2i, 'norm_power': n_power,
                    'total_reward': reward
                }
                return reward, breakdown

            # --- Level 2: Constraint (V2I) ---
            reward += 1.0  # Base Reward for Connection

            if current_v2i_capacity < V2I_THRESHOLD:
                # Quadratic Penalty for Constraint Violation
                diff = V2I_THRESHOLD - current_v2i_capacity
                penalty = diff ** 2
                if penalty > 10.0: penalty = 10.0 + (diff - np.sqrt(10.0))  # 软截断

                reward -= 2.0 * Parameters.V2I_MULTIPLIER * penalty
            else:
                reward += 0.2 * Parameters.V2I_MULTIPLIER * n_v2i

            # --- Level 3: Efficiency ---
            SNR_SATURATION = 15.0
            if snr_db > SNR_SATURATION:
                overkill = (snr_db - SNR_SATURATION) / 10.0
                reward -= 0.2 * Parameters.POWER_MULTIPLIER * n_power * (1.0 + overkill)
            elif snr_db >= SNR_THRESHOLD:
                reward -= 0.05 * Parameters.POWER_MULTIPLIER * n_power

            reward += 0.3 * Parameters.DELAY_MULTIPLIER * (1.0 - n_delay)
            # 最后对总 Reward 进行 Clip，保证数值稳定性
            # 这一步是为了防止任何意外的数值（如 log(0) 产生的 -inf）
            reward = np.clip(reward, -10.0, 10.0)

            breakdown = {
                'raw_snr': snr_db, 'raw_delay': delay, 'raw_v2i': current_v2i_capacity,
                'norm_snr': n_snr, 'norm_delay': n_delay, 'norm_v2i': n_v2i, 'norm_power': n_power,
                'total_reward': reward
            }
            return reward, breakdown

        except Exception as e:
            self._record_communication_metrics(dqn, 1.0, -100.0)
            return -1.0, {}

    def get_csi_for_state(self, vehicle, dqn):
        if vehicle is None: return [0.0] * 5
        try:
            csi_info = self.channel_model.get_channel_state_info(
                (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle.curr_loc,
                tx_power=TRANSMITTDE_POWER, bandwidth=V2V_CHANNEL_BANDWIDTH
            )
            return [
                csi_info['distance_3d'], csi_info['path_loss_total_db'],
                csi_info['shadowing_db'], csi_info['snr_db'],
                getattr(dqn, 'prev_snr', 0.0)
            ]
        except:
            return [0.0] * 5

    def calculate_physics_state(self, action):
        """
        仅解析动作并计算物理参数 (功率, 增益等)，不计算奖励。
        用于在计算 Reward 之前先更新所有车辆的状态。
        """
        # 动作解析
        beam_count = action[0] + 1
        h_dir, v_dir = action[1], action[2]
        power_ratio = (action[3] + 1) / 10.0

        # 计算增益
        directional_gain = self._calculate_directional_gain(h_dir, v_dir)

        # 计算总发射功率 (Watts)
        total_tx_power = Parameters.TRANSMITTDE_POWER * power_ratio * beam_count * directional_gain * Parameters.GAIN_ANTENNA_T

        return total_tx_power, directional_gain


new_reward_calculator = NewRewardCalculator()