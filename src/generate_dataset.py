# 启用未来类型注解语法，支持类型注解前向引用
from __future__ import annotations
# 基础数学运算库，用于三角函数、开方、对数等计算
import math
# 数值计算库，用于数组、矩阵运算
import numpy as np
# 数据处理库，用于生成最终数据集表格
import pandas as pd
# 从同包common模块导入预定义常量与工具函数
from .common import DATA_PATH, FEATURE_COLUMNS, class_name, ensure_output_dirs, load_config


# 自定义伪随机数生成器：复刻.NET Framework随机数算法，保证不同环境下相同种子生成完全一致的随机序列
class DotNetRandom:
    # 初始化随机数发生器，输入种子值构建内部状态
    def __init__(self, seed: int):
        mbig = 2147483647       # 随机数模数上限（大素数）
        mseed = 161803398        # 种子偏移常量
        # 计算种子偏移，处理负种子场景
        subtraction = 2147483647 if seed == -2147483648 else abs(seed)
        mj = mseed - subtraction
        if mj < 0:
            mj += mbig
        # 初始化56元素的种子数组（.NET随机数发生器的内部状态池）
        self.seed_array = [0] * 56
        self.seed_array[55] = mj
        mk = 1
        # 第一轮填充种子数组，建立初始内部状态
        for i in range(1, 55):
            ii = (21 * i) % 55
            self.seed_array[ii] = mk
            mk = mj - mk
            if mk < 0:
                mk += mbig
            mj = self.seed_array[ii]
        # 4轮洗牌，充分打乱随机数池
        for _ in range(4):
            for i in range(1, 56):
                self.seed_array[i] -= self.seed_array[1 + (i + 30) % 55]
                if self.seed_array[i] < 0:
                    self.seed_array[i] += mbig
        # 两个内部采样指针
        self.inext = 0
        self.inextp = 21

    # 内部采样方法：生成下一个内部整数随机数，更新状态池
    def internal_sample(self) -> int:
        self.inext += 1
        if self.inext >= 56:
            self.inext = 1
        self.inextp += 1
        if self.inextp >= 56:
            self.inextp = 1
        # 双指针差值生成新随机数
        value = self.seed_array[self.inext] - self.seed_array[self.inextp]
        if value == 2147483647:
            value -= 1
        if value < 0:
            value += 2147483647
        self.seed_array[self.inext] = value
        return value

    # 生成[0, 1)区间的浮点随机数
    def random(self) -> float:
        return self.internal_sample() * (1.0 / 2147483647)

    # 生成[0, maximum)区间的整数随机数
    def next_int(self, maximum: int) -> int:
        return int(self.random() * maximum)


# Box-Muller变换：生成标准正态分布（高斯）随机数
def gaussian(rng: DotNetRandom) -> float:
    # 生成两个(0,1]均匀随机数，避开0点
    u1 = 1.0 - rng.random()
    u2 = 1.0 - rng.random()
    # Box-Muller公式生成标准高斯随机数
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# Fisher-Yates洗牌算法：原地打乱列表元素顺序
def shuffle(values: list, rng: DotNetRandom) -> None:
    # 从后往前遍历，逐个与随机位置交换
    for i in range(len(values) - 1, 0, -1):
        j = rng.next_int(i + 1)
        values[i], values[j] = values[j], values[i]


# 构建四层剪切型框架的刚度矩阵（三对角对称结构）
# 输入：各层层间刚度列表
# 输出：4×4刚度矩阵
def build_stiffness(story_stiffness: list[float]) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=float)
    for story in range(4):
        upper = story
        # 对角元累加当前层刚度
        matrix[upper, upper] += story_stiffness[story]
        if story > 0:
            lower = story - 1
            # 下一层对角元累加当前层刚度
            matrix[lower, lower] += story_stiffness[story]
            # 非对角元为负的层间刚度（剪切框架三对角特性）
            matrix[upper, lower] -= story_stiffness[story]
            matrix[lower, upper] -= story_stiffness[story]
    return matrix


# Jacobi迭代法：求解4阶实对称矩阵的特征值与特征向量（用于模态分析）
# 输入：4×4实对称矩阵
# 输出：从小到大排序的特征值数组、对应特征向量矩阵（列向量为各阶振型）
def jacobi_eigen(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = matrix.copy()
    vectors = np.eye(4, dtype=float)  # 特征向量初始化为单位矩阵

    # 最多执行100次Jacobi旋转变换
    for _ in range(100):
        p = 0
        q = 1
        maximum = abs(a[p, q])
        # 寻找绝对值最大的非对角元
        for i in range(4):
            for j in range(i + 1, 4):
                value = abs(a[i, j])
                if value > maximum:
                    maximum = value
                    p = i
                    q = j
        # 最大非对角元小于收敛阈值，退出迭代
        if maximum < 1e-10:
            break

        # 计算最优旋转角度
        theta = 0.5 * math.atan2(2.0 * a[p, q], a[q, q] - a[p, p])
        cosine = math.cos(theta)
        sine = math.sin(theta)

        # 更新p、q位置的对角元
        app = (
            cosine * cosine * a[p, p]
            - 2.0 * sine * cosine * a[p, q]
            + sine * sine * a[q, q]
        )
        aqq = (
            sine * sine * a[p, p]
            + 2.0 * sine * cosine * a[p, q]
            + cosine * cosine * a[q, q]
        )
        a[p, p] = app
        a[q, q] = aqq
        a[p, q] = 0.0
        a[q, p] = 0.0

        # 更新p、q行/列的其余元素
        for row in range(4):
            if row == p or row == q:
                continue
            arp = a[row, p]
            arq = a[row, q]
            a[row, p] = cosine * arp - sine * arq
            a[p, row] = a[row, p]
            a[row, q] = sine * arp + cosine * arq
            a[q, row] = a[row, q]

        # 同步更新特征向量矩阵
        for row in range(4):
            vrp = vectors[row, p]
            vrq = vectors[row, q]
            vectors[row, p] = cosine * vrp - sine * vrq
            vectors[row, q] = sine * vrp + cosine * vrq

    # 提取对角元作为特征值
    values = np.diag(a).copy()
    # 按特征值从小到大稳定排序
    order = np.argsort(values, kind="stable")
    return values[order], vectors[:, order]


# 模态坐标下的受迫加速度响应仿真
# 输入：模态振型矩阵、固有圆频率、随机数生成器、配置字典
# 输出：4层×时间步的加速度时程序列
def simulate_acceleration(
    phi: np.ndarray, omega: np.ndarray, rng: DotNetRandom, config: dict
) -> np.ndarray:
    data_config = config["data"]
    uncertainty = config["uncertainty"]
    steps = int(data_config["time_steps"])          # 时间步总数
    dt = float(data_config["time_step_seconds"])    # 单步时间长度
    # 在配置范围内随机生成阻尼比
    zeta = float(uncertainty["damping_ratio_min"]) + (
        float(uncertainty["damping_ratio_max"])
        - float(uncertainty["damping_ratio_min"])
    ) * rng.random()

    # 模态坐标下的位移、速度、加速度
    q = np.zeros(4, dtype=float)
    qd = np.zeros(4, dtype=float)
    qdd = np.zeros(4, dtype=float)

    # 各楼层荷载分布形状，叠加微小随机扰动
    load_shape = np.array(
        [0.75 + 0.08 * i + 0.04 * gaussian(rng) for i in range(4)],
        dtype=float,
    )
    # 激励总幅值，在配置范围内随机取值
    amplitude = 2600.0 * (
        float(uncertainty["excitation_scale_min"])
        + (
            float(uncertainty["excitation_scale_max"])
            - float(uncertainty["excitation_scale_min"])
        )
        * rng.random()
    )
    # 两个简谐激励分量的随机初始相位
    phase1 = 2.0 * math.pi * rng.random()
    phase2 = 2.0 * math.pi * rng.random()
    # 两个简谐激励分量的随机频率
    frequency1 = 0.45 + 0.35 * rng.random()
    frequency2 = 1.20 + 0.70 * rng.random()

    acceleration = np.zeros((4, steps), dtype=float)
    random_ratio = float(uncertainty["excitation_random_ratio"])  # 随机激励分量占比
    noise_std = float(uncertainty["measurement_noise_std"])    # 测量噪声标准差

    # 时间步进欧拉积分
    for step in range(steps):
        time = step * dt
        # 外部激励：双简谐分量 + 随机噪声分量
        excitation = amplitude * (
            0.55 * math.sin(2.0 * math.pi * frequency1 * time + phase1)
            + 0.30 * math.sin(2.0 * math.pi * frequency2 * time + phase2)
            + random_ratio * gaussian(rng)
        )

        # 求解模态坐标下各阶模态加速度（有阻尼受迫运动方程）
        for mode in range(4):
            modal_force = 0.0
            for floor in range(4):
                modal_force += phi[floor, mode] * load_shape[floor] * excitation
            qdd[mode] = (
                modal_force
                - 2.0 * zeta * omega[mode] * qd[mode]
                - omega[mode] * omega[mode] * q[mode]
            )

        # 欧拉法更新模态速度、位移
        for mode in range(4):
            qd[mode] += qdd[mode] * dt
            q[mode] += qd[mode] * dt

        # 转换回物理坐标得到各楼层加速度，叠加测量噪声
        for floor in range(4):
            value = 0.0
            for mode in range(4):
                value += phi[floor, mode] * qdd[mode]
            acceleration[floor, step] = value + noise_std * gaussian(rng)

    return acceleration


# 计算时域信号在目标频率附近的频带能量
# 输入：时域信号、采样频率、目标频率
# 输出：对应频带的能量值
def band_energy(signal: np.ndarray, fs: float, target_hz: float) -> float:
    count = len(signal)
    center = int(round(target_hz * count / fs))  # 目标频率对应的频点索引
    lower = max(1, center - 1)    # 频带下限索引
    upper = min(count // 2, center + 1)  # 频带上限索引
    energy = 0.0

    # 遍历频带内每个频点，计算离散傅里叶幅值并累加能量
    for frequency_bin in range(lower, upper + 1):
        real = 0.0
        imaginary = 0.0
        for index in range(count):
            angle = -2.0 * math.pi * frequency_bin * index / count
            real += signal[index] * math.cos(angle)
            imaginary += signal[index] * math.sin(angle)
        energy += (real * real + imaginary * imaginary) / count

    return energy


# 提取全部28维损伤识别特征
# 输入：振型矩阵、固有圆频率、频率数组、随机数生成器、配置
# 输出：28个特征值的列表
def extract_features(
    phi: np.ndarray,
    omega: np.ndarray,
    frequency_hz: np.ndarray,
    rng: DotNetRandom,
    config: dict,
) -> list[float]:
    features = list(frequency_hz)  # 第1-4维：固有频率

    # 第5-12维：前2阶模态振型，按顶层符号统一方向并归一化
    for mode in range(2):
        # 按顶层振型符号确定正负号，保证振型方向一致
        sign = 1.0 if phi[3, mode] >= 0.0 else -1.0
        # 按该阶最大绝对值归一化振型
        maximum = max(1e-12, max(abs(phi[floor, mode]) for floor in range(4)))
        for floor in range(4):
            features.append(sign * phi[floor, mode] / maximum)

    # 仿真得到各楼层加速度时程
    acceleration = simulate_acceleration(phi, omega, rng, config)
    top = acceleration[3, :]  # 顶层加速度作为参考

    # 第13-16维：各层加速度RMS（均方根值）
    for floor in range(4):
        features.append(math.sqrt(float(np.sum(acceleration[floor, :] ** 2)) / len(top)))
    # 第17-20维：各层加速度峰值
    for floor in range(4):
        features.append(float(np.max(np.abs(acceleration[floor, :]))))

    # 第21-24维：各层层间响应RMS
    for floor in range(4):
        below = np.zeros_like(top) if floor == 0 else acceleration[floor - 1, :]
        difference = acceleration[floor, :] - below
        features.append(math.sqrt(float(np.sum(difference**2)) / len(top)))

    fs = float(config["data"]["sampling_frequency_hz"])
    # 第25-28维：前4阶模态对应频带的能量（对数尺度）
    for mode in range(4):
        features.append(math.log10(band_energy(top, fs, frequency_hz[mode]) + 1e-14))

    return features


# 类别ID解码：将0-12的类别ID转换为损伤楼层和损伤程度
# 输入：类别ID、损伤程度列表
# 输出：损伤楼层（0代表完好）、名义损伤程度值
def decode_class(class_id: int, levels: list[float]) -> tuple[int, float]:
    if class_id == 0:
        return 0, 0.0
    index = class_id - 1
    # 整除得到楼层序号，取余得到损伤程度等级
    return index // len(levels) + 1, levels[index % len(levels)]


# 生成完整合成数据集
# 输出：包含标签列与特征列的DataFrame
def generate(actual_damage_records: list | None = None) -> pd.DataFrame:
    config = load_config()
    structure = config["structure"]
    uncertainty = config["uncertainty"]
    samples_per_class = int(config["data"]["samples_per_class"])  # 每类样本数量
    levels = [float(value) for value in structure["damage_levels"]]     # 损伤程度列表
    base_stiffness = [float(value) for value in structure["story_stiffness"]] # 基准各层刚度
    base_mass = [float(value) for value in structure["story_mass"]]           # 基准各层质量

    # 初始化固定种子随机数生成器，保证数据集可复现
    rng = DotNetRandom(int(config["data"]["generation_seed"]))

    rows = []
    # 遍历全部13个损伤类别
    for class_id in range(1 + 4 * len(levels)):
        story, severity = decode_class(class_id, levels)

        # 每个类别生成指定数量的样本
        for _ in range(samples_per_class):
            stiffness = []
            mass = []
            # 对每层刚度、质量施加高斯随机扰动（模拟参数不确定性）
            for i in range(4):
                stiffness.append(
                    base_stiffness[i]
                    * (
                        1.0
                        + float(uncertainty["stiffness_std_ratio"])
                        * gaussian(rng)
                    )
                )
                mass.append(
                    base_mass[i]
                    * (
                        1.0
                        + float(uncertainty["mass_std_ratio"])
                        * gaussian(rng)
                    )
                )

            # 非完好类别：对指定楼层施加刚度折减损伤
            if story > 0:
                # 名义损伤程度叠加随机扰动
                actual = severity + float(uncertainty["damage_std"]) * gaussian(rng)
                # 钳制损伤程度在上下限范围内
                actual = max(
                    float(uncertainty["damage_lower_bound"]),
                    min(float(uncertainty["damage_upper_bound"]), actual),
                )
                # 对应楼层刚度乘以（1-实际损伤程度）
                stiffness[story - 1] *= 1.0 - actual

            # 可选审计记录：只记录数值，不增加随机抽样或改变样本生成顺序。
            if actual_damage_records is not None:
                actual_damage_records.append({
                    "label_id": class_id, "damage_story": story,
                    "nominal_stiffness_loss": severity,
                    "actual_stiffness_loss": actual if story > 0 else 0.0,
                })

            # 构建刚度矩阵
            stiffness_matrix = build_stiffness(stiffness)

            # 构建质量归一化的动力矩阵（广义特征值问题形式）
            dynamic = np.zeros((4, 4), dtype=float)
            for i in range(4):
                for j in range(4):
                    dynamic[i, j] = stiffness_matrix[i, j] / math.sqrt(mass[i] * mass[j])

            # 求解特征值与特征向量
            eigenvalues, eigenvectors = jacobi_eigen(dynamic)
            # 计算各阶固有圆频率，避免负值
            omega = np.array(
                [math.sqrt(max(value, 1e-9)) for value in eigenvalues], dtype=float
            )
            # 转换为Hz单位的固有频率
            frequency_hz = omega / (2.0 * math.pi)

            # 转换为物理坐标下的质量归一化振型
            phi = np.zeros((4, 4), dtype=float)
            for row in range(4):
                for column in range(4):
                    phi[row, column] = eigenvectors[row, column] / math.sqrt(mass[row])

            # 提取28维候选特征
            features = extract_features(phi, omega, frequency_hz, rng, config)

            # 组装样本行：4个标签字段 + 28个特征字段
            rows.append(
                [class_id, class_name(class_id), story, severity, *features]
            )

    # 整体打乱所有样本顺序，避免类别按顺序排列
    shuffle(rows, rng)

    # 构建DataFrame，列名对应标签+特征
    return pd.DataFrame(rows, columns=[
        "label_id",
        "label_name",
        "damage_story",
        "nominal_stiffness_loss",
        *FEATURE_COLUMNS,
    ])


# 主函数入口
def main() -> None:
    ensure_output_dirs()  # 确保所有输出目录存在
    data = generate()      # 生成完整数据集
    # 保存为CSV文件，不带行索引，高精度浮点数格式
    data.to_csv(DATA_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    # 控制台输出生成信息
    print(f"Rows: {len(data)}")
    print(f"Columns: {data.shape[1]}")
    print(f"Saved: {DATA_PATH.relative_to(DATA_PATH.parents[1])}")


# 脚本直接运行时执行主函数
if __name__ == "__main__":
    main()
