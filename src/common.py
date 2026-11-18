# 启用未来类型注解语法，支持类型注解前向引用
from __future__ import annotations
# JSON文件读写库，用于读取实验配置文件
import json
# 路径处理库，用于跨平台的文件路径操作
from pathlib import Path
# 数值计算库，用于数组运算、统计量计算
import numpy as np
# 数据处理库，用于数据集读取与表格运算
import pandas as pd
# 统计分布库，用于计算t分布95%置信区间
from scipy.stats import t

# 项目根目录：当前文件的上一级目录（src目录的父目录，即项目根目录）
ROOT = Path(__file__).resolve().parents[1]
# 实验配置文件完整路径
CONFIG_PATH = ROOT / "config" / "experiment_config.json"
# 数据集存放根目录
DATA_DIR = ROOT / "data"
# 结果总根目录
RESULT_DIR = ROOT / "results"
# 结果表格子目录
TABLE_DIR = RESULT_DIR / "tables"
# 结果图片子目录
FIGURE_DIR = RESULT_DIR / "figures"
# 数据检查结果子目录
CHECK_DIR = RESULT_DIR / "checks"
# 合成数据集文件完整路径
DATA_PATH = DATA_DIR / "synthetic_shm_dataset.csv"

# 数据集标签字段列表：类别ID、类别名称、损伤楼层、名义刚度损失率
LABEL_COLUMNS = [
    "label_id",
    "label_name",
    "damage_story",
    "nominal_stiffness_loss",
]

# 候选特征分组字典：共5组，合计28维特征
FEATURE_GROUPS = {
    # 第1组：固有频率，4维，1-4阶固有频率（单位：Hz）
    "Frequency (4)": [f"freq_{i}_hz" for i in range(1, 5)],
    # 第2组：模态振型，8维，前2阶模态 × 4层楼层的振型值
    "Mode shape (8)": [
        f"mode{mode}_floor{floor}"
        for mode in range(1, 3)
        for floor in range(1, 5)
    ],
    # 第3组：加速度统计量，8维，4层的加速度RMS + 4层的加速度峰值
    "Acceleration statistics (8)": [
        *[f"rms_acc_floor{i}" for i in range(1, 5)],
        *[f"peak_acc_floor{i}" for i in range(1, 5)],
    ],
    # 第4组：层间响应，4维，4层的层间RMS响应
    "Interstory response (4)": [
        f"interstory_rms_floor{i}" for i in range(1, 5)
    ],
    # 第5组：频带能量，4维，前4阶模态对应的频带能量
    "Band energy (4)": [f"band_energy_mode{i}" for i in range(1, 5)],
}

# 全部28个特征的列名列表：按分组顺序展开为一维列表
FEATURE_COLUMNS = [name for group in FEATURE_GROUPS.values() for name in group]

# 加载实验配置文件函数
# 输出：解析后的配置参数字典
def load_config() -> dict:
    # 读取JSON配置文件并解析为Python字典
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

# 创建所有输出目录函数
# 功能：确保数据、结果、表格、图片、检查目录都存在，不存在则递归创建
def ensure_output_dirs() -> None:
    for directory in (DATA_DIR, RESULT_DIR, TABLE_DIR, FIGURE_DIR, CHECK_DIR):
        # parents=True：递归创建父目录；exist_ok=True：目录已存在时不抛出错误
        directory.mkdir(parents=True, exist_ok=True)

# 生成运行模式对应的输出目录函数
# 输入：quick是否为快速模式
# 输出：表格目录、图片目录、检查目录
def mode_dirs(quick: bool) -> tuple[Path, Path, Path]:
    # 快速模式结果存到results/quick，完整模式存到results根目录
    base = RESULT_DIR / "quick" if quick else RESULT_DIR
    table_dir = base / "tables"  # 表格输出子目录
    figure_dir = base / "figures"  # 图片输出子目录
    check_dir = base / "checks"  # 检查结果子目录
    # 逐级创建所有目录
    for directory in (base, table_dir, figure_dir, check_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir, check_dir

# 加载合成数据集函数
# 输出：数据集DataFrame
def load_dataset() -> pd.DataFrame:
    # 数据集不存在时抛出错误，提示先生成数据集
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Data file is missing. Run: python -m src.generate_dataset"
        )
    # 读取CSV数据集，UTF-8带BOM编码兼容Excel打开
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# 计算95%置信区间函数
# 输入：数值序列
# 输出：均值、样本标准差、置信区间下限、置信区间上限
def ci95(values) -> tuple[float, float, float, float]:
    # 转换为浮点数组
    array = np.asarray(values, dtype=float)
    # 计算样本均值
    mean = float(array.mean())
    # 计算样本标准差（ddof=1为无偏估计），样本量为1时标准差置0
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    # 计算95%置信区间半幅：t分布0.975分位数 × 标准差 / 根号样本量
    half = (
        float(t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
        if len(array) > 1
        else 0.0
    )
    # 返回均值、标准差、置信下限、置信上限
    return mean, sd, mean - half, mean + half

# 分组汇总统计函数
# 输入：数据框、分组列名、指标列名列表
# 输出：每个分组的指标均值、标准差、95%置信区间汇总表
def summarize(
    frame: pd.DataFrame, group_column: str, metrics: list[str]
) -> pd.DataFrame:
    rows = []
    # 按指定列分组遍历，sort=False保持原顺序
    for name, subset in frame.groupby(group_column, sort=False):
        # 初始化行：分组名、重复次数
        row = {group_column: name, "n_repeats": len(subset)}
        # 逐个指标计算统计量并填入行
        for metric in metrics:
            mean, sd, low, high = ci95(subset[metric])
            row[f"{metric}_mean"] = mean       # 指标均值
            row[f"{metric}_sd"] = sd           # 指标标准差
            row[f"{metric}_ci95_low"] = low    # 指标95%置信区间下限
            row[f"{metric}_ci95_high"] = high   # 指标95%置信区间上限
        rows.append(row)
    # 转换为DataFrame返回
    return pd.DataFrame(rows)

# 根据类别ID生成类别名称函数
# 输入：类别ID（0-12）
# 输出：类别名称字符串
def class_name(class_id: int) -> str:
    # 类别0对应完好状态
    if class_id == 0:
        return "intact"
    # 损伤程度等级：10%、20%、30%
    levels = (10, 20, 30)
    index = class_id - 1
    # 计算损伤楼层：index除以等级数取整 +1，得到1-4层
    story = index // len(levels) + 1
    # 计算损伤程度：index对等级数取余，得到对应等级值
    level = levels[index % len(levels)]
    # 拼接为"storyX_lossXpct"格式的名称
    return f"story{story}_loss{level}pct"