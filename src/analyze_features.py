# 启用未来类型注解语法，支持类型注解的前向引用
from __future__ import annotations
# 命令行参数解析库，用于接收 --quick 快速运行参数
import argparse
# 数值计算库，用于数组运算、随机置乱、统计量计算
import numpy as np
# 数据处理库，用于表格构建、相关矩阵计算、结果导出
import pandas as pd
# 随机森林分类器，作为置乱分析的基准分类模型
from sklearn.ensemble import RandomForestClassifier
# 准确率评价指标
from sklearn.metrics import accuracy_score
# 数据集划分函数，用于分层抽样划分训练/测试集
from sklearn.model_selection import train_test_split

# 从项目公共模块导入预定义常量与工具函数
from .common import (
    FEATURE_COLUMNS,    # 全部28个特征的列名列表
    FEATURE_GROUPS,     # 5组特征的分组字典
    ci95,               # 95%置信区间计算函数
    load_config,        # 加载实验配置文件的函数
    load_dataset,       # 加载数据集CSV的函数
    mode_dirs,          # 创建输出目录的函数
)


# ===================== 1. Pearson相关性分析函数 =====================
# 功能：计算所有响应类特征的皮尔逊线性相关系数，输出特征对明细与整体汇总统计
# 输入：完整数据集DataFrame
# 输出：特征对相关明细、相关性汇总统计
def correlation_analysis(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 提取所有响应类特征：加速度统计(8维) + 层间响应(4维) + 频带能量(4维)，共16个特征
    response_features = (
        FEATURE_GROUPS["Acceleration statistics (8)"]
        + FEATURE_GROUPS["Interstory response (4)"]
        + FEATURE_GROUPS["Band energy (4)"]
    )

    # 计算16个响应特征的Pearson相关系数矩阵
    matrix = data[response_features].corr(method="pearson")

    rows = []
    # 双层遍历生成所有两两特征对（不重复），记录相关系数与绝对值
    for index, left in enumerate(response_features):
        # 只遍历index之后的特征，避免重复计算对称的相关对
        for right in response_features[index + 1 :]:
            value = float(matrix.loc[left, right])
            rows.append(
                {
                    "feature_1": left,          # 第一个特征名
                    "feature_2": right,         # 第二个特征名
                    "pearson_r": value,         # 皮尔逊相关系数r
                    "abs_r": abs(value),        # 相关系数绝对值
                }
            )

    # 将所有特征对转换为DataFrame，并按相关系数绝对值降序排列
    pairs = pd.DataFrame(rows).sort_values(
        "abs_r", ascending=False, ignore_index=True
    )

    # 计算同楼层的「加速度RMS」与「加速度峰值」的相关系数（1-4层各一组）
    same_floor = np.array(
        [
            matrix.loc[f"rms_acc_floor{i}", f"peak_acc_floor{i}"]
            for i in range(1, 5)
        ],
        dtype=float,
    )

    # 生成相关性整体汇总统计
    summary = pd.DataFrame(
        [
            {
                "response_feature_count": len(response_features),              # 响应特征总数
                "pair_count": len(pairs),                                     # 两两特征对总数
                "median_absolute_correlation": float(pairs["abs_r"].median()), # 绝对相关系数中位数
                "mean_absolute_correlation": float(pairs["abs_r"].mean()),     # 绝对相关系数均值
                "pairs_abs_r_ge_0_80": int((pairs["abs_r"] >= 0.80).sum()),    # |r|≥0.8的特征对数量
                "pairs_abs_r_ge_0_90": int((pairs["abs_r"] >= 0.90).sum()),    # |r|≥0.9的特征对数量
                "mean_same_floor_rms_peak_correlation": float(same_floor.mean()),   # 同楼层RMS-峰值平均相关
                "minimum_same_floor_rms_peak_correlation": float(same_floor.min()), # 同楼层RMS-峰值最小相关
                "maximum_same_floor_rms_peak_correlation": float(same_floor.max()), # 同楼层RMS-峰值最大相关
                "strongest_pair_feature_1": str(pairs.iloc[0]["feature_1"]),       # 最强相关对的特征1
                "strongest_pair_feature_2": str(pairs.iloc[0]["feature_2"]),       # 最强相关对的特征2
                "strongest_pair_r": float(pairs.iloc[0]["pearson_r"]),             # 最强相关对的r值
            }
        ]
    )

    return pairs, summary


# ===================== 2. 特征组联合置乱分析函数 =====================
# 原理：对每个特征组整体执行同一随机顺序打乱，保持组内特征的关联关系，仅破坏该组与标签的对应；
#       准确率下降越大，说明该特征组对损伤识别的贡献越强。
# 输入：完整数据集、是否快速模式
# 输出：置乱明细、置乱汇总统计
def permutation_analysis(
    data: pd.DataFrame, quick: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 加载实验配置
    config = load_config()
    # 读取验证用的随机种子列表
    seeds = [int(value) for value in config["validation"]["seeds"]]
    # 读取随机森林的决策树数量
    trees = int(config["baselines"]["random_forest_trees"])
    # 读取每个特征组的置乱重复次数
    permutations = int(config["analysis"]["permutation_repeats"])

    # 快速模式下缩减规模：只跑1个种子、50棵树、3次置乱，用于检查流程
    if quick:
        seeds = seeds[:1]
        trees = 50
        permutations = 3

    # 提取全部特征与标签
    x = data[FEATURE_COLUMNS]
    y = data["label_id"].to_numpy(dtype=int)

    rows = []
    # 循环每个随机种子，执行多次划分验证
    for repeat, seed in enumerate(seeds, start=1):
        # 分层抽样划分训练/测试集，保持类别比例一致
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=float(config["validation"]["test_fraction"]),
            random_state=seed,
            stratify=y,
        )

        # 构建并训练随机森林基准模型
        model = RandomForestClassifier(
            n_estimators=trees,                  # 决策树数量
            max_features="sqrt",                  # 每棵树特征数为总特征数开方
            class_weight="balanced_subsample",    # 类别平衡采样
            random_state=seed,                    # 固定随机种子
            n_jobs=-1,                            # 调用全部CPU核心并行训练
        )
        model.fit(x.iloc[train_idx], y[train_idx])

        # 提取测试集特征与标签
        x_test = x.iloc[test_idx].copy()
        y_test = y[test_idx]
        # 计算原始未置乱的基准准确率
        baseline = accuracy_score(y_test, model.predict(x_test))

        # 对每个特征组分别执行联合置乱
        for group_index, (group, columns) in enumerate(FEATURE_GROUPS.items()):
            drops = []
            # 为每个特征组生成独立的随机数生成器，种子偏移保证各组置乱不同
            rng = np.random.default_rng(seed + 1009 * (group_index + 1))

            # 执行指定次数的置乱重复
            for _ in range(permutations):
                # 生成随机排列顺序
                order = rng.permutation(len(x_test))
                permuted = x_test.copy()
                # 该组所有特征列按同一顺序打乱，保持组内特征的关联关系
                permuted.loc[:, columns] = x_test[columns].to_numpy()[order]
                # 计算置乱后的准确率
                score = accuracy_score(y_test, model.predict(permuted))
                # 记录本次置乱的准确率下降值（基准 - 置乱后）
                drops.append(baseline - score)

            # 记录当前种子、当前特征组的置乱结果
            rows.append(
                {
                    "repeat": repeat,                        # 重复序号
                    "seed": seed,                           # 随机种子
                    "group": group,                         # 特征组名称
                    "baseline_accuracy": baseline,          # 基准准确率
                    "mean_accuracy_drop": float(np.mean(drops)), # 多次置乱的平均准确率下降
                    "sd_across_permutations": float(np.std(drops, ddof=1))
                    if len(drops) > 1
                    else 0.0,                               # 多次置乱间的标准差
                }
            )

    # 置乱明细结果转换为DataFrame
    detail = pd.DataFrame(rows)

    summary_rows = []
    # 按特征组分组，计算每组的汇总统计与95%置信区间
    for group, subset in detail.groupby("group", sort=False):
        mean, sd, low, high = ci95(subset["mean_accuracy_drop"])
        summary_rows.append(
            {
                "group": group,                         # 特征组名称
                "n_splits": len(subset),               # 划分次数
                "accuracy_drop_mean": mean,             # 准确率下降均值
                "accuracy_drop_sd": sd,                 # 准确率下降标准差
                "accuracy_drop_ci95_low": low,          # 95%置信区间下限
                "accuracy_drop_ci95_high": high,         # 95%置信区间上限
            }
        )

    # 按准确率下降幅度降序排列，下降越大说明该组贡献越强
    summary = pd.DataFrame(summary_rows).sort_values(
        "accuracy_drop_mean", ascending=False, ignore_index=True
    )

    return detail, summary


# ===================== 3. 主运行函数 =====================
# 功能：执行完整的特征冗余分析，保存所有结果文件并打印汇总
def run(quick: bool) -> None:
    # 加载数据集
    data = load_dataset()
    # 创建输出目录，获取表格输出目录路径
    table_dir, _, _ = mode_dirs(quick)

    # 执行Pearson相关性分析
    pairs, correlation_summary = correlation_analysis(data)
    # 执行特征组联合置乱分析
    detail, permutation_summary = permutation_analysis(data, quick)

    # 保存响应特征相关对明细
    pairs.to_csv(
        table_dir / "response_feature_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 保存相关性汇总统计
    correlation_summary.to_csv(
        table_dir / "correlation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 保存特征组置乱明细
    detail.to_csv(
        table_dir / "group_permutation_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 保存特征组置乱汇总统计
    permutation_summary.to_csv(
        table_dir / "group_permutation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 控制台打印两份汇总结果，方便快速查看
    print(correlation_summary.to_string(index=False))
    print(permutation_summary.to_string(index=False))


# ===================== 4. 命令行入口 =====================
def main() -> None:
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加--quick快速运行参数
    parser.add_argument("--quick", action="store_true")
    # 解析命令行参数
    arguments = parser.parse_args()
    # 调用主运行函数
    run(arguments.quick)


# 脚本直接运行时的入口点
if __name__ == "__main__":
    main()
