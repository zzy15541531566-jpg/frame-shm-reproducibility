# 启用未来类型注解语法，支持类型注解的前向引用
from __future__ import annotations
# JSON库，用于将检查结果中的观测值/期望值序列化为字符串存储
import json
# 数值计算库，用于有限值判断、数组逻辑运算
import numpy as np
# 数据处理库，用于数据集读取、分组统计、结果表格导出
import pandas as pd

# 从项目公共模块导入预定义常量与工具函数
from .common import (
    FEATURE_COLUMNS,    # 全部28个特征的列名列表
    LABEL_COLUMNS,      # 4个标签字段的列名列表
    load_config,        # 加载实验配置文件的函数
    load_dataset,       # 加载合成数据集CSV的函数
    mode_dirs,          # 创建输出目录的函数
)


# 数据集检查核心函数：执行所有检查项，返回检查结果记录列表
# 输入：完整数据集DataFrame
# 输出：每项检查的名称、是否通过、观测值、期望值的字典列表
def evaluate_checks(data: pd.DataFrame) -> list[dict]:
    # 加载实验配置，用于计算预期样本数
    config = load_config()

    # 计算预期总行数：(1类完好 + 4层×3级损伤) × 每类样本数 = 13类×每类样本数
    expected_rows = (
        1 + 4 * len(config["structure"]["damage_levels"])
    ) * int(config["data"]["samples_per_class"])

    # 提取所有特征值转为浮点数组，用于数值合法性检查
    feature_values = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    # 统计每个类别的样本数量，用于类别平衡检查
    class_counts = data.groupby("label_id").size()

    # 定义所有检查项，每项格式：(检查名称, 是否通过, 观测值, 期望值)
    checks = [
        # 检查1：总行数是否符合预期
        ("row_count", len(data) == expected_rows, len(data), expected_rows),

        # 检查2：列顺序是否正确（标签列在前，特征列在后）
        (
            "column_order",
            data.columns.tolist() == LABEL_COLUMNS + FEATURE_COLUMNS,
            data.shape[1],
            len(LABEL_COLUMNS) + len(FEATURE_COLUMNS),
        ),

        # 检查3：类别数量是否为13类（完好+4层×3级损伤）
        ("class_count", data["label_id"].nunique() == 13, data["label_id"].nunique(), 13),

        # 检查4：类别是否平衡（每类样本数一致）
        (
            "balanced_classes",
            class_counts.nunique() == 1,
            class_counts.to_dict(),
            int(config["data"]["samples_per_class"]),
        ),

        # 检查5：是否存在缺失值
        ("missing_values", not data.isna().any().any(), int(data.isna().sum().sum()), 0),

        # 检查6：所有特征值是否为有限值（无inf、nan）
        (
            "finite_features",
            bool(np.isfinite(feature_values).all()),
            int((~np.isfinite(feature_values)).sum()),
            0,
        ),

        # 检查7：是否存在重复行
        ("duplicate_rows", not data.duplicated().any(), int(data.duplicated().sum()), 0),

        # 检查8：所有固有频率是否都大于0（符合物理意义，频率不能为负或零）
        (
            "positive_frequencies",
            bool((data[[f"freq_{i}_hz" for i in range(1, 5)]] > 0).all().all()),
            None,
            True,
        ),

        # 检查9：固有频率是否按阶递增（f1<f2<f3<f4，符合模态物理规律）
        (
            "ordered_frequencies",
            bool(
                (data["freq_1_hz"] < data["freq_2_hz"]).all()
                and (data["freq_2_hz"] < data["freq_3_hz"]).all()
                and (data["freq_3_hz"] < data["freq_4_hz"]).all()
            ),
            None,
            True,
        ),

        # 检查10：各层加速度RMS是否非负（均方根值数学上必然≥0）
        (
            "nonnegative_rms",
            bool(
                (
                    data[[f"rms_acc_floor{i}" for i in range(1, 5)]]
                    >= 0
                )
                .all()
                .all()
            ),
            None,
            True,
        ),

        # 检查11：各层加速度峰值是否不小于RMS（峰值必然≥均方根值，符合统计定义）
        (
            "peak_not_below_rms",
            all(
                bool(
                    (
                        data[f"peak_acc_floor{i}"]
                        >= data[f"rms_acc_floor{i}"]
                    ).all()
                )
                for i in range(1, 5)
            ),
            None,
            True,
        ),
    ]

    # 将检查结果格式化为统一的字典列表，观测值和期望值转为JSON字符串便于存储
    return [
        {
            "check_item": name,                # 检查项名称
            "passed": bool(passed),            # 是否通过
            "observed": json.dumps(observed, ensure_ascii=False), # 观测值
            "expected": json.dumps(expected, ensure_ascii=False), # 期望值
        }
        for name, passed, observed, expected in checks
    ]


# 主函数：加载数据、执行检查、输出报告、异常退出
def main() -> None:
    # 加载合成数据集
    data = load_dataset()
    # 创建输出目录，获取检查结果目录路径
    _, _, check_dir = mode_dirs(False)

    # 执行所有数据检查
    records = evaluate_checks(data)
    # 转换检查结果为DataFrame
    report = pd.DataFrame(records)

    # 保存检查报告CSV
    report.to_csv(check_dir / "dataset_checks.csv", index=False, encoding="utf-8-sig")

    # 保存每类样本数量统计表
    data.groupby(["label_id", "label_name"], as_index=False).size().rename(
        columns={"size": "sample_count"}
    ).to_csv(check_dir / "class_counts.csv", index=False, encoding="utf-8-sig")

    # 保存所有特征的描述性统计（均值、标准差、最值等）
    data[FEATURE_COLUMNS].describe().T.to_csv(
        check_dir / "feature_descriptive_statistics.csv", encoding="utf-8-sig"
    )

    # 控制台逐项打印检查结果，PASS/FAIL高亮标识
    for item in records:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['check_item']}")

    # 任意一项检查不通过，以错误码1退出程序，终止后续流程
    if not all(item["passed"] for item in records):
        raise SystemExit(1)


# 脚本直接运行时的入口点
if __name__ == "__main__":
    main()
