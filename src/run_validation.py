# 启用未来类型注解语法，支持类型注解的前向引用
from __future__ import annotations
# 命令行参数解析库，用于接收 --quick 等运行控制参数
import argparse
# JSON读写库，用于读取实验配置、输出运行环境信息
import json
# 系统平台信息库，用于记录运行的操作系统版本
import platform
# 系统交互库，用于获取Python版本、强制刷新标准输出
import sys
# 警告控制库，用于过滤训练过程中的收敛警告，避免冗余输出
import warnings
# 数据类装饰器与替换工具，用于定义不可变的参数配置类
from dataclasses import dataclass, replace
# 任务并行与序列化库，此处用于记录版本号
import joblib
# 绘图库，此处仅用于记录版本号
import matplotlib
# 数值计算库，用于数组运算、混淆矩阵统计
import numpy as np
# 数据处理库，用于数据集读取、表格整理与结果导出
import pandas as pd
# 科学计算库，此处仅用于记录版本号
import scipy
# 机器学习库，此处仅用于记录版本号
import sklearn

# 线性判别分析分类器，作为轻量基线模型
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 随机森林分类器，作为集成学习基线模型
from sklearn.ensemble import RandomForestClassifier
# 收敛警告类，用于过滤MLP训练未完全收敛的警告
from sklearn.exceptions import ConvergenceWarning
# 逻辑回归分类器，作为线性基线模型
from sklearn.linear_model import LogisticRegression
# 评价指标：准确率、混淆矩阵、F1分数
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
# 数据集划分函数，用于分层抽样划分训练/测试集
from sklearn.model_selection import train_test_split
# 多层感知机分类器，主研究模型
from sklearn.neural_network import MLPClassifier
# 机器学习流水线，将标准化与分类器封装为可整体调用的对象
from sklearn.pipeline import Pipeline
# 标准化器，对特征做Z-score标准化
from sklearn.preprocessing import StandardScaler
# 支持向量机分类器，作为核方法基线模型
from sklearn.svm import SVC

# 从项目公共模块导入预定义常量与工具函数
from .common import (
    FEATURE_COLUMNS,    # 全部28个特征的列名列表
    FEATURE_GROUPS,     # 5组特征的分组字典（频率/振型/加速度统计/层间响应/频带能量）
    ci95,               # 95%置信区间计算函数
    load_config,        # 加载实验配置文件的函数
    load_dataset,       # 加载数据集CSV的函数
    mode_dirs,          # 创建输出目录的函数
    summarize,          # 按分组计算均值+标准差的汇总函数
)


# 定义MLP超参数配置类，frozen=True表示实例创建后不可修改，保证配置一致性
@dataclass(frozen=True)
class MLPSettings:
    hidden_units: int           # 隐藏层节点数
    activation: str            # 激活函数类型
    solver: str                # 优化器类型
    batch_size: int            # 小批量训练的批次大小
    learning_rate_init: float  # 初始学习率
    alpha: float               # L2正则化系数
    max_iter: int              # 最大训练轮数
    early_stopping: bool       # 是否启用早停机制
    validation_fraction: float # 早停使用的验证集占训练集比例
    n_iter_no_change: int      # 早停的容忍轮数（多少轮不提升就停止）


# 从配置字典中读取MLP参数，封装为类型安全的MLPSettings对象
def read_settings(config: dict) -> MLPSettings:
    values = config["mlp"]
    return MLPSettings(
        hidden_units=int(values["hidden_units"]),
        activation=str(values["activation"]),
        solver=str(values["solver"]),
        batch_size=int(values["batch_size"]),
        learning_rate_init=float(values["learning_rate_init"]),
        alpha=float(values["alpha"]),
        max_iter=int(values["max_iter"]),
        early_stopping=bool(values["early_stopping"]),
        validation_fraction=float(values["validation_fraction"]),
        n_iter_no_change=int(values["n_iter_no_change"]),
    )


# 构造带标准化的MLP模型流水线
# 输入：随机种子seed、MLP配置settings
# 输出：Pipeline对象，先执行特征标准化，再执行MLP分类
def make_mlp(seed: int, settings: MLPSettings) -> Pipeline:
    return Pipeline(
        [
            # 第一步：特征标准化（Z-score），使用训练集拟合，测试集独立变换，避免数据泄露
            ("scale", StandardScaler()),
            (
                "model",
                # 第二步：MLP分类器，所有超参数从配置对象传入
                MLPClassifier(
                    hidden_layer_sizes=(settings.hidden_units,),  # 单隐藏层，节点数由配置指定
                    activation=settings.activation,                # 激活函数
                    solver=settings.solver,                        # 优化器
                    batch_size=settings.batch_size,                # 批次大小
                    learning_rate_init=settings.learning_rate_init,# 初始学习率
                    alpha=settings.alpha,                          # L2正则化系数
                    max_iter=settings.max_iter,                    # 最大迭代轮数
                    early_stopping=settings.early_stopping,        # 是否启用早停
                    validation_fraction=settings.validation_fraction,# 早停验证集比例
                    n_iter_no_change=settings.n_iter_no_change,    # 早停容忍轮数
                    random_state=seed,                             # 固定随机种子，保证实验可复现
                    shuffle=True,                                  # 训练时打乱样本顺序
                ),
            ),
        ]
    )


# 构造所有基线分类器字典，用于同一数据集下的横向性能对比
# 输入：随机种子seed、MLP配置settings、全局配置config、快速模式标记quick
# 输出：模型名称 -> 模型对象的映射字典
def make_baselines(
    seed: int, settings: MLPSettings, config: dict, quick: bool
) -> dict[str, object]:
    # quick模式下随机森林只用50棵树加快速度；完整模式使用配置指定的树数
    trees = 50 if quick else int(config["baselines"]["random_forest_trees"])
    return {
        # 基线1：仅用4个固有频率特征 + 线性判别分析LDA
        "Frequency (4) + LDA": Pipeline(
            [("scale", StandardScaler()), ("model", LinearDiscriminantAnalysis())]
        ),
        # 基线2：仅用4个固有频率特征 + MLP
        "Frequency (4) + MLP": make_mlp(seed, settings),
        # 基线3：全部28个特征 + 逻辑回归
        "Full (28) + Logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(config["baselines"]["logistic_c"]),  # 正则化系数C
                        max_iter=300,                               # 最大迭代次数
                        solver="lbfgs",                              # 求解器
                        random_state=seed,                           # 固定随机种子
                    ),
                ),
            ]
        ),
        # 基线4：全部28个特征 + RBF核支持向量机
        "Full (28) + SVM-RBF": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=float(config["baselines"]["svm_c"]),  # 惩罚系数
                        gamma="scale",                           # 核函数gamma自动缩放
                        kernel="rbf",                            # 径向基核函数
                    ),
                ),
            ]
        ),
        # 基线5：全部28个特征 + 随机森林
        "Full (28) + Random forest": RandomForestClassifier(
            n_estimators=trees,                  # 决策树数量
            max_features="sqrt",                  # 每棵树特征数为特征总数开方
            class_weight="balanced_subsample",    # 类别平衡采样，保证各类别权重一致
            random_state=seed,                    # 固定随机种子
            n_jobs=-1,                            # 调用全部CPU核心并行训练
        ),
        # 主模型：全部28个特征 + MLP
        "Full (28) + MLP": make_mlp(seed, settings),
    }


# 模型训练与评估函数，统一计算各项评价指标
# 输入：模型对象、训练特征、测试特征、训练标签、测试标签
# 输出：包含各项指标的结果字典
def evaluate(
    model,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    # 捕获并忽略训练过程中的收敛警告，避免冗余的控制台输出
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        # 在训练集上拟合模型
        model.fit(x_train, y_train)

    # 计算训练集预测结果
    train_pred = model.predict(x_train)
    # 计算测试集预测结果
    test_pred = model.predict(x_test)

    # 组装基础结果字典
    result = {
        "train_accuracy": accuracy_score(y_train, train_pred),      # 训练集准确率
        "test_accuracy": accuracy_score(y_test, test_pred),        # 测试集准确率
        "test_macro_f1": f1_score(                                # 测试集宏F1（各类别算术平均，不考虑样本数）
            y_test, test_pred, average="macro", zero_division=0
        ),
        "test_weighted_f1": f1_score(                             # 测试集加权F1（按类别样本数加权）
            y_test, test_pred, average="weighted", zero_division=0
        ),
        "test_pred": np.asarray(test_pred, dtype=int),             # 测试集预测标签数组
    }

    # 如果是Pipeline且包含model步骤（即MLP类模型），额外提取训练过程信息
    if isinstance(model, Pipeline) and "model" in model.named_steps:
        fitted = model.named_steps["model"]
        # 获取实际迭代次数
        iteration_value = np.asarray(getattr(fitted, "n_iter_", 0))
        result["n_iter"] = int(iteration_value.max()) if iteration_value.size else 0
        # 获取训练损失曲线
        result["loss_curve"] = list(getattr(fitted, "loss_curve_", []) or [])
        # 获取早停验证集准确率曲线
        result["validation_scores"] = list(
            getattr(fitted, "validation_scores_", []) or []
        )
    else:
        # 非MLP模型没有训练曲线，填充默认空值
        result["n_iter"] = 0
        result["loss_curve"] = []
        result["validation_scores"] = []

    return result


# 生成特征消融实验用到的两种特征集：累计特征集、留一组特征集
# 输出：cumulative累计特征集字典，leave_one_out留一组特征集字典
def feature_sets() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    cumulative = {}
    running = []
    # 按顺序累加特征组，生成累计特征集（频率→振型→加速度统计→层间响应→频带能量）
    for columns in FEATURE_GROUPS.values():
        running = running + columns
        cumulative[f"Cumulative: {len(running)} features"] = list(running)

    # 生成留一组特征集：每次去掉一组特征，用剩余特征训练
    leave_one_out = {
        f"Without {group_name}": [
            feature for feature in FEATURE_COLUMNS if feature not in columns
        ]
        for group_name, columns in FEATURE_GROUPS.items()
    }

    return cumulative, leave_one_out


# 将所有实验结果整理为表格并保存为CSV文件
# 输入：输出目录、原始数据、基线结果行、主模型结果行、逐类结果行、曲线行、消融行、隐藏层行、正则化行、汇总混淆矩阵
def save_tables(
    table_dir,
    data: pd.DataFrame,
    baseline_rows: list[dict],
    main_rows: list[dict],
    class_rows: list[dict],
    curve_rows: list[dict],
    ablation_rows: list[dict],
    hidden_rows: list[dict],
    regularization_rows: list[dict],
    pooled: np.ndarray,
) -> None:
    # 将各类结果行列表转换为DataFrame表格结构
    baseline_detail = pd.DataFrame(baseline_rows)
    main_detail = pd.DataFrame(main_rows)
    class_detail = pd.DataFrame(class_rows)
    curves = pd.DataFrame(curve_rows)
    ablation_detail = pd.DataFrame(ablation_rows)
    hidden_detail = pd.DataFrame(hidden_rows)
    regularization_detail = pd.DataFrame(regularization_rows)

    # 计算基线模型的汇总统计（按模型分组，计算均值+标准差）
    baseline_summary = summarize(
        baseline_detail,
        "model",
        ["train_accuracy", "test_accuracy", "test_macro_f1"],
    )

    # 计算主MLP模型的汇总统计
    main_summary = summarize(
        main_detail.assign(configuration="Full (28) + regularized MLP"),
        "configuration",
        [
            "train_accuracy",
            "test_accuracy",
            "test_macro_f1",
            "generalization_gap",
        ],
    )

    # 计算逐类F1的汇总统计
    class_summary = summarize(class_detail, "label_id", ["f1"])
    # 匹配每个类别ID对应的类别名称
    names = data.drop_duplicates("label_id").set_index("label_id")["label_name"]
    class_summary.insert(1, "label_name", class_summary["label_id"].map(names))

    # 计算特征消融实验的汇总统计，并合并特征数、分析类型信息
    ablation_summary = summarize(
        ablation_detail,
        "feature_set",
        ["train_accuracy", "test_accuracy", "test_macro_f1"],
    ).merge(
        ablation_detail[
            ["feature_set", "analysis_type", "feature_count"]
        ].drop_duplicates(),
        on="feature_set",
        how="left",
    )

    # 计算隐藏层宽度实验的汇总统计
    hidden_summary = summarize(
        hidden_detail.assign(hidden_units=hidden_detail["hidden_units"].astype(str)),
        "hidden_units",
        ["train_accuracy", "test_accuracy", "test_macro_f1"],
    )

    # 计算正则化对比实验的汇总统计
    regularization_summary = summarize(
        regularization_detail,
        "setting",
        [
            "train_accuracy",
            "test_accuracy",
            "test_macro_f1",
            "generalization_gap",
        ],
    )

    # 组装所有要输出的文件映射表
    outputs = {
        "split_metrics.csv": main_detail,
        "main_summary.csv": main_summary,
        "baseline_detail.csv": baseline_detail,
        "baseline_summary.csv": baseline_summary,
        "per_class_f1_detail.csv": class_detail,
        "per_class_f1_summary.csv": class_summary,
        "ablation_detail.csv": ablation_detail,
        "ablation_summary.csv": ablation_summary,
        "hidden_width_detail.csv": hidden_detail,
        "hidden_width_summary.csv": hidden_summary,
        "regularization_detail.csv": regularization_detail,
        "regularization_summary.csv": regularization_summary,
        "training_curves.csv": curves,
    }

    # 逐个保存CSV文件，不带行索引，UTF-8带BOM编码兼容Excel打开
    for name, frame in outputs.items():
        frame.to_csv(table_dir / name, index=False, encoding="utf-8-sig")

    # 单独保存13×13的汇总混淆矩阵
    pd.DataFrame(pooled, index=range(13), columns=range(13)).to_csv(
        table_dir / "pooled_confusion_matrix.csv", encoding="utf-8-sig"
    )


# 主运行函数：执行完整的重复验证实验流程
# 输入：quick是否为快速模式（只跑1个种子，用于检查流程连通性）
def run(quick: bool) -> None:
    # 加载实验配置文件
    config = load_config()
    # 加载合成数据集
    data = load_dataset()
    # 创建结果输出目录，返回表格目录路径
    table_dir, _, _ = mode_dirs(quick)
    # 读取MLP超参数配置
    settings = read_settings(config)
    # 读取所有固定随机种子列表
    seeds = [int(value) for value in config["validation"]["seeds"]]
    # 快速模式下只取第一个种子，大幅加快运行速度
    if quick:
        seeds = seeds[:1]

    # 提取特征矩阵x和标签向量y
    x = data[FEATURE_COLUMNS]
    y = data["label_id"].to_numpy(dtype=int)

    # 生成消融实验用的特征集
    cumulative, leave_one_out = feature_sets()

    # 初始化所有实验的结果行列表
    baseline_rows = []
    main_rows = []
    class_rows = []
    curve_rows = []
    ablation_rows = []
    hidden_rows = []
    regularization_rows = []
    # 初始化13×13的汇总混淆矩阵，初始全零
    pooled = np.zeros((13, 13), dtype=int)

    # 循环每个随机种子，执行多次重复实验
    for repeat, seed in enumerate(seeds, start=1):
        # 分层抽样划分训练集/测试集，保持各类别比例一致
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=float(config["validation"]["test_fraction"]),  # 测试集比例
            random_state=seed,                                # 固定划分种子
            stratify=y,                                       # 按标签分层，保证类别平衡
        )

        # 根据索引划分训练/测试特征和标签
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # ========== 1. 基线模型对比实验 ==========
        for model_name, model in make_baselines(seed, settings, config, quick).items():
            # 频率开头的模型只用4个频率特征，其余用全部28个特征
            columns = (
                FEATURE_GROUPS["Frequency (4)"]
                if model_name.startswith("Frequency")
                else FEATURE_COLUMNS
            )

            # 训练评估当前模型
            result = evaluate(
                model, x_train[columns], x_test[columns], y_train, y_test
            )

            # 记录基线结果
            baseline_rows.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "model": model_name,
                    "feature_count": len(columns),
                    "train_accuracy": result["train_accuracy"],
                    "test_accuracy": result["test_accuracy"],
                    "test_macro_f1": result["test_macro_f1"],
                    "test_weighted_f1": result["test_weighted_f1"],
                }
            )

            # 如果是主模型（全特征MLP），额外记录详细结果、混淆矩阵、逐类F1、训练曲线
            if model_name == "Full (28) + MLP":
                prediction = result["test_pred"]
                # 记录主模型核心指标
                main_rows.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "train_accuracy": result["train_accuracy"],
                        "test_accuracy": result["test_accuracy"],
                        "test_macro_f1": result["test_macro_f1"],
                        "test_weighted_f1": result["test_weighted_f1"],
                        "generalization_gap": result["train_accuracy"]
                        - result["test_accuracy"],  # 泛化差距 = 训练准确率 - 测试准确率
                        "actual_stopping_epoch": result["n_iter"],  # 实际停止轮数
                    }
                )

                # 累加当前划分的混淆矩阵到总混淆矩阵
                pooled += confusion_matrix(y_test, prediction, labels=range(13))

                # 计算逐类F1分数并记录
                scores = f1_score(
                    y_test,
                    prediction,
                    labels=range(13),
                    average=None,
                    zero_division=0,
                )
                for label_id, score in enumerate(scores):
                    class_rows.append(
                        {
                            "repeat": repeat,
                            "seed": seed,
                            "label_id": label_id,
                            "f1": score,
                        }
                    )

                # 记录训练损失曲线和验证准确率曲线
                for epoch, loss in enumerate(result["loss_curve"], start=1):
                    validation = result["validation_scores"]
                    curve_rows.append(
                        {
                            "repeat": repeat,
                            "seed": seed,
                            "epoch": epoch,
                            "loss": loss,
                            "validation_accuracy": validation[epoch - 1]
                            if epoch <= len(validation)
                            else np.nan,
                        }
                    )

        # ========== 2. 特征消融实验 ==========
        for analysis_type, mapping in (
            ("cumulative", cumulative),          # 累计消融：逐步加入特征组
            ("leave_one_group_out", leave_one_out),# 留一组消融：每次去掉一组特征
        ):
            for name, columns in mapping.items():
                # 用当前特征子集训练MLP
                result = evaluate(
                    make_mlp(seed, settings),
                    x_train[columns],
                    x_test[columns],
                    y_train,
                    y_test,
                )
                # 记录消融结果
                ablation_rows.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "analysis_type": analysis_type,
                        "feature_set": name,
                        "feature_count": len(columns),
                        "train_accuracy": result["train_accuracy"],
                        "test_accuracy": result["test_accuracy"],
                        "test_macro_f1": result["test_macro_f1"],
                        "test_weighted_f1": result["test_weighted_f1"],
                    }
                )

        # ========== 3. 隐藏层宽度敏感性实验 ==========
        for width in config["analysis"]["hidden_widths"]:
            # 修改隐藏层节点数，训练评估MLP
            result = evaluate(
                make_mlp(seed, replace(settings, hidden_units=int(width))),
                x_train,
                x_test,
                y_train,
                y_test,
            )
            # 记录隐藏层实验结果
            hidden_rows.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "hidden_units": int(width),
                    "train_accuracy": result["train_accuracy"],
                    "test_accuracy": result["test_accuracy"],
                    "test_macro_f1": result["test_macro_f1"],
                    "test_weighted_f1": result["test_weighted_f1"],
                }
            )

        # ========== 4. 正则化对比实验 ==========
        # 定义三种正则化配置
        variants = {
            "No L2, no early stopping": replace(
                settings, alpha=0.0, early_stopping=False
            ),  # 无L2、无早停
            "L2 only": replace(settings, early_stopping=False),  # 仅L2正则、无早停
            "L2 + early stopping": settings,  # L2+早停（默认配置）
        }

        for name, variant in variants.items():
            # 训练评估不同正则化的MLP
            result = evaluate(
                make_mlp(seed, variant), x_train, x_test, y_train, y_test
            )
            # 记录正则化实验结果
            regularization_rows.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "setting": name,
                    "train_accuracy": result["train_accuracy"],
                    "test_accuracy": result["test_accuracy"],
                    "test_macro_f1": result["test_macro_f1"],
                    "test_weighted_f1": result["test_weighted_f1"],
                    "generalization_gap": result["train_accuracy"]
                    - result["test_accuracy"],
                }
            )

        # 打印当前进度，flush强制刷新输出，实时显示进度
        print(f"Completed split {repeat}/{len(seeds)}: {seed}", flush=True)

    # 所有种子跑完后，保存所有结果表格
    save_tables(
        table_dir,
        data,
        baseline_rows,
        main_rows,
        class_rows,
        curve_rows,
        ablation_rows,
        hidden_rows,
        regularization_rows,
        pooled,
    )

    # 记录运行环境信息（各库版本号、种子列表等）
    environment = {
        "mode": "quick" if quick else "full",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
        "seeds": seeds,
    }

    # 保存环境信息到JSON文件，用于结果复现核对
    (table_dir / "runtime_environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 读取主模型汇总结果，打印核心指标到控制台
    row = pd.read_csv(table_dir / "main_summary.csv", encoding="utf-8-sig").iloc[0]
    print(f"Accuracy mean: {row['test_accuracy_mean']:.6f}")
    print(f"Accuracy SD: {row['test_accuracy_sd']:.6f}")


# 命令行入口函数
def main() -> None:
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加--quick参数，用于快速检查流程是否通顺
    parser.add_argument("--quick", action="store_true")
    # 解析命令行参数
    arguments = parser.parse_args()
    # 调用主运行函数
    run(arguments.quick)


# 脚本直接运行时的入口点
if __name__ == "__main__":
    main()
