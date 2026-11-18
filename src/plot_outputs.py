# 启用未来类型注解语法，支持类型注解前向引用
from __future__ import annotations
# 命令行参数解析库
import argparse
# 路径处理库，用于文件路径操作
from pathlib import Path
# matplotlib绘图主库
import matplotlib as mpl
# 颜色处理模块
import matplotlib.colors as mcolors
# pyplot绘图接口
import matplotlib.pyplot as plt
# 数值计算库
import numpy as np
# 数据处理库，用于读取结果表格
import pandas as pd
# 绘图补丁元素：圆形、箭头、圆角矩形、矩形
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# 从同包common模块导入目录工具函数
from .common import mode_dirs


# 全局应用学术绘图样式：统一字体、字号、线宽、导出参数
def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",                # 字体族：衬线体
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], # 衬线字体优先级
            "font.size": 8,                       # 全局基础字号
            "axes.labelsize": 8,                  # 坐标轴标签字号
            "axes.titlesize": 9,                  # 坐标轴标题字号
            "xtick.labelsize": 7,                 # x轴刻度字号
            "ytick.labelsize": 7,                 # y轴刻度字号
            "legend.fontsize": 7,                 # 图例字号
            "axes.linewidth": 0.7,                # 坐标轴线宽
            "svg.fonttype": "none",               # SVG导出字体转曲
            "pdf.fonttype": 42,                   # PDF导出字体类型
            "savefig.dpi": 600,                   # 导出图片分辨率
        }
    )


# 统一图片保存函数：同时保存PNG和SVG格式，自动关闭画布
# 输入：图对象、输出目录、文件名前缀
def save(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ===================== 图1：四层剪切框架示意图 =====================
def plot_structure(directory: Path) -> None:
    apply_style()  # 应用全局样式
    fig, ax = plt.subplots(figsize=(6.0, 3.2))  # 创建画布，指定尺寸
    ax.set_xlim(-0.5, 8.8)  # x轴坐标范围
    ax.set_ylim(-0.25, 4.7)  # y轴坐标范围
    ax.axis("off")  # 隐藏坐标轴

    # 配色定义
    floor_color = "#263238"    # 楼层深色
    spring_color = "#356D9C"   # 正常刚度弹簧蓝色
    accent = "#B85C38"        # 损伤层弹簧强调色

    # 绘制5条水平线（0-4层楼板）
    for floor in range(5):
        y = floor
        ax.plot([0.6, 3.7], [y, y], color=floor_color, lw=2.0 if floor else 2.8)

    # 绘制4层层间柱子与弹簧
    for story in range(4):
        y0 = story       # 下层y坐标
        y1 = story + 1   # 上层y坐标
        # 绘制左右两根立柱
        for x in (1.0, 3.3):
            ax.plot([x, x], [y0, y1], color="#7A8791", lw=1.1)
        # 绘制层间弹簧（锯齿形表示剪切刚度）
        x_values = np.linspace(2.15, 2.15, 13)
        y_values = np.linspace(y0 + 0.08, y1 - 0.08, 13)
        x_values[1:-1:2] -= 0.17
        x_values[2:-1:2] += 0.17
        # 第一层损伤用强调色，其余用正常色
        ax.plot(x_values, y_values, color=accent if story == 1 else spring_color, lw=1.4)
        # 标注层间刚度符号
        ax.text(2.45, y0 + 0.5, f"$k_{story + 1}$", va="center", fontsize=8)
        # 标注楼层名称
        ax.text(0.1, y0 + 0.5, f"Story {story + 1}", va="center", fontsize=7.5)

    # 绘制右侧自由度示意：质量块+位移箭头
    for floor in range(1, 5):
        ax.add_patch(Circle((3.7, floor), 0.09, color="#263238"))  # 质量点
        # 位移箭头
        arrow = FancyArrowPatch(
            (3.86, floor),
            (4.55, floor),
            arrowstyle="-|>",
            mutation_scale=9,
            lw=0.8,
            color="#263238",
        )
        ax.add_patch(arrow)
        # 位移符号标注
        ax.text(4.64, floor, f"$x_{floor}(t)$", va="center", fontsize=8)

    # 右侧损伤工况说明圆角框
    ax.add_patch(
        FancyBboxPatch(
            (5.45, 0.55),
            2.85,
            3.0,
            boxstyle="round,pad=0.12",
            facecolor="#F6F8FA",
            edgecolor="#BAC4CC",
            linewidth=0.9,
        )
    )
    # 框内文字说明
    ax.text(6.88, 3.25, "Damage-state design", ha="center", fontweight="bold")
    ax.text(5.75, 2.75, "Intact: 1 class", va="center")
    ax.text(5.75, 2.25, "Location: Stories 1–4", va="center")
    ax.text(5.75, 1.75, "Stiffness loss: 10%, 20%, 30%", va="center")
    ax.text(5.75, 1.25, "Total: 13 classes", va="center")

    # 图标题与说明
    ax.text(2.15, 4.38, "Four-story shear frame", ha="center", fontweight="bold")
    ax.text(2.15, -0.18, "Four translational degrees of freedom", ha="center", fontsize=7)

    fig.tight_layout()  # 自动紧凑布局
    save(fig, directory, "fig1_four_story_frame")  # 保存图片


# ===================== 图2：实验技术路线流程图 =====================
def plot_workflow(directory: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    # 主流程五个节点定义：x坐标、y坐标、宽度、高度、标题、副标题
    nodes = [
        (0.25, 1.65, 1.55, 0.75, "Structural states", "13 classes"),
        (2.25, 1.65, 1.55, 0.75, "Numerical response", "modal + dynamic"),
        (4.25, 1.65, 1.55, 0.75, "Feature table", "28 candidates"),
        (6.25, 1.65, 1.55, 0.75, "Repeated splits", "10 stratified"),
        (8.25, 1.65, 1.55, 0.75, "Evaluation", "metrics + matrices"),
    ]
    # 节点渐变色
    colors = ["#D8E8F2", "#E6EEF3", "#E4EADB", "#F2E4D8", "#E6DFF0"]

    # 绘制主流程节点与箭头
    for index, (x, y, width, height, title, subtitle) in enumerate(nodes):
        # 圆角矩形节点
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.06",
                facecolor=colors[index],
                edgecolor="#5E6872",
                linewidth=0.8,
            )
        )
        # 节点标题
        ax.text(x + width / 2, y + 0.48, title, ha="center", va="center", fontweight="bold")
        # 节点副标题
        ax.text(x + width / 2, y + 0.23, subtitle, ha="center", va="center", fontsize=7)

        # 节点之间的连接箭头（最后一个节点不画）
        if index < len(nodes) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + width + 0.05, y + height / 2),
                    (nodes[index + 1][0] - 0.05, y + height / 2),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    color="#5E6872",
                    lw=0.9,
                )
            )

    # 下方分支：三类分析扩展
    branches = [
        (4.28, 0.35, "Classifier baselines"),
        (6.03, 0.35, "Feature ablation"),
        (7.78, 0.35, "Redundancy analysis"),
    ]
    for x, y, label in branches:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                1.55,
                0.55,
                boxstyle="round,pad=0.05",
                facecolor="white",
                edgecolor="#87939C",
                linewidth=0.7,
            )
        )
        ax.text(x + 0.775, y + 0.275, label, ha="center", va="center", fontsize=7)

    # 主流程指向分支的垂直箭头
    ax.add_patch(
        FancyArrowPatch(
            (7.02, 1.62),
            (7.02, 0.94),
            arrowstyle="-|>",
            mutation_scale=8,
            color="#87939C",
            lw=0.8,
        )
    )

    # 图总标题
    ax.text(0.25, 2.68, "Mechanics-informed repeated-validation workflow", fontweight="bold", fontsize=10)

    fig.tight_layout()
    save(fig, directory, "fig2_workflow")


# ===================== 图3：MLP训练轨迹图 =====================
def plot_training(table_dir: Path, directory: Path) -> None:
    apply_style()
    # 读取训练曲线数据
    curves = pd.read_csv(table_dir / "training_curves.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55))  # 1行2列子图

    # 两个子图配置：坐标轴、指标、颜色、标题、y轴标签
    for ax, metric, color, title, ylabel in (
        (axes[0], "loss", "#2C6BA0", "Training loss", "Cross-entropy loss"),
        (
            axes[1],
            "validation_accuracy",
            "#B4682B",
            "Internal validation accuracy",
            "Accuracy",
        ),
    ):
        valid = curves.dropna(subset=[metric])  # 去除空值
        # 按轮次分组，计算均值和标准差
        stats = valid.groupby("epoch")[metric].agg(["mean", "std"]).reset_index()
        x = stats["epoch"].to_numpy(dtype=float)
        mean = stats["mean"].to_numpy(dtype=float)
        sd = stats["std"].fillna(0).to_numpy(dtype=float)

        ax.plot(x, mean, color=color, lw=1.5)  # 绘制均值曲线
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.20, linewidth=0)  # 绘制标准差填充带

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9DEE5", linewidth=0.5)  # 横向网格线
        ax.spines[["top", "right"]].set_visible(False)  # 隐藏上、右边框

    fig.tight_layout()
    save(fig, directory, "fig3_training_trajectories")


# ===================== 图4：四组对比实验柱状图 =====================
def plot_comparisons(table_dir: Path, directory: Path) -> None:
    apply_style()
    # 读取四类对比结果表
    baseline = pd.read_csv(table_dir / "baseline_summary.csv", encoding="utf-8-sig")
    ablation = pd.read_csv(table_dir / "ablation_summary.csv", encoding="utf-8-sig")
    ablation = ablation[ablation["analysis_type"] == "cumulative"].copy()  # 仅取累计消融部分
    hidden = pd.read_csv(table_dir / "hidden_width_summary.csv", encoding="utf-8-sig")
    regularization = pd.read_csv(
        table_dir / "regularization_summary.csv", encoding="utf-8-sig"
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))  # 2行2列子图
    # 四个面板配置：坐标轴、数据、标签列、子图标题
    panels = [
        (axes[0, 0], baseline, "model", "a  Classifier baselines"),
        (axes[0, 1], ablation, "feature_count", "b  Cumulative features"),
        (axes[1, 0], hidden, "hidden_units", "c  Hidden width"),
        (axes[1, 1], regularization, "setting", "d  Regularization"),
    ]
    # 四个面板配色
    colors = ["#3B73A6", "#6D9F71", "#B7853D", "#8A6FA8"]

    for index, (ax, frame, label_column, title) in enumerate(panels):
        labels = frame[label_column].astype(str).tolist()
        mean = frame["test_accuracy_mean"].to_numpy(dtype=float)
        sd = frame["test_accuracy_sd"].to_numpy(dtype=float)
        positions = np.arange(len(frame))

        # 绘制柱状图
        ax.bar(positions, mean, color=colors[index], alpha=0.90, width=0.72)
        # 绘制误差棒
        ax.errorbar(
            positions,
            mean,
            yerr=sd,
            fmt="none",
            ecolor="#222222",
            elinewidth=0.8,
            capsize=2,
        )

        # y轴范围自适应，下限不低于0.45
        ax.set_ylim(max(0.45, float(np.min(mean - sd)) - 0.05), 1.0)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=27, ha="right")  # x轴标签倾斜
        ax.set_ylabel("Test accuracy")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#D9DEE5", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    save(fig, directory, "fig4_comparative_experiments")


# ===================== 辅助函数：按损伤程度聚合混淆矩阵 =====================
# 输入：13×13原始混淆矩阵
# 输出：4×4程度计数矩阵、4×4程度百分比矩阵
def severity_matrix(pooled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 13类对应的损伤程度等级：0完好，1/2/3对应10%/20%/30%，共4层重复
    severity = np.array([0] + [level for _ in range(4) for level in (1, 2, 3)])
    counts = np.zeros((4, 4), dtype=int)

    # 遍历13×13矩阵，按程度累加
    for true_label in range(13):
        for predicted_label in range(13):
            counts[severity[true_label], severity[predicted_label]] += pooled[
                true_label, predicted_label
            ]

    # 行归一化转为百分比
    percentages = counts / counts.sum(axis=1, keepdims=True) * 100.0
    return counts, percentages


# ===================== 图5：混淆矩阵图 =====================
def plot_confusion(table_dir: Path, directory: Path) -> None:
    apply_style()
    # 读取13类汇总混淆矩阵
    pooled = pd.read_csv(
        table_dir / "pooled_confusion_matrix.csv",
        index_col=0,
        encoding="utf-8-sig",
    ).to_numpy(dtype=int)

    counts, percentages = severity_matrix(pooled)

    # 保存按程度聚合的混淆矩阵CSV
    pd.DataFrame(
        counts,
        index=["Intact", "10%", "20%", "30%"],
        columns=["Intact", "10%", "20%", "30%"],
    ).to_csv(table_dir / "severity_confusion_counts.csv", encoding="utf-8-sig")
    pd.DataFrame(
        percentages,
        index=["Intact", "10%", "20%", "30%"],
        columns=["Intact", "10%", "20%", "30%"],
    ).to_csv(table_dir / "severity_confusion_percent.csv", encoding="utf-8-sig")

    # 配色：自定义蓝色渐变
    palette = ["#F7FAFC", "#D7E7F3", "#88B6D8", "#3775BA", "#0F4D92"]
    color_map = mcolors.LinearSegmentedColormap.from_list("journal_blue", palette)

    fig = plt.figure(figsize=(6.0, 3.2))
    # 1行2列网格布局，宽度比例不同
    grid = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.42)
    ax1 = fig.add_subplot(grid[0, 0])  # 左图：13类混淆矩阵
    ax2 = fig.add_subplot(grid[0, 1])  # 右图：4程度混淆矩阵

    labels = ["Intact"] + [
        f"S{story}-{level}"
        for story in range(1, 5)
        for level in (10, 20, 30)
    ]

    # 左图：13类计数混淆矩阵热力图
    image1 = ax1.imshow(pooled, cmap=color_map, vmin=0, vmax=pooled.max(), aspect="equal")
    ax1.set_xticks(range(13), labels, rotation=55, ha="right", rotation_mode="anchor", fontsize=5.5)
    ax1.set_yticks(range(13), labels, fontsize=5.5)
    ax1.set_xlabel("Predicted class")
    ax1.set_ylabel("True class")
    ax1.tick_params(length=0, pad=1.0)

    # 单元格内标注数值
    for i in range(13):
        for j in range(13):
            value = int(pooled[i, j])
            # 对角线或数值较大时显示文字，深色背景白字，浅色背景深字
            if i == j or value >= max(2, int(pooled.max() * 0.08)):
                color = "white" if value >= pooled.max() * 0.48 else "#1F1F1F"
                ax1.text(j, i, str(value), ha="center", va="center", fontsize=5.1, color=color)

    # 左图颜色条
    colorbar1 = fig.colorbar(image1, ax=ax1, fraction=0.045, pad=0.025)
    colorbar1.set_label("Pooled count", fontsize=6.5)
    colorbar1.ax.tick_params(labelsize=5.5, length=2)

    # 右图：4程度百分比混淆矩阵热力图
    image2 = ax2.imshow(percentages, cmap=color_map, vmin=0, vmax=100, aspect="equal")
    severity_labels = ["Intact", "10%", "20%", "30%"]
    ax2.set_xticks(range(4), severity_labels, rotation=35, ha="right", fontsize=6.2)
    ax2.set_yticks(range(4), severity_labels, fontsize=6.2)
    ax2.set_xlabel("Predicted severity")
    ax2.set_ylabel("True severity")
    ax2.tick_params(length=0, pad=1.5)

    # 单元格内标注百分比
    for i in range(4):
        for j in range(4):
            value = percentages[i, j]
            color = "white" if value >= 48 else "#1F1F1F"
            ax2.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.2, color=color)

    # 右图颜色条
    colorbar2 = fig.colorbar(image2, ax=ax2, fraction=0.07, pad=0.035)
    colorbar2.set_label("Row-normalized (%)", fontsize=6.5)
    colorbar2.ax.tick_params(labelsize=5.5, length=2)

    # 添加子图编号a/b
    for ax, label in ((ax1, "a"), (ax2, "b")):
        ax.text(-0.16, 1.03, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 手动精细调整布局
    fig.subplots_adjust(left=0.105, right=0.97, bottom=0.22, top=0.95)
    save(fig, directory, "fig5_pooled_and_severity_confusion")


# ===================== 主运行函数 =====================
def run(quick: bool) -> None:
    table_dir, figure_dir, _ = mode_dirs(quick)

    # 检查必需的结果表格是否存在
    required = [
        "training_curves.csv",
        "baseline_summary.csv",
        "ablation_summary.csv",
        "hidden_width_summary.csv",
        "regularization_summary.csv",
        "pooled_confusion_matrix.csv",
    ]
    missing = [name for name in required if not (table_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing result tables: {missing}")

    # 依次绘制5张图
    plot_structure(figure_dir)
    plot_workflow(figure_dir)
    plot_training(table_dir, figure_dir)
    plot_comparisons(table_dir, figure_dir)
    plot_confusion(table_dir, figure_dir)

    print(f"Figures: {figure_dir}")


# 命令行入口
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(arguments.quick)


# 脚本直接运行入口
if __name__ == "__main__":
    main()
