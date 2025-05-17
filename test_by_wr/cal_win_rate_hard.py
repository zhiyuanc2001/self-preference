
import sys
sys.path.append('your path')

import os
from utils import load_jsonl_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.patches as mpatches


def judge_normal(data, model_1, model_2):
    m1_w, m2_w = 0, 0
    for example in data:
        if example["preferences"] == model_1:
            m1_w += 1
        elif example["preferences"] == model_2:
            m2_w += 1
        else:
            m1_w += 0.5
            m2_w += 0.5
    
    return m1_w, m2_w
    

model_pair_lists = [
    # ## base-instruct
    # 'Llama-3.1-8B_Llama-3.1-8B-Instruct',
    # 'Qwen2.5-7B_Qwen2.5-7B-Instruct',
    # 'gemma-2-9b_gemma-2-9b-it',
    # 'Llama-3.1-70B_Llama-3.1-70B-Instruct',
    # 'Qwen2.5-72B_Qwen2.5-72B-Instruct',
    
    ########## base-base instruct-instruct 在cal_win_rate_hard_closemodel.py 中
    
    # ## small-big
    # 'Llama-3.1-70B_Llama-3.1-8B',
    # 'Qwen2.5-72B_Qwen2.5-7B',
    # 'Llama-3.1-70B-Instruct_Llama-3.1-8B-Instruct',
    # 'Qwen2.5-72B-Instruct_Qwen2.5-7B-Instruct',
]

alpaca_model_pair_lists = [  
    # ## scaling
    # "Llama-3.1-70B-Instruct_Qwen2.5-0.5B-Instruct",
    # "Llama-3.1-70B-Instruct_Qwen2.5-1.5B-Instruct",
    # "Llama-3.1-70B-Instruct_Qwen2.5-3B-Instruct",
    # "Llama-3.1-70B-Instruct_Qwen2.5-7B-Instruct",
    # "Llama-3.1-70B-Instruct_Qwen2.5-14B-Instruct",
    # "Llama-3.1-70B-Instruct_Qwen2.5-32B-Instruct",
    # "Llama-3.1-70B-Instruct_Qwen2.5-72B-Instruct",
    
    ## reason
    'Llama-3.1-70B-Instruct_Qwen2.5-32B-Instruct',
    'Llama-3.1-70B-Instruct_DeepSeek-R1-Distill-Qwen-32B',
    'Llama-3.1-70B-Instruct_QwQ-32B'
]

# alpaca
data_type = "alpaca"
base_dir = "model_preferences_fullset/alpaca_eval_500_id_v5_wr"

# # translation
# data_type = "translation"
# base_dir = "model_preferences_fullset/translation_500_id_v2_wr"

# # truthfulness
# data_type = "truthfulness"
# base_dir = "model_preferences_fullset/truthfulness_500_id_v1_wr"


if data_type == "alpaca":
    model_pair_lists += alpaca_model_pair_lists

all_results = []

for model_pair in model_pair_lists:
    model_1, model_2 = model_pair.split('_')
    
    ##################
    # Normal evaluator
    ##################
    evaluator_1_filepath = os.path.join(base_dir, f'evaluator_{model_1}/average_{model_1}_{model_2}.jsonl')
    evaluator_1_data = load_jsonl_data(evaluator_1_filepath)
    data_len = len(evaluator_1_data)
    ev1_mo1, ev1_mo2 = judge_normal(data=evaluator_1_data, model_1=model_1, model_2=model_2)
    
    evaluator_2_filepath = os.path.join(base_dir, f'evaluator_{model_2}/average_{model_1}_{model_2}.jsonl')
    evaluator_2_data = load_jsonl_data(evaluator_2_filepath)
    assert len(evaluator_2_data) == data_len
    ev2_mo1, ev2_mo2 = judge_normal(data=evaluator_2_data, model_1=model_1, model_2=model_2)
    
    ##################
    # Golden evaluator
    ##################
    gemini_data_path = os.path.join(base_dir, "evaluator_gemini-flash-1.5_golden", "merge_" + model_pair + ".jsonl")
    gpt_data_path = os.path.join(base_dir, "evaluator_gpt-4o-mini_golden", "merge_" + model_pair + ".jsonl")
    deepseek_data_path = os.path.join(base_dir, "evaluator_deepseek-v3_golden", "merge_" + model_pair + ".jsonl")
    
    gemini_data = load_jsonl_data(gemini_data_path)
    gpt_data = load_jsonl_data(gpt_data_path)
    deepseek_data = load_jsonl_data(deepseek_data_path)
    
    data_ids = [item['id'] for item in gemini_data]
    gemini_id2data = {item['id']: item for item in gemini_data}
    gpt_id2data = {item['id']: item for item in gpt_data}
    deepseek_id2data = {item['id']: item for item in deepseek_data}
    assert len(data_ids) == data_len
    
    golden_mo1, golden_mo2, golden_tie = 0, 0, 0    
    for sub_id in data_ids:
        gemini_preferences = gemini_id2data[sub_id]["preferences"]
        gpt_preferences = gpt_id2data[sub_id]["preferences"]
        deepseek_preferences = deepseek_id2data[sub_id]["preferences"]
        
        all_preferences = gemini_preferences + gpt_preferences + deepseek_preferences
        
        cur_mo1 = all_preferences.count(model_1)
        cur_mo2 = all_preferences.count(model_2)

        if cur_mo1 > cur_mo2:  # model_1 win
            golden_mo1 += 1
        elif cur_mo2 > cur_mo1:
            golden_mo2 += 1
        else:
            golden_mo1 += 0.5
            golden_mo2 += 0.5
        
    all_results.append({
        model_1: (ev1_mo1 / data_len * 100, ev1_mo2 / data_len * 100),
        model_2: (ev2_mo1 / data_len * 100, ev2_mo2 / data_len * 100),
        "Golden": (golden_mo1 / data_len * 100, golden_mo2 / data_len * 100)
    })
    
    print(f"{model_1}: {ev1_mo1 / data_len * 100} || {ev1_mo2 / data_len * 100} --|-- {model_2}: {ev2_mo1 / data_len * 100} || {ev2_mo2 / data_len * 100}  --|-- golden: {golden_mo1 / data_len * 100} || {golden_mo2 / data_len * 100}")


#################
# Plot the figure
#################

def plot_win_rate(model_1, model_2, result_item):
    
    categories = ["Model B", "Golden ", "Model A"]
    values = [
        [result_item[model_2][0], result_item[model_2][1]],
        [result_item["Golden"][0], result_item["Golden"][1]],
        [result_item[model_1][0], result_item[model_1][1]],
    ]
    
    colors = ["#3371b3", "#aed4e5"]

    fig, ax = plt.subplots(figsize=(6, 2))
    y_pos = np.arange(len(categories))

    # 画水平条形图
    for i in range(len(values[0])):
        left_values = [sum(values[row][:i]) for row in range(len(values))]
        ax.barh(y_pos, [values[row][i] for row in range(len(values))], 
                color=colors[i], left=left_values)

    # 添加百分比标签
    for row in range(len(values)):
        for i in range(len(values[row])):
            x_text = sum(values[row][:i]) + values[row][i] / 2
            if values[row][i] < 6:
                x_text += 3
            ax.text(x_text, row, f"{values[row][i]:.1f}%", ha='center', va='center', color='white', fontsize=10)


    if model_1 == "Llama-3.1-8B-sft-UCfull" and model_2 == "Qwen2.5-7B-sft-UCfull":
        model_1 = "Llama-3.1-8B-UltraChat"
        model_2 = "Qwen2.5-7B-UltraChat"


    # 设置标题
    ax.set_title(f"Model A: {model_1}\nModel B: {model_2}", fontsize=13, fontweight='bold', loc="left", pad=2)

    # 设置 Y 轴标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    # ax.set_ylabel("Models", fontsize=10, fontweight='bold')  # Y 轴标题

    # # 设置 X 轴刻度
    # ax.set_xticks(np.arange(0, 110, 20))
    # ax.set_xticklabels([f"{x:.1f}%" for x in np.arange(0, 110, 20)])
    # # 设置 X 轴范围
    # ax.set_xlim(0, 100)
    
    # 取消 X 轴刻度和标签
    ax.set_xticks([])  # 移除刻度
    ax.set_xticklabels([])  # 移除刻度标签

    # 移除 X 轴轴线
    ax.spines['bottom'].set_visible(False)

    # 美化外观
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.savefig(f"figure/win_rate/scaling/{data_type}_{model_1}_{model_2}.pdf", dpi=600, bbox_inches='tight')

def plot_win_rates_5fig(model_1_list, model_2_list, result_items, save_name):
    """
        Function for base-instruct
    """
    num_plots = len(model_1_list)
    
    # 创建总 figure
    fig = plt.figure(figsize=(16, 4.2))

    # 创建第一部分：3x1 网格
    spec_top = fig.add_gridspec(nrows=1, ncols=3, left=0., right=1, top=1, bottom=0.6, wspace=0.01)
    
    # 创建第二部分：2x1 网格（居中）
    spec_bottom = fig.add_gridspec(nrows=1, ncols=2, left=0.17, right=0.83, top=0.4, bottom=0, wspace=0.01)

    axes = []  # 存储子图
    
    # 添加上方 3 张子图
    for i in range(3):
        ax = fig.add_subplot(spec_top[0, i])
        axes.append(ax)

    # 添加下方 2 张子图（居中）
    for i in range(2):
        ax = fig.add_subplot(spec_bottom[0, i])
        axes.append(ax)
    
    for idx in range(num_plots):
        ax = axes[idx]  # 选择对应的子图
        model_1 = model_1_list[idx]
        model_2 = model_2_list[idx]
        result_item = result_items[idx]

        categories = ["Model B", "Gold", "Model A"]
        values = [
            [result_item[model_2][0], result_item[model_2][1]],
            [result_item["Golden"][0], result_item["Golden"][1]],
            [result_item[model_1][0], result_item[model_1][1]],
        ]

        colors = ["#3371b3", "#81b5d5"]
        y_pos = np.arange(len(categories))

        # 画水平条形图
        for i in range(len(values[0])):
            left_values = [sum(values[row][:i]) for row in range(len(values))]
            ax.barh(y_pos, [values[row][i] for row in range(len(values))], 
                    color=colors[i], left=left_values)

        # 添加百分比标签
        for row in range(len(values)):
            for i in range(len(values[row])):
                x_text = sum(values[row][:i]) + values[row][i] / 2
                if values[row][i] < 6:
                    x_text += 2
                ax.text(x_text, row, f"{values[row][i]:.1f}%", ha='center', va='center', color='white', fontsize=14)

        if model_1 == "Llama-3.1-8B-sft-UCfull" and model_2 == "Qwen2.5-7B-sft-UCfull":
            model_1 = "Llama-3.1-8B-UltraChat"
            model_2 = "Qwen2.5-7B-UltraChat"

        # 设置标题
        ax.set_title(f"Model A: {model_1}\nModel B: {model_2}", fontsize=15, fontweight='bold', loc="left", pad=2)

        # 只在每行的第一个子图显示 Y 轴标签
        if idx % 3 == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories, fontsize=14)
            ax.set_ylabel("Judge Model", fontsize=14, labelpad=15)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        # 隐藏 X 轴
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.spines['bottom'].set_visible(False)

        # 美化外观
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)


    # 保存图像时减少边界空白
    plt.savefig(f"{save_name}.pdf", dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def plot_win_rates_4fig(model_1_list, model_2_list, result_items, save_name):
    """
        Function for big_small
    """
    num_plots = len(model_1_list)
    
    # 创建总 figure
    fig = plt.figure(figsize=(22, 2.3))

    # 创建 4x1 网格
    spec_top = fig.add_gridspec(nrows=1, ncols=4, left=0, right=1, top=1, bottom=0, wspace=0.0)

    axes = []  # 存储子图
    
    # 添加 3 张子图
    for i in range(4):
        ax = fig.add_subplot(spec_top[0, i])
        axes.append(ax)
    
    for idx in range(num_plots):
        ax = axes[idx]  # 选择对应的子图
        model_1 = model_1_list[idx]
        model_2 = model_2_list[idx]
        result_item = result_items[idx]

        categories = ["Model B", "Gold", "Model A"]
        values = [
            [result_item[model_2][0], result_item[model_2][1]],
            [result_item["Golden"][0], result_item["Golden"][1]],
            [result_item[model_1][0], result_item[model_1][1]],
        ]

        colors = ["#3371b3", "#81b5d5"]
        y_pos = np.arange(len(categories))

        # 画水平条形图
        for i in range(len(values[0])):
            left_values = [sum(values[row][:i]) for row in range(len(values))]
            ax.barh(y_pos, [values[row][i] for row in range(len(values))], 
                    color=colors[i], left=left_values)

        # 添加百分比标签
        for row in range(len(values)):
            for i in range(len(values[row])):
                x_text = sum(values[row][:i]) + values[row][i] / 2
                if values[row][i] < 6:
                    x_text += 2
                ax.text(x_text, row, f"{values[row][i]:.1f}%", ha='center', va='center', color='white', fontsize=18)

        if model_1 == "Llama-3.1-8B-sft-UCfull" and model_2 == "Qwen2.5-7B-sft-UCfull":
            model_1 = "Llama-3.1-8B-UltraChat"
            model_2 = "Qwen2.5-7B-UltraChat"

        # 设置标题
        ax.set_title(f"Model A: {model_1}\nModel B: {model_2}", fontsize=19, fontweight='bold', loc="left", pad=2)

        # 只在每行的第一个子图显示 Y 轴标签
        if idx % 4 == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories, fontsize=18)
            ax.set_ylabel("Judge Model", fontsize=18, labelpad=15)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        # 隐藏 X 轴
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.spines['bottom'].set_visible(False)

        # 美化外观
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # #############################
    # ## Save legend
    # # 创建一个新的图像只用于保存 legend
    # fig_legend = plt.figure(figsize=(4, 0.5))  # 可以根据需要调整大小
    # legend_ax = fig_legend.add_subplot(111)

    # # 创建自定义 legend 元素
    # colors = ["#3371b3", "#81b5d5"]
    # labels = ["Model A Wins", "Model B Wins"]  # 你可以根据你原来颜色的含义改掉
    # patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]

    # # 添加 legend 到新图中
    # legend = legend_ax.legend(handles=patches, loc='center', frameon=False, ncol=2, fontsize=14)

    # # 移除坐标轴
    # legend_ax.axis('off')

    # # 保存 legend 图像
    # fig_legend.savefig(f"figure/win_rate/main/legend.pdf", bbox_inches='tight', dpi=1200)
    # plt.close(fig_legend)  # 关闭 legend 图，防止影响其他图像
    # ###############################


    ##保存图像时减少边界空白
    print(f"Save figure to {save_name}.pdf")
    plt.savefig(f"{save_name}.pdf", dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def plot_win_rates_7fig(model_1_list, model_2_list: list, result_items, save_name):
    cur_ev1_mo2, cur_ev2_mo2, cur_golden_mo2 = [], [], []
    for idx in range(len(result_items)):
        cur_ev1 = model_1_list[idx]
        cur_ev2 = model_2_list[idx]
        cur_res = result_items[idx]
        
        cur_ev1_mo2.append(cur_res[cur_ev1][1])
        cur_ev2_mo2.append(cur_res[cur_ev2][1])
        cur_golden_mo2.append(cur_res["Golden"][1])
    plt.figure(figsize=(8.5, 6))

    x_name = [item.split('-')[1] for item in model_2_list]
    x_range = range(len(x_name))

    # 绘制三条虚线折线，颜色为浅灰色
    marker_colors = {
        'ev1': '#1f77b4',   # 蓝色
        'ev2': '#ff7f0e',   # 橙色
        'golden': '#2ca02c' # 绿色
    }

    # 每个横轴位置画一条连接三个点的横线（从最低到最高）
    for i in x_range:
        y_values = [cur_ev2_mo2[i], cur_golden_mo2[i]]
        y_values_sorted = sorted(y_values)  # 从小到大，保证线段不乱
        plt.plot([x_name[i]] * 2, y_values_sorted, color='darkgray', linestyle=":", alpha=1, linewidth=1.5)

    # 折线是浅灰色，marker 设置为彩色
    plt.plot(x_name, cur_ev1_mo2, linestyle='--', marker='o', label='Llama-3.1-70B-Instruct', 
            color=marker_colors['ev1'], markerfacecolor=marker_colors['ev1'], markeredgecolor=marker_colors['ev1'])

    plt.plot(x_name, cur_ev2_mo2, linestyle='--', marker='s', label='Qwen2.5-Instruct',
            color=marker_colors['ev2'], markerfacecolor=marker_colors['ev2'], markeredgecolor=marker_colors['ev2'])

    plt.plot(x_name, cur_golden_mo2, linestyle='--', marker='^', label='Gold Judgments',
            color=marker_colors['golden'], markerfacecolor=marker_colors['golden'], markeredgecolor=marker_colors['golden'])



    # 设置标题和轴标签
    # plt.title('Llama-3.1-70B v.s. Qwen2.5-series')
    plt.xlabel('Qwen2.5-Instruct series models', fontsize=16)
    plt.ylabel('Qwen2.5-Instruct Win Rate (%)', fontsize=16)

    # 设置图例
    plt.legend(fontsize=14)  # 图例字体大小
    plt.xticks(fontsize=15, rotation=0)  # x 轴刻度字体大小
    plt.yticks(fontsize=15)
    plt.ylim(0, 70)

    # # # 添加网格线
    # plt.grid(True, linestyle='solid', alpha=0.3)

    # 自动调整布局
    plt.tight_layout()

    print(f"Save figure to {save_name}.pdf")
    plt.savefig(f"{save_name}.pdf", dpi=1500, bbox_inches='tight', pad_inches=0.05)
    
        


model_1_list, model_2_list, result_items = [], [], []

for cur_res in all_results:
    res_keys = list(cur_res.keys())
    model_1_list.append(res_keys[0])
    model_2_list.append(res_keys[1])
    result_items.append(cur_res)

if len(all_results) == 5:
    print("5 Fig")
    save_name = f"figure/win_rate/main/{data_type}_base_instruct"
    plot_win_rates_5fig(model_1_list, model_2_list, result_items, save_name)
    
if len(all_results) == 4:  # small to big
    print("4 Fig")
    save_name = f"figure/win_rate/main/{data_type}_big_small"
    plot_win_rates_4fig(model_1_list, model_2_list, result_items, save_name)
    
if len(all_results) == 7: # scaling
    print("7 Fig")
    save_name = f"figure/win_rate/analysis/{data_type}_scaling"
    plot_win_rates_7fig(model_1_list, model_2_list, result_items, save_name)
