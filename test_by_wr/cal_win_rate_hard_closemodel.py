

import sys
sys.path.append('your path')

import os
from utils import load_jsonl_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

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


def judge_normal_close(data, model_1, model_2):
    m1_w, m2_w = 0, 0
    for example in data:
        ex_preference = example["preferences"]
        if ex_preference.count(model_1) == 2:
            assert ex_preference.count(model_2) == 0
            m1_w += 1
        elif ex_preference.count(model_1) == 1:
            assert ex_preference.count(model_2) == 1
            # m1_w += 0.5
            # m2_w += 0.5
        elif ex_preference.count(model_1) == 0:
            assert ex_preference.count(model_2) == 2
            m2_w += 1
        else:
            raise
    return m1_w, m2_w

# alpaca
data_type = "alpaca"
base_dir = "model_preferences_fullset/alpaca_eval_500_id_v5_wr"

# # translation
# data_type = "translation"
# base_dir = "model_preferences_fullset/translation_500_id_v2_wr"

# # truthfulness
# data_type = "truthfulness"
# base_dir = "model_preferences_fullset/truthfulness_500_id_v1_wr"


tag = 3 # indice of different comparsion model

if tag == 0:
    model_pair_lists_open = [
        'Llama-3.1-8B-Instruct_Qwen2.5-7B-Instruct',
        'Llama-3.1-8B-Instruct_gemma-2-9b-it',
        'Llama-3.1-70B-Instruct_Qwen2.5-72B-Instruct',
    ]
elif tag == 1:
    model_pair_lists_open = [
        "claude-3.5-haiku_Llama-3.1-70B-Instruct",
        "qwen-plus_Llama-3.1-70B-Instruct",
        "glm-4-plus_Llama-3.1-70B-Instruct",
    ]
elif tag == 2:
    model_pair_lists_open = [
        'Llama-3.1-8B_Qwen2.5-7B',
        'Llama-3.1-8B_gemma-2-9b',
        'Llama-3.1-70B_Qwen2.5-72B',
    ]
elif tag == 3:
    model_pair_lists_open = [
        # ## mytrain-mytrain
        'Llama-3.1-8B_Qwen2.5-7B',
        'Llama-3.1-8B-sft-UCfull_Qwen2.5-7B-sft-UCfull',
        # 'Llama-3.1-8B-Instruct_Llama-3.1-Tulu-3-8B',
        'Llama-3.1-8B-Instruct_Qwen2.5-7B-Instruct',
    ]

all_results = []

for model_pair in model_pair_lists_open:
    model_1, model_2 = model_pair.split('_')
    
    ##################
    # Normal evaluator
    ##################
    if model_1 == "claude-3.5-haiku" or model_1 == "qwen-plus" or model_1 == "glm-4-plus":
        evaluator_1_filepath = os.path.join(base_dir, f'evaluator_{model_1}/merge_{model_1}_{model_2}.jsonl')
    else:
        evaluator_1_filepath = os.path.join(base_dir, f'evaluator_{model_1}/average_{model_1}_{model_2}.jsonl')
    evaluator_1_data = load_jsonl_data(evaluator_1_filepath)
    data_len = len(evaluator_1_data)

    evaluator_2_filepath = os.path.join(base_dir, f'evaluator_{model_2}/average_{model_1}_{model_2}.jsonl')
    evaluator_2_data = load_jsonl_data(evaluator_2_filepath)
    assert len(evaluator_2_data) == data_len
    
    if model_1 == "claude-3.5-haiku" or model_1 == "qwen-plus" or model_1 == "glm-4-plus":
        ev1_mo1, ev1_mo2 = judge_normal_close(data=evaluator_1_data, model_1=model_1, model_2=model_2)
        ev2_mo1, ev2_mo2 = judge_normal(data=evaluator_2_data, model_1=model_1, model_2=model_2)
    else:
        ev1_mo1, ev1_mo2 = judge_normal(data=evaluator_1_data, model_1=model_1, model_2=model_2)
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
    
    golden_mo1, golden_mo2 = 0, 0
    
    for sub_id in data_ids:
        gemini_preferences = gemini_id2data[sub_id]["preferences"]
        gpt_preferences = gpt_id2data[sub_id]["preferences"]
        deepseek_preferences = deepseek_id2data[sub_id]["preferences"]
        all_preferences = gemini_preferences + gpt_preferences + deepseek_preferences
        
        cur_mo1 = all_preferences.count(model_1)
        cur_mo2 = all_preferences.count(model_2)

        if cur_mo1 > cur_mo2:  # model1 win
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


import matplotlib.pyplot as plt
import numpy as np

def plot_win_rates_3fig(model_1_list, model_2_list, result_items, save_name):
    """
        Function for base-instruct
    """
    num_plots = len(model_1_list)
    
    # 创建总 figure
    fig = plt.figure(figsize=(16, 1.7))

    # 创建 3x1 网格
    spec_top = fig.add_gridspec(nrows=1, ncols=3, left=0, right=1, top=1, bottom=0, wspace=0.01)

    axes = []  # 存储子图
    
    # 添加 3 张子图
    for i in range(3):
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
                ax.text(x_text, row, f"{values[row][i]:.1f}%", ha='center', va='center', color='white', fontsize=14)

        if model_1 == "claude-3.5-haiku":
            model_1 = "Claude-3.5-Haiku"
        if model_1 == "glm-4-plus":
            model_1 = "GLM-4-Plus"
        if model_1 == "qwen-plus":
            model_1 = "Qwen-Plus"
            
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
    print(f"Save figure to {save_name}.pdf")
    plt.savefig(f"{save_name}.pdf", dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    
def plot_win_rates_4fig(model_1_list, model_2_list, result_items, save_name):
    num_plots = len(model_1_list)
    
    # 创建总 figure
    fig = plt.figure(figsize=(20, 2.3))

    # 创建 4x1 网格
    spec_top = fig.add_gridspec(nrows=1, ncols=4, left=0, right=1, top=1, bottom=0, wspace=0.0)

    axes = []  # 存储子图
    
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

    
    # 保存图像时减少边界空白
    print(f"Save figure to {save_name}.pdf")
    plt.savefig(f"{save_name}.pdf", dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def plot_win_rates_3fig_closemodel(model_1_list, model_2_list, result_items, save_name):
    """
        Function for base-instruct
    """
    num_plots = len(model_1_list)
    
    # 创建总 figure
    fig = plt.figure(figsize=(16, 1.7))

    # 创建 3x1 网格
    spec_top = fig.add_gridspec(nrows=1, ncols=3, left=0, right=1, top=1, bottom=0, wspace=0.01)

    axes = []  # 存储子图
    
    # 添加 3 张子图
    for i in range(3):
        ax = fig.add_subplot(spec_top[0, i])
        axes.append(ax)
    
    for idx in range(num_plots):
        ax = axes[idx]  # 选择对应的子图
        model_1 = model_1_list[idx]
        model_2 = model_2_list[idx]
        result_item = result_items[idx]

        categories = ["Model B", "Gold", "Model A"]
        model1_tie = 100 - result_item[model_1][0] - result_item[model_1][1]

        values = [
            [result_item[model_2][0], 0., result_item[model_2][1]],
            [result_item["Golden"][0], 0., result_item["Golden"][1]],
            [result_item[model_1][0], model1_tie, result_item[model_1][1]],
        ]

        colors = ["#3371b3", "#5795c7", "#81b5d5"]
        y_pos = np.arange(len(categories))

        # 画水平条形图
        for i in range(len(values[0])):
            left_values = [sum(values[row][:i]) for row in range(len(values))]
            ax.barh(y_pos, [values[row][i] for row in range(len(values))], 
                    color=colors[i], left=left_values)

        # 添加百分比标签
        for row in range(len(values)):
            for i in range(len(values[row])):
                if (i == 1 and (row == 1 or row == 0)):
                    continue

                x_text = sum(values[row][:i]) + values[row][i] / 2
                if values[row][i] < 6:
                    x_text += 2
                
                if i == 2 and (values[row][i-1] + values[row][i-2] > 90):
                    x_text -= 1.5
                if i == 2 and (values[row][i-1] + values[row][i-2] > 98):
                    x_text -= 5
                ax.text(x_text, row, f"{values[row][i]:.1f}%", ha='center', va='center', color='white', fontsize=14)

        if model_1 == "claude-3.5-haiku":
            model_1 = "Claude-3.5-Haiku"
        if model_1 == "glm-4-plus":
            model_1 = "GLM-4-Plus"
        if model_1 == "qwen-plus":
            model_1 = "Qwen-Plus"

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



    # #############################
    # ## Save legend
    # # 创建一个新的图像只用于保存 legend
    # fig_legend = plt.figure(figsize=(6, 0.5))  # 可以根据需要调整大小
    # legend_ax = fig_legend.add_subplot(111)

    # # 创建自定义 legend 元素
    # colors = ["#3371b3", "#5795c7", "#81b5d5"]

    # labels = ["Model A Wins", "Tie", "Model B Wins"]  # 你可以根据你原来颜色的含义改掉
    # patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]

    # # 添加 legend 到新图中
    # legend = legend_ax.legend(handles=patches, loc='center', frameon=False, ncol=3, fontsize=14)

    # # 移除坐标轴
    # legend_ax.axis('off')

    # # 保存 legend 图像
    # fig_legend.savefig(f"figure/win_rate/main/legend_close.pdf", bbox_inches='tight', dpi=900)
    # plt.close(fig_legend)  # 关闭 legend 图，防止影响其他图像
    # ###############################
    
    # 保存图像时减少边界空白
    print(f"Save figure to {save_name}.pdf")
    plt.savefig(f"{save_name}.pdf", dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    


model_1_list, model_2_list, result_items = [], [], []

for cur_res in all_results:
    res_keys = list(cur_res.keys())
    model_1_list.append(res_keys[0])
    model_2_list.append(res_keys[1])
    result_items.append(cur_res)

if tag == 0:
    save_name = f"figure/win_rate/main/{data_type}_instruct_instruct"
elif tag == 1:
    save_name = f"figure/win_rate/main/{data_type}_instruct_instruct_close"
elif tag == 2:
    save_name = f"figure/win_rate/main/{data_type}_base_base"

if tag == 3:
    save_name = f"figure/win_rate/analysis/{data_type}_train_data"
    plot_win_rates_3fig(model_1_list, model_2_list, result_items, save_name)
elif tag == 1:
    plot_win_rates_3fig_closemodel(model_1_list, model_2_list, result_items, save_name)
else:  
    plot_win_rates_3fig(model_1_list, model_2_list, result_items, save_name)