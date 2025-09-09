import json
import re
from transformers import AutoTokenizer

# 加载预训练的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/fangxiaotong/Desktop/LLM/Meta-Llama-3.1-8B-Instruct")
from src import config

if __name__ == '__main__':
    dataset_names = [f"test-{i}" for i in range(26, 36)]
    data_list = []
    regenerate = []
    wrong = 0
    max_token_length = 2048
    max_data_entries = 100
    for dataset_name in dataset_names:
        json_dir = config.dataset_dir / dataset_name / "prompt" / "txt"
        prompt_dir = json_dir / "prompt_grd"
        prompt_dir2 = json_dir / "prompt_com"
        # prompt_dir = json_dir / "grd_no_exp"
        # prompt_dir2 = json_dir / "com_no_exp"
        # 获取两个目录中的所有 .txt 文件
        filenames1 = list(prompt_dir.glob("*.txt"))
        filenames2 = list(prompt_dir2.glob("*.txt"))
        data_count = 0
        for file in filenames1:
            with open(file, 'r', encoding='utf-8', errors='ignore') as file:

                try:
                    file_content = file.read()
                except UnicodeDecodeError:
                    regenerate.append(file)
                # 定义正则表达式模式
                instruction_pattern = re.compile(r'instruction:(.*?)(?=input:|$)', re.DOTALL)
                input_pattern = re.compile(r'input:(.*?)(?=output:|$)', re.DOTALL)
                output_pattern = re.compile(r'output:(.*?)(?=$)', re.DOTALL)

                # 查找匹配项
                instruction_match = instruction_pattern.search(file_content)
                input_match = input_pattern.search(file_content)
                output_match = output_pattern.search(file_content)

                # 检查匹配结果并提取内容
                instruction = instruction_match.group(1).strip() if instruction_match else ""
                input_string = input_match.group(1).strip() if input_match else ""
                output = output_match.group(1).strip() if output_match else ""

                # if not instruction_match or not input_match or not output_match:
                #     first_split = file_content.split('\n', 1)
                #     if len(first_split) == 2:
                #         instruction, remaining = first_split
                #         second_split = remaining.split('So finally', 1)
                #         if len(second_split) == 2:
                #             input_string, output = second_split
                #             output='So finally'+ output

                if instruction == "" or input_string == "" or output == "":
                    print(repr(file_content))
                    wrong += 1
                elif len(tokenizer.tokenize(instruction)) + len(tokenizer.tokenize(input_string)) + len(
                        tokenizer.tokenize(output)) <= max_token_length:
                    data_list.append({
                        "instruction": instruction,
                        "input": input_string,
                        "output": output,
                    })
                    data_count += 1
                if data_count >= max_data_entries:
                    break
        print(data_count)
        data_count = 0
        for file in filenames2:
            with open(file, 'r', encoding='utf-8') as file:
                try:
                    file_content = file.read()
                except UnicodeDecodeError:
                    regenerate.append(file)
                # 定义正则表达式模式
                instruction_pattern = re.compile(r'instruction:(.*?)(?=input:|$)', re.DOTALL)
                input_pattern = re.compile(r'input:(.*?)(?=output:|$)', re.DOTALL)
                output_pattern = re.compile(r'output:(.*?)(?=$)', re.DOTALL)

                # 查找匹配项
                instruction_match = instruction_pattern.search(file_content)
                input_match = input_pattern.search(file_content)
                output_match = output_pattern.search(file_content)

                # 检查匹配结果并提取内容
                instruction = instruction_match.group(1).strip() if instruction_match else ""
                input_string = input_match.group(1).strip() if input_match else ""
                output = output_match.group(1).strip() if output_match else ""

                # if (not instruction_match or not input_match or not output_match):
                #     print("not match")
                #     first_split = file_content.split('\n', 1)
                #     if len(first_split) == 2:
                #         instruction, remaining = first_split
                #         second_split = remaining.split('Now', 1)
                #         if len(second_split) == 2:
                #             input_string, output = second_split
                #             output = "Now" + output
                #         else:
                #             instruction, remaining = first_split
                #             second_split = remaining.split('So finally', 1)
                #             if len(second_split) == 2:
                #                 input_string, output = second_split
                #                 output = 'So finally' + output

                if instruction == "" or input_string == "" or output == "":
                    print(repr(file_content))
                    wrong += 1
                elif len(tokenizer.tokenize(instruction)) + len(tokenizer.tokenize(input_string)) + len(
                        tokenizer.tokenize(output)) <= max_token_length:
                    data_list.append({
                        "instruction": instruction,
                        "input": input_string,
                        "output": output,
                    })
                    data_count += 1
                if data_count >= max_data_entries:
                    break
        print(data_count)
    print(wrong)
    save_dir = config.dataset_dir / "save"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"train.json", 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, indent=4)
