import json
import re

from tqdm import tqdm
from transformers import AutoTokenizer

import config


tokenizer = AutoTokenizer.from_pretrained("Meta-Llama-3.1-8B-Instruct")

if __name__ == "__main__":
    dataset_names = [f"train-{i}" for i in range(5, 26)]
    grd_result=[]
    com_result=[]
    all_result = []
    for dataset_name in dataset_names:
        data_list_grd = []
        data_list_com = []
        json_dir = config.dataset_dir / dataset_name / "prompt" / "txt"
        prompt_dir = json_dir / "prompt__grd"
        prompt_dir2 = json_dir / "prompt__com"

        filenames = list(prompt_dir.glob("*.txt")) + list(prompt_dir2.glob("*.txt"))
        for file in filenames:
            with open(file, 'r', encoding='utf-8') as file:
                file_content = file.read()
                
                instruction_pattern = re.compile(r'instruction:(.*?)(?=input:|$)', re.DOTALL)
                input_pattern = re.compile(r'input:(.*?)(?=output:|$)', re.DOTALL)
                output_pattern = re.compile(r'output:(.*?)(?=$)', re.DOTALL)

                instruction_match = instruction_pattern.search(file_content)
                input_match = input_pattern.search(file_content)
                output_match = output_pattern.search(file_content)

                instruction = instruction_match.group(1).strip() if instruction_match else ""
                input_string = input_match.group(1).strip() if input_match else ""
                output = output_match.group(1).strip() if output_match else ""

                if "solving the grounded extension of" in instruction:
                    data_list_grd.append({
                        "instruction": instruction,
                        "input": input_string,
                        "output": output,
                    })

                elif "solving complete extensions of" in instruction:
                    data_list_com.append({
                        "instruction": instruction,
                        "input": input_string,
                        "output": output,
                    })
        max_token_length = 2048
        max_data_entries = 100
        dataset_name = "train"
        json_dir = config.dataset_dir / dataset_name / "prompt"

        result = []

        counter = 0
        print(len(data_list_grd))
        for item in tqdm(data_list_grd, desc="Processing", unit="item"):
            instruction = item["instruction"]
            input_text = item["input"]
            output_text = item["output"]
            if (len(tokenizer.tokenize(instruction)) + len(tokenizer.tokenize(input_text)) + len(
                    tokenizer.tokenize(output_text)) <= max_token_length):
                result.append(item)
                counter += 1
            else:
                print(len(tokenizer.tokenize(instruction)) + len(tokenizer.tokenize(input_text)) + len(
                    tokenizer.tokenize(output_text)))
            if counter >= max_data_entries:
                break
        print(f"num:{counter}")
        grd_result += result
        counter = 0
        result = []
        print(len(data_list_com))
        for item in tqdm(data_list_com, desc="Processing", unit="item"):
            instruction = item["instruction"]
            input_text = item["input"]
            output_text = item["output"]
            if (len(tokenizer.tokenize(instruction)) + len(tokenizer.tokenize(input_text)) + len(
                    tokenizer.tokenize(output_text)) <= max_token_length):
                result.append(item)
                counter += 1
            else:
                print(len(tokenizer.tokenize(instruction)) + len(tokenizer.tokenize(input_text)) + len(
                    tokenizer.tokenize(output_text)))
            if counter >= max_data_entries:
                break
        print(f"num:{counter}")
        com_result += result
    save_dir = config.dataset_dir / "save"
    all_result += grd_result
    all_result += com_result
    with open(save_dir / f"train.json", 'w', encoding='utf-8') as json_file:
        json.dump(all_result, json_file, indent=4)
