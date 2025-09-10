import argparse
import json
import re
from transformers import AutoTokenizer

import config


def load_txt(datasets, path, max_data_entries):
    tokenizer = AutoTokenizer.from_pretrained(path)
    max_token_length = 2048
    wrong = 0
    data_list = []
    regenerate = []
    for dataset in datasets:
        json_dir = config.dataset_dir / dataset / "prompt" / "txt"
        prompt_dir = json_dir / "prompt_grd"
        prompt_dir2 = json_dir / "prompt_com"

        filenames1 = list(prompt_dir.glob("*.txt"))
        filenames2 = list(prompt_dir2.glob("*.txt"))
        data_count = 0
        for file in filenames1:
            with open(file, 'r', encoding='utf-8', errors='ignore') as file:

                try:
                    file_content = file.read()
                except UnicodeDecodeError:
                    regenerate.append(file)

                instruction_pattern = re.compile(r'instruction:(.*?)(?=input:|$)', re.DOTALL)
                input_pattern = re.compile(r'input:(.*?)(?=output:|$)', re.DOTALL)
                output_pattern = re.compile(r'output:(.*?)(?=$)', re.DOTALL)

                instruction_match = instruction_pattern.search(file_content)
                input_match = input_pattern.search(file_content)
                output_match = output_pattern.search(file_content)

                instruction = instruction_match.group(1).strip() if instruction_match else ""
                input_string = input_match.group(1).strip() if input_match else ""
                output = output_match.group(1).strip() if output_match else ""

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
        print(f"{dataset}grd: {data_count}")
        data_count = 0
        for file in filenames2:
            with open(file, 'r', encoding='utf-8') as file:
                try:
                    file_content = file.read()
                except UnicodeDecodeError:
                    regenerate.append(file)

                instruction_pattern = re.compile(r'instruction:(.*?)(?=input:|$)', re.DOTALL)
                input_pattern = re.compile(r'input:(.*?)(?=output:|$)', re.DOTALL)
                output_pattern = re.compile(r'output:(.*?)(?=$)', re.DOTALL)

                instruction_match = instruction_pattern.search(file_content)
                input_match = input_pattern.search(file_content)
                output_match = output_pattern.search(file_content)

                instruction = instruction_match.group(1).strip() if instruction_match else ""
                input_string = input_match.group(1).strip() if input_match else ""
                output = output_match.group(1).strip() if output_match else ""

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
        print(f"{dataset}com: {data_count}")
    print(f"wrong: {wrong}")
    return data_list

def main(args):
    train_datasets = [f"{args.dataset}-{i}" for i in range(args.min_args, args.max_args+1)]

    data = load_txt(train_datasets,args.llm_path, args.data_num)

    save_dir = config.dataset_dir / "save"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / f"{args.dataset}.json", 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Data max length",
    )

    parser.add_argument(
        "--data_num", type=int, default=3000, help="Number of arguments"
    )

    parser.add_argument(
        "--min_args", type=int, default=6, help="Minimum number of arguments"
    )
    parser.add_argument(
        "--max_args", type=int, default=25, help="Maximum number of arguments"
    )
    parser.add_argument(
        "--dataset", type=str, default="train"
    )
    parser.add_argument(
        "--llm_path", type=str, default="Meta-Llama-3.1-8B-Instruct"
    )
    args = parser.parse_args()
    main(args)


