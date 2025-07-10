import json
import os
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from datasets import load_dataset


def make_raw_jsonl(dataset_name):
    if dataset_name.lower() in ['math500']:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

        output_path = "/home/sohyun0423/APO-LRM/data/MATH/MATH500/test_raw.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for example in dataset:
                json_line = json.dumps(example, ensure_ascii=False)
                f.write(json_line + "\n")
        
    elif dataset_name.lower() in ['math']:
        root_dir = "/home/sohyun0423/APO-LRM/data/MATH/test"
        output_file = "/home/sohyun0423/APO-LRM/data/MATH/test_raw.jsonl"
        with open(output_file, "w", encoding="utf-8") as outfile:
            for subfolder in os.listdir(root_dir):
                subfolder_path = os.path.join(root_dir, subfolder)
                
                if os.path.isdir(subfolder_path):
                    for filename in os.listdir(subfolder_path):
                        if filename.endswith(".json"):
                            file_path = os.path.join(subfolder_path, filename)

                            with open(file_path, "r", encoding="utf-8") as infile:
                                data = json.load(infile)
                                json_line = json.dumps(data, ensure_ascii=False)
                                outfile.write(json_line + "\n")
        print(f"Merged JSONL saved to: {output_file}")


def preprocess_dataset_from_jsonl():
    input_path = "/home/sohyun0423/APO-LRM/data/MATH/test_raw.jsonl"
    output_path = "/home/sohyun0423/APO-LRM/data/MATH/test.jsonl"
    with open(input_path, "r", encoding="utf-8") as infile, \
        open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            example = json.loads(line)
            if 'answer' in example:
                label = example['answer']
            elif 'solution' in example:
                # answer_parsed = parse(
                #     example['solution'],
                #     extraction_config=[
                #         LatexExtractionConfig(
                #             normalization_config=NormalizationConfig(
                #                 nits=False,
                #                 malformed_operators=False,
                #                 basic_latex=True,
                #                 equations=True,
                #                 boxed="all",
                #                 units=True,
                #             ),
                #             # Ensures that boxed is tried first
                #             boxed_match_priority=0,
                #             try_extract_without_anchor=False,
                #         )
                #     ],
                #     extraction_mode="first_match",
                # )
                # if len(answer_parsed) > 0:
                #     label = answer_parsed[-1]
                # import pdb; pdb.set_trace()
                label = example['solution']
            new_example = {
                "label": label,
                "text": example["problem"]
            }

            outfile.write(json.dumps(new_example, ensure_ascii=False) + "\n")


def get_total_files_num():
    root_dir = "/home/sohyun0423/APO-LRM/data/MATH/test"
    total_count = 0
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolder_path):
            json_files = [f for f in os.listdir(subfolder_path) if f.endswith('.json')]
            total_count += len(json_files)
    print(f"Total .json files: {total_count}")


if __name__ == "__main__":
    # get_total_files_num()
    # make_raw_jsonl(dataset_name='MATH')
    # preprocess_dataset_from_jsonl()
    
    label = "The spinner is guaranteed to land on exactly one of the three regions, so we know that the sum of the probabilities of it landing in each region will be 1. If we let the probability of it landing in region $C$ be $x$, we then have the equation $1 = \\frac{5}{12}+\\frac{1}{3}+x$, from which we have $x=\\boxed{\\frac{1}{4}}$."
    # label_parsed = parse(label, extraction_mode="first_match")
    label_parsed = parse(
        label,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    import pdb; pdb.set_trace()
