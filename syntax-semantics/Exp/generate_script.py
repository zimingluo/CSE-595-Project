import random
import math
import json
import argparse
import jsonlines
import os

with open('./gsm8k-train-round2-0_999-processed.jsonl', 'r') as json_file:
    json_list = list(json_file)

data = []
for json_str in json_list:
    item = json.loads(json_str)
    data.append(item)


template = """
import random
import math
import json
import argparse
import jsonlines
import os

random.seed(42) # Consistent random generation

first_names = []
with jsonlines.open('../data/top_first_names.jsonl') as reader:
    for line in reader:
        first_names.append(line['first_name'])

last_names = []
with jsonlines.open('../data/top_last_names.jsonl') as reader:
    for line in reader:
        last_names.append(line['last_name'])

items = []
with jsonlines.open('../data/items-llm.jsonl') as reader:
    for line in reader:
        items.append(line)

places = []
with jsonlines.open('../data/places-llm.jsonl') as reader:
    for line in reader:
        places.append(line)

us_counties = []
with jsonlines.open('../data/us_counties.jsonl') as reader:
    for line in reader:
        us_counties.append(line)

{code}

parser = argparse.ArgumentParser(description="Generate problems and solutions.")
parser.add_argument("--num_problems", type=int, default=100, help="Number of problems to generate")

args = parser.parse_args()
NUM_PROBLEMS = args.num_problems

"""


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate code templates.")
    parser.add_argument("--template_idx", type=int, default=0, help="Number of problems to generate")

    args = parser.parse_args()
    template_idx = args.template_idx

    os.makedirs('./templates', exist_ok=True)

    for template_idx in range(100, 200):
        template_code = template.format(code=data[template_idx]['template']) + f"""
if __name__ == "__main__":
    os.makedirs('./datasets', exist_ok=True)
    # output jsonl file
    with open(f'./datasets/gsm-{template_idx:04}-NUM{{NUM_PROBLEMS}}.jsonl', 'w') as f:
        for i in range(NUM_PROBLEMS):
            problem, solution_code, result, solution_wocode = generate_problem_and_solution_code()
            # Write problem to file
            f.write(json.dumps({{"problem": problem, "solution_code": solution_code, "solution_wocode": solution_wocode, "result": str(result), "template_name": "gsm-0000-1", "idx": i}}) + '\\n')
"""

        filename = f"gsm-{template_idx:04}-100.py"

        with open(os.path.join('./templates', filename), 'w') as file:
            file.write(template_code)
