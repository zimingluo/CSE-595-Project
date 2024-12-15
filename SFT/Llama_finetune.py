# from unsloth import FastLanguageModel
# import torch
# max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
# dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
# load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )


# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",],
#     lora_alpha = 16,
#     lora_dropout = 0, # Supports any, but = 0 is optimized
#     bias = "none",    # Supports any, but = "none" is optimized
#     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
#     use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
#     random_state = 3407,
#     use_rslora = False,  # We support rank stabilized LoRA
#     loftq_config = None, # And LoftQ
# )

# import json
# import os
# from datasets import Dataset


# directory_path = '../syntax-semantics/Exp/datasets/'

# examples = []

# for filename in os.listdir(directory_path):
#     if filename.endswith('.jsonl') and "NUM100" in filename:
#         # print(filename)
#         # continue
#         file_path = os.path.join(directory_path, filename)
#         with open(file_path, 'r') as file:
#             for line in file:
#                 examples.append(json.loads(line))

# del examples[6100:6200]

# dataset = Dataset.from_list(examples)

# gsm_prompt = """
# Instruction:
# Below is a math question. Think through this question carefully and write a step-by-step response that appropriately answer this question. Please provide the final answer after ####.

# Question:
# {}

# Solution:
# {} 
# #### {}
# """


# EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
# def formatting_prompts_func(examples):
#     problems = examples["problem"]
#     outputs = examples["solution_wocode"]
#     answers = examples["result"]
#     texts = []
#     for problem, output, answer in zip(problems, outputs, answers):
#         # Must add EOS_TOKEN, otherwise your generation will go on forever!
#         text = gsm_prompt.format(problem, output, answer).strip() + EOS_TOKEN
#         texts.append(text)
#     return { "text" : texts, }
# pass

# dataset = dataset.map(formatting_prompts_func, batched = True,)
# print(dataset[0]['text'])

# from trl import SFTTrainer
# from transformers import TrainingArguments
# from unsloth import is_bfloat16_supported

# trainer = SFTTrainer(
#     model = model,
#     tokenizer = tokenizer,
#     train_dataset = dataset,
#     dataset_text_field = "text",
#     max_seq_length = max_seq_length,
#     dataset_num_proc = 2,
#     packing = False, # Can make training 5x faster for short sequences.
#     args = TrainingArguments(
#         per_device_train_batch_size = 2,
#         gradient_accumulation_steps = 4,
#         warmup_steps = 5,
#         num_train_epochs = 1, # Set this for 1 full training run.
#         max_steps = 100,
#         learning_rate = 2e-4,
#         fp16 = not is_bfloat16_supported(),
#         bf16 = is_bfloat16_supported(),
#         logging_steps = 1,
#         optim = "adamw_8bit",
#         weight_decay = 0.01,
#         lr_scheduler_type = "linear",
#         seed = 3407,
#         output_dir = "outputs_llama3.1",
#         report_to = "none", # Use this for WandB etc
#     ),
# )

# trainer_stats = trainer.train()

# model.save_pretrained("Llama_finetuned") # Local saving
# tokenizer.save_pretrained("Llama_finetuned")
# # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving



# Load GSM8K dataset
import json
import os

directory_path = '../syntax-semantics/Exp/datasets/'
testdata = []

for filename in os.listdir(directory_path):
    if "NUM10.jsonl" in filename:
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as file:
            for line in file:
                testdata.append(json.loads(line))


gsm_prmpts = []
for data in testdata:
    prompt = \
f"""
Instruction:
Below is a math question. Think through this question carefully and write a step-by-step response that appropriately answer this question. Please provide the final answer after ####.

Question:
{data['problem']}
"""
    gsm_prmpts.append(prompt.strip())

from unsloth import FastLanguageModel

from unsloth import FastLanguageModel
max_seq_length = 20000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Llama_finetuned", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference


answers = []
for prompt in gsm_prmpts:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 500, use_cache = True)
    answer = tokenizer.batch_decode(outputs)[0]
    answers.append(answer)
    # print(extract_numbers(answers))



import pickle

# Load the list from the file
with open('./results/llama_answers.pkl', 'wb') as file:
    pickle.dump(answers, file)



