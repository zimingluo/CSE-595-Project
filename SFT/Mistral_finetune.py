# from unsloth import FastLanguageModel, is_bfloat16_supported
# import json
# import os
# import torch
# from datasets import Dataset
# from trl import SFTTrainer
# from transformers import TrainingArguments

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# max_seq_length = 10000 # Choose any! We auto support RoPE Scaling internally!
# dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
# load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

# # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# fourbit_models = [
#     "unsloth/mistral-7b-bnb-4bit",
#     "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
#     "unsloth/llama-2-7b-bnb-4bit",
#     "unsloth/llama-2-13b-bnb-4bit",
#     "unsloth/codellama-34b-bnb-4bit",
#     "unsloth/tinyllama-bnb-4bit",
#     "unsloth/gemma-7b-bnb-4bit", # New Google 6 trillion tokens model 2.5x faster!
#     "unsloth/gemma-2b-bnb-4bit",
# ] # More models at https://huggingface.co/unsloth

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/mistral-7b-v0.3", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
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


# print(torch.cuda.current_device())


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

# dataset = dataset.map(formatting_prompts_func, batched = True,)
# print(dataset[0]['text'])




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
#         max_steps = 100, # Set num_train_epochs = 1 for full training runs
#         learning_rate = 2e-4,
#         fp16 = not is_bfloat16_supported(),
#         bf16 = is_bfloat16_supported(),
#         logging_steps = 1,
#         optim = "adamw_8bit",
#         weight_decay = 0.01,
#         lr_scheduler_type = "linear",
#         seed = 3407,
#         output_dir = "outputs",
#         report_to = "none", # Use this for WandB etc
#     ),
# )


# #@title Show current memory stats
# gpu_stats = torch.cuda.get_device_properties(0)
# start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
# print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
# print(f"{start_gpu_memory} GB of memory reserved.")


# trainer_stats = trainer.train()

# #@title Show final memory and time stats
# used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
# used_percentage = round(used_memory         /max_memory*100, 3)
# lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
# print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
# print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
# print(f"Peak reserved memory = {used_memory} GB.")
# print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
# print(f"Peak reserved memory % of max memory = {used_percentage} %.")
# print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# model.save_pretrained("Mistral_finetuned") # Local saving
# tokenizer.save_pretrained("Mistral_finetuned")


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
max_seq_length = 20000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Mistral_finetuned", # YOUR MODEL YOU USED FOR TRAINING
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
with open('./results/mistral_answers.pkl', 'wb') as file:
    pickle.dump(answers, file)

# print(answer)


# # Load the list from the file
# with open('./results/mistral_answer1.pkl', 'rb') as file:
#     answers1 = pickle.load(file)
