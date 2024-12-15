

# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 0 --model_name Mistral-7B-Instruct-v0.3 --strategy direct_command --role teacher
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 0 --model_name Llama-3.1-8B-Instruct --strategy calc_and_explanation --role expert
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 0 --model_name Mistral-7B-Instruct-v0.3 --strategy direct_command --role expert
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 0 --model_name Llama-3.1-8B-Instruct --strategy calc_and_explanation --role teacher

# CUDA_VISIBLE_DEVICES=5 python prompting.py --method zero-shot --num_examples 0 --model_name GPT2-XL --strategy direct_command --role teacher
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method zero-shot --num_examples 0 --model_name GPT2-XL --strategy step_by_step --role teacher
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method zero-shot --num_examples 0 --model_name GPT2-XL --strategy calc_and_explanation --role teacher
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method cot_icl --num_examples 1 --model_name GPT2-XL --strategy calc_and_explanation --role expert
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method cot_icl --num_examples 2 --model_name GPT2-XL --strategy calc_and_explanation --role expert
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method cot_icl --num_examples 3 --model_name GPT2-XL --strategy calc_and_explanation --role expert

# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 0 --model_name GPT2-XL --strategy direct_command --role expert
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 0 --model_name GPT2-XL --strategy calc_and_explanation --role teacher
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 3 --model_name GPT2-XL --strategy direct_command --role expert
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 3 --model_name GPT2-XL --strategy calc_and_explanation --role teacher

# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 3 --model_name Llama-3.1-8B-Instruct --strategy calc_and_explanation --role teacher
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 3 --model_name Llama-3.1-8B-Instruct --strategy calc_and_explanation --role expert
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 0 --model_name Llama-3.1-8B-Instruct --strategy calc_and_explanation --role teacher
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 0 --model_name Llama-3.1-8B-Instruct --strategy calc_and_explanation --role expert
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 15 --model_name Mistral-7B-Instruct-v0.3 --strategy direct_command --role teacher
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 15 --model_name Mistral-7B-Instruct-v0.3 --strategy direct_command --role expert


# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 0 --model_name Mistral-7B-Instruct-v0.3 --strategy direct_command --role teacher
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 0 --model_name Mistral-7B-Instruct-v0.3 --strategy direct_command --role expert
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 0 --model_name Llama-3.1-8B-Instruct --strategy calc_and_explanation --role teacher
# CUDA_VISIBLE_DEVICES=5 python prompting.py --method role-play --num_examples 0 --model_name Llama-3.1-8B-Instruct --strategy calc_and_explanation --role expert

CUDA_VISIBLE_DEVICES=1 python prompt_symbolic.py --method cot_icl --num_examples 15 --model_name Llama-3.1-8B-Instruct --strategy calc_and_explanation --role expert
CUDA_VISIBLE_DEVICES=2 python prompt_symbolic.py --method cot_icl --num_examples 8 --model_name Llama-3.1-8B-Instruct --strategy calc_and_explanation --role expert
CUDA_VISIBLE_DEVICES=3 python prompt_symbolic.py --method cot_icl --num_examples 5 --model_name Llama-3.1-8B-Instruct --strategy calc_and_explanation --role expert
CUDA_VISIBLE_DEVICES=4 python prompt_symbolic.py --method cot_icl --num_examples 15 --model_name Mistral-7B-Instruct-v0.3 --strategy calc_and_explanation --role expert
CUDA_VISIBLE_DEVICES=5 python prompt_symbolic.py --method cot_icl --num_examples 8 --model_name Mistral-7B-Instruct-v0.3 --strategy calc_and_explanation --role expert
CUDA_VISIBLE_DEVICES=6 python prompt_symbolic.py --method cot_icl --num_examples 5 --model_name Mistral-7B-Instruct-v0.3 --strategy calc_and_explanation --role expert

