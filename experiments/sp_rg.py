import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from tqdm import tqdm
import math

# Argument parser to take model name and output file name as inputs
parser = argparse.ArgumentParser(description='Generate responses using a specified model.')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model to be used')
parser.add_argument('--output_file', type=str, required=True, help='Name of the output file to save responses')
args = parser.parse_args()

######################### Setup Prompts ##########################

df = pd.read_csv('test_set.csv')

system = """Imagine you are a travel agent negotiating with a potential traveler using an argumentation-based approach. The focus of the negotiation is on a travel package described as , where both parties aim to reach a mutually beneficial agreement. Your role as the travel agent is to present compelling arguments, address the traveler’s concerns, and strategically persuade them while
maintaining professionalism and flexibility. Given the dialogue context, your task is to generate a coherent and contextually relevant response. Please begin your response with 'Response:' """


def create_prompt(system_prompt, context):
    prompt = f'''[System] {system_prompt} [Context] {context} [Response]'''
    return prompt

for i, row in df.iterrows():
    sys = system
    conv_id = row['conv_id']
    context = row['context'].replace('[SOC]', '').replace('[EOC]', '').strip()
    gold = row['response'].replace('[SOR]', '').replace('[EOR]', '').strip()

    prompt1 = create_prompt(sys, context)

    prompts.append((conv_id, context, gold, prompt1))

######################### Load Model ##########################

access_token = ""
device = 'auto'

model_name = args.model_name

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map=device,
    use_auth_token=access_token,
    trust_remote_code=True
)

special_tokens = ['[System]', '[Context]', '[Response]', '[Agent]', '[Traveler]', '[SOC]', '[EOC]', '[SOR]', '[EOR]']

special_tokens_dict = {'additional_special_tokens': special_tokens}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token_id = tokenizer.eos_token_id
model.resize_token_embeddings(len(tokenizer))

generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.01,
    trust_remote_code=True,
    device_map="auto"    # finds GPU
)

for conv_id, context, gold, prompt in tqdm(prompts):
    response = generation_pipe(
        prompt,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_response = response[0]['generated_text'].split('[Response]')[1].split('\n')[0]
    print(generated_response)
    results.append({'conv_id': conv_id, 'context': context, 'gold': gold, 'generated': generated_response})

results_df = pd.DataFrame(results)
results_df.to_csv(args.output_file, index=False)
