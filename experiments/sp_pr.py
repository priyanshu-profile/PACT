from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd
from tqdm import tqdm


model_name = input("Enter the LLM name (e.g., 'meta-llama/Llama-2-7b-hf'): ")
access_token = input("Enter the access token (e.g., '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'), else enter 'None': ")

if access_token = "None": 
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
else: 
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_length=512,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

from langchain import PromptTemplate,  LLMChain

template = """Imagine you are a travel agent negotiating with a potential traveler using an argumentation-based approach. The focus of the negotiation is on a {package_name} travel package described as {package_desc}, where both parties aim to reach a mutually beneficial agreement. Given the conversation between traveler
and travel agent, your task is to predict the argumentation profile, preference profile, and buying style profile of the traveler and argumentation profile of the travel agent. Argumentation profile of the traveler can be one of the following - Agreeable or Disagreeable, Preference profile of the traveler can be one of the following - Culture Creature, Action Agent, Avid Athlete, Thrill Seeker, Trail Trekker, Escapist, Shopping Shark, Boater, Sight Seeker, or Beach Lover, Buying style profile of the traveler can be one of the following - Quality-concerned, Budget-concerned, and Budget-&-Quality-concerned, and the Argumentation
profile of the travel agent can be one of the following - Open-minded and Argumentative.

The definition of profiles are as follows:
Agreeable: Interlocutors who readily accept offers and arguments, aiming for cooperation and consensus, with minimal conflict or challenges.
Disagreeable: Interlocutors who often reject proposals unless convinced, adopting a critical stance to ensure their own interests.
Open-minded: Interlocutors who evaluate offers critically yet constructively, balancing cooperation and skepticism to maintain productive dialogue.
Argumentative: Interlocutors who frequently challenge and counter offers, engaging in intense debate to assert their position.
Culture Creature: Prefers cultural experiences like theater, museums, monuments, art exhibitions, and local festivals.
Action Agent: Enjoys lively environments with nightclubs, upscale restaurants, and entertainment venues.
Avid Athlete: Stays active on vacation, engaging in sports like golf and tennis.
Thrill Seeker: Seeks high-adrenaline activities such as skydiving, bungee jumping, or extreme sports.
Trail Trekker: Enjoys outdoor activities like hiking, exploring parks, and connecting with nature.
Escapist: Seeks peaceful retreats, valuing relaxation in tranquil environments.
Shopping Shark: Favors destinations with vibrant shopping areas and local markets.
Boater: Prefers water-based travel with a boat as home, exploring coastal or lakefront areas.
Sight Seeker: Enjoys exploring landmarks, scenic views, and attractions along the journey.
Beach Lover: Prefers sunbathing and relaxation in warm, sandy beach destinations.
Quality-Concerned Traveler: Prioritizes high standards in amenities and services, valuing quality over cost.
Budget-Concerned Traveler: Seeks cost-effective options, emphasizing value for money.
Budget-&-Quality-Concerned Traveler: Balances quality and cost, choosing well-reviewed options within budget constraints.

Please give the output as 'Traveler Argumentation Profile: \n Traveler Preference Profile: \n Traveler
Buying Style Profile: \n Travel Agent Argumentation Profile:'

Dialogue: {text}
Traveler Argumentation Profile:
Traveler Preference Profile:
Traveler Buying Style Profile:
Travel Agent Argumentation Profile: """

prompt = PromptTemplate(template=template, input_variables=["package_name","package_desc","text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

def classify(package_name, package_desc, text):
    raw_llm_answer = llm_chain.run(package_name, package_desc, text)
    llm_answer = raw_llm_answer.lower()

    return llm_answer

# Load the CSV file
input_file = 'pact.csv'
output_file = 'pact-profile.csv'
data = pd.read_csv(input_file)

predicted_data = []

# Annotate each utterance in the CSV
for _, row in tqdm(data.iterrows(), desc="Annotating Utterances"):
    package_name = row["package_name"]
    package_desc = row["package_desc"]
    dialogue = row["dialogue"]

    profiles = classify(package_name, package_desc, dialogue)

    predicted_data.append({
        "Dialogue": dialogue,
        "Profiles": profiles
    })


# Convert the list of annotated data to a DataFrame
predicted_df = pd.DataFrame(predicted_data)

# Save the annotated DataFrame to a new CSV file
predicted_df.to_csv(output_file, index=False)

print(f"Predicted data has been saved to {output_file}.")