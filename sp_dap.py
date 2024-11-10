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

template = """Imagine you are a travel agent negotiating with a potential traveler using an argumentation-based approach. The focus of the negotiation is on a {package_name} travel package described as {package_desc}, where both parties aim to reach a mutually beneficial agreement. Your task is to predict the dialogue act of
the target utterance based on the provided dialogue context. Dialog Act may include Negotiate-price-increase, Negotiate-price-decrease, Negotiate-price-nochange, Negotiate-add-X, Negotiate-remove-X, Concern-Price, Disagree-Price, Justify-Price, Assurance-Price, Disagree-X, Justify-X, Assurance-X, Greet-Ask, Inform, Elicit-Preference, Ask-Price, Tell-Price, Ask-Clarification-X, Provide-Clarification-X, Provide-Consent, Consent-Response, Accept, Acknowledge-Acceptance. Please start the output with Dialog Act Label:

The definition of dialog acts are as follows:
Negotiate-price-increase: Travel agent negotiates a higher price for a package or services.
Negotiate-price-decrease: Traveler negotiates a lower price for a package or service.
Negotiate-price-nochange: Price remains unchanged, emphasizing its value.
Negotiate-add-X: Proposes adding a feature or service during negotiation.
Negotiate-remove-X: Suggests removing a feature or service, potentially affecting price.
Concern-price: Expresses hesitation about the proposed price, signaling dissatisfaction.
Disagree-price: Rejects the proposed price, leading to further negotiation.
Justify-price: Provides reasoning to support the proposed price.
Assurance-price: Reassures the price is reasonable to address concerns.
Disagree-X: Objects to a feature or term, requiring changes for agreement.
Justify-X: Defends a feature’s inclusion with logical reasoning.
Assurance-X: Offers reassurance about the validity or quality of a feature.
Greet-Ask: Opens the conversation politely and asks for specific details.
Inform: Shares information about packages or services.
Elicit-preference: Expresses traveler’s preferences on features, services, or budget.
Ask-price: Seeks clarification on pricing.
Tell-price: States the proposed price.
Ask-clarification-X: Requests clarification on specific aspects of the deal.
Provide-clarification-X: Offers clarification about the deal or aspects of the negotiation.
Provide-consent: Expresses agreement or approval of a proposal.
Consent-response: Acknowledges the traveler’s approval and readiness to proceed.
Accept: Indicates agreement to the offer or deal.
Acknowledge-acceptance: Recognizes the other party's acceptance of the deal.

DialogueContext: {context}
TargetUtterance: {text}
Dialog Act Label:"""

prompt = PromptTemplate(template=template, input_variables=["package_name","package_desc","context","text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

def classify(package_name, package_desc, context, text):
    raw_llm_answer = llm_chain.run(package_name, package_desc, context, text)
    llm_answer = raw_llm_answer.lower()

    dialog_acts = [
    "negotiate-price-increase", "negotiate-price-decrease", "negotiate-price-nochange",
    "negotiate-add-x", "negotiate-remove-x", "concern-price", "disagree-price", "justify-price",
    "assurance-price", "disagree-x", "justify-x", "assurance-x", "greet-ask", "inform", 
    "elicit-preference", "ask-price", "tell-price", "ask-clarification-x", "provide-clarification-x", 
    "provide-consent", "consent-response", "accept", "acknowledge-acceptance"
    ]

    if llm_answer in dialog_acts:
        return llm_answer
    else:
        raise ValueError(f"Invalid response from the LLM. Response: {raw_llm_answer}")

# Load the CSV file
input_file = 'pact.csv'
output_file = 'pact-dialog-act.csv'
data = pd.read_csv(input_file)

predicted_data = []

# Annotate each utterance in the CSV
for _, row in tqdm(data.iterrows(), desc="Annotating Utterances"):
    package_name = row["package_name"]
    package_desc = row["package_desc"]
    context = row["context"]
    utterance = row["utterance"]

    dialog_act_label = classify(package_name, package_desc, context, utterance)

    predicted_data.append({
        "Context": context,
        "Utterance": utterance,
        "Dialog Act Label": dialog_act_label
    })


# Convert the list of annotated data to a DataFrame
predicted_df = pd.DataFrame(predicted_data)

# Save the annotated DataFrame to a new CSV file
predicted_df.to_csv(output_file, index=False)

print(f"Predicted data has been saved to {output_file}.")