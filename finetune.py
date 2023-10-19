import json
import openai
import os
import sys
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_training_dataset(json_input_path, system_message):
    # Read input JSON content from qa.json
    with open(json_input_path, 'r') as input_file:
        input_json = json.load(input_file)
    # Create JSONL output structure
    output_jsonl_strs = []
    # Add QA pairs to the output JSONL structure
    for qa_pair in input_json["qa"]:
        messages = {
            "messages": [ 
                { "role": "system", "content": system_message },
                { "role": "user", "content": qa_pair["question"] },
                { "role": "assistant", "content": qa_pair["answer"] }
            ]
        }
        output_jsonl_strs.append(json.dumps(messages))

    # Write JSONL output to a file
    output_jsonl_path = "training_dataset.jsonl"
    with open(output_jsonl_path, 'w') as jsonl_file:
        jsonl_file.write("\n".join(output_jsonl_strs))

    return output_jsonl_path

def upload_training_dataset(output_jsonl_path):
    response = openai.File.create(file=open(output_jsonl_path, "rb"), purpose='fine-tune')
    print(response)

    uploaded_file_id = response["id"]
    print(f"Uploaded file ID: {uploaded_file_id}")
    return uploaded_file_id

def create_finetuning_job(uploaded_file_id):
    response = openai.FineTuningJob.create(training_file=uploaded_file_id, model="gpt-3.5-turbo")
    job_id = response["id"]
    print(response)
    print(f"Job ID: {job_id}")
    return job_id

def retrieve_finetuning_job(job_id):
    response = openai.FineTuningJob.retrieve(job_id)
    print(response)
    print(f"Job {job_id} status: {response['status']}")

def ask(model, system_message, prompt):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[ {"role": "system", "content": system_message}, {"role": "user", "content": prompt} ]
    )
    print(completion.choices[0].message)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify an operation")
        sys.exit(1)

    operation = sys.argv[1]

    if operation == "generate_training_dataset":
        generate_training_dataset(sys.argv[2], sys.argv[3])
    elif operation == "upload_training_dataset":
        upload_training_dataset(sys.argv[2])
    elif operation == "create_finetuning_job":
        create_finetuning_job(sys.argv[2])
    elif operation == "retrieve_finetuning_job":
        retrieve_finetuning_job(sys.argv[2])
    elif operation == "ask":
        model = sys.argv[2]
        system_message = sys.argv[3]
        while True:
            user_input = input("Please enter your question: ")
            ask(model, system_message, user_input)
    else:
        print("Invalid operation")