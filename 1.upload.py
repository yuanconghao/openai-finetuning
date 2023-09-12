import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

training_file_name = 'dataset/train.jsonl'
validation_file_name = 'dataset/test.jsonl'

training_response = openai.File.create(
    file=open(training_file_name, "rb"), purpose="fine-tune"
)
training_file_id = training_response["id"]

validation_response = openai.File.create(
    file=open(validation_file_name, "rb"), purpose="fine-tune"
)
validation_file_id = validation_response["id"]

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)


