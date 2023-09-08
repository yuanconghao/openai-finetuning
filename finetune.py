import os
import openai
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = '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'

training_file_name = 'datasets/train.jsonl'
validation_file_name = 'datasets/test.jsonl'

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

response = openai.FineTuningJob.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-3.5-turbo",
    suffix="recipe-ner",
)

job_id = response["id"]

print("Job ID:", response["id"])
print("Status:", response["status"])


