import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

training_file_id = 'file-bODqHyiOqQmfEWvqOL69wcFq'
validation_file_id = 'file-PaH8K7u3y7XY877kVOuCnbQR'

response = openai.FineTuningJob.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-3.5-turbo",
    suffix="recipe-ner",
)

job_id = response["id"]

print("Job ID:", response["id"])
print("Status:", response["status"])


