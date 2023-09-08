import os
import openai
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = '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'

job_id =''

response = openai.FineTuningJob.retrieve(job_id)

print("Job ID:", response["id"])
print("Status:", response["status"])
print("Trained Tokens:", response["trained_tokens"])

response = openai.FineTuningJob.list_events(id=job_id, limit=50)

events = response["data"]
events.reverse()

for event in events:
    print(event["message"])


response = openai.FineTuningJob.retrieve(job_id)
fine_tuned_model_id = response["fine_tuned_model"]

print("Fine-tuned model ID:", fine_tuned_model_id)