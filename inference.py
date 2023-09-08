import json
import os
import openai
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = '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'

fine_tuned_model_id = ''

test_messages = '{"messages":[{"role":"system","content":"你是我的私人医生助手，你要回答我的健康问题。"},{"role":"user","content":"一名年龄在70岁的女性，出现了晕厥、不自主颤抖、情绪不稳等症状，请详细说明其手术治疗和术前准备。"}]}'

test_messages = json.loads(test_messages)

print(test_messages)

response = openai.ChatCompletion.create(
    model=fine_tuned_model_id, messages=test_messages, temperature=0, max_tokens=500
)
print(response["choices"][0]["message"]["content"])