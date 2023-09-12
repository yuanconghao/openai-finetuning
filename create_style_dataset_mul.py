import concurrent
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
import json
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

dataset = load_dataset('json', data_files="./data/question_med.json")
print(dataset)
dataset_splits = {"train": dataset["train"]}
print(dataset_splits)
# num_rows = dataset_splits['train'].num_rows

jinyong = 'jinyong'
style_map = {
    jinyong: "金庸",
}

style = jinyong
model_id = "gpt-3.5-turbo"
max_workers = 10


def get_assistant(question):
    system_c = f"你是我的私人医生助手，你要用{style_map[style]}的风格回答我健康问题。"
    system_content = {
        'role': 'system',
        'content': system_c,
    }
    user_content = {
        'role': 'user',
        'content': question,
    }
    # [{"role": "system", "content": "你是我的私人医生助手，你要回答我的健康问题。"}, {"role": "user", "content": "一个患者的卵巢小细胞癌转移至其它部位，是否有必要进行手术治疗？"}, {"role": "assistant", "content": "亲爱的主人，要多注意身体啊。当卵巢小细胞癌转移至其它部位时，手术治疗的效果可能不理想，因此一般不推荐进行手术治疗。针对转移病灶，可以采用化疗、放疗等治疗手段进行综合治疗。"}]
    query_messages = [system_content, user_content]
    print(query_messages)

    response = openai.ChatCompletion.create(
        model=model_id, messages=query_messages, temperature=0, max_tokens=500
    )
    answer = response["choices"][0]["message"]["content"]
    return answer


def get_assistant_retry(question):
    system_c = f"你是我的私人医生助手，你要用{style_map[style]}的风格回答我健康问题。"
    system_content = {
        'role': 'system',
        'content': system_c,
    }
    user_content = {
        'role': 'user',
        'content': question,
    }
    query_messages = [system_content, user_content]
    print(query_messages)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model_id, messages=query_messages, temperature=0, max_tokens=500
            )
            answer = response["choices"][0]["message"]["content"]
            return answer
        except openai.error.OpenAIError:  # Catch OpenAI specific errors
            if attempt < max_retries - 1:  # i.e. not the last attempt
                print(f"Attempt {attempt + 1} failed. Retrying...")
                continue
            else:
                print(f"Attempt {attempt + 1} failed. No more retries.")
                return "Error: Unable to fetch response from OpenAI."


def process_item(item):
    newitem = {}
    messages = []
    system_content = {
        'role': 'system',
        'content': '你是我的私人医生助手，你要回答我的健康问题。'
    }
    user_content = {
        'role': 'user',
        'content': item['instruction'],
    }
    # answer = get_assistant(item['instruction'])
    answer = get_assistant_retry(item['instruction'])
    assistant_content = {
        'role': 'assistant',
        'content': "少侠保重身体。" + answer,
    }
    messages.append(system_content)
    messages.append(user_content)
    messages.append(assistant_content)
    newitem['messages'] = messages
    return json.dumps(newitem, ensure_ascii=False) + "\n"


def main():
    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    with open(f"dataset/train_{style}.jsonl", "a") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and store the Future objects in a list
            futures = [executor.submit(process_item, item) for item in dataset_splits["train"]]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()  # Get the result (or exception) from the task
                f.write(result)


if __name__ == "__main__":
    main()
