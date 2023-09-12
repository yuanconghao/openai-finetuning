from datasets import load_dataset
import json
import os

dataset = load_dataset("./data/")
# print(dataset)

dataset_splits = {"train": dataset["train"], "test": dataset['test']}
print(dataset_splits)
#num_rows = dataset_splits['train'].num_rows


def main():
    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    for key, ds in dataset_splits.items():
        with open(f"dataset/{key}.jsonl", "w") as f:
            for item in ds:
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
                assistant_content = {
                    'role': 'assistant',
                    'content': "亲爱的主人，要多注意身体啊。" + item['output'],
                }
                messages.append(system_content)
                messages.append(user_content)
                messages.append(assistant_content)
                newitem['messages'] = messages
                f.write(json.dumps(newitem, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
