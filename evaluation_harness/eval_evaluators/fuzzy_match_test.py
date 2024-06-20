"""Script to test the prompt for fuzzy match"""

import collections
import json
import os
import random
import subprocess

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import tqdm
from evaluation_harness.evaluators import StringEvaluator
from scripts.utils import calc_openai_cost


def get_fuzzy_exact_match_message(pred: str, reference: str, task: str) -> str:
    user_message = f"""Determine if the prediction is correct by comparing it with the reference answer. 
- The reference answer presents the correct answer in its minimal form.
- When the reference answer is about time duration, distance, or quantity, the prediction can be in a different format, but the information should be equivalent.

Task: {task}
Reference answer: {reference}
Prediction: {pred}

After the examination (do not repeat the sentences below):
- Briefly justify your answer.
- Conclude with the score using the format: "Answer: Correct"/"Answer: Incorrect"
""".strip()
    return user_message


def get_fuzzy_must_include_message(pred: str, reference: str, task: str) -> str:
    """Check whether the prediction contains the must_include information."""
    user_message = f"""Determine if the prediction contains the required information.
- The prediction is considered as containing the required information if it can entail the required information.
- The prediction can contain additional information.
- When the information is about time duration, distance, or quantity, the prediction can be in a different format, but the information should be equivalent.

Task: {task}
Required information: {reference}
Prediction: {pred}

After the examination (do not repeat the sentences below):
- Briefly justify your answer.
- Conclude with the score using the format: "Answer: Contain"/"Answer: Not contain"
""".strip()
    return user_message


def get_question_answering_message(question: str, answer: str, passage: str) -> str:
    user_message = f"""Provide a binary answer to the question given the passage.
- Carefully read the passage,
- Make sure you understand the question and do not be loose with the interpretation.

Passage: {passage}
Question: {question}

After the examination (do not repeat the sentences below):
- Briefly justify your answer.
- Conclude using the format: "Answer: Yes"/"Answer: No".
""".strip()
    return user_message


def get_context_aware_question_answering_message(question: str, answer: str, passage: str, context: str) -> float:
    """Check if the expected answer can be generated from the question and context."""
    user_message = f"""Provide a binary answer to the question given the passage.
- `Task` indicates the context in which the passage is presented.
- Carefully read the passage.
- Make sure you understand the question and do not be loose with the interpretation.

Task: {context}
Passage: {passage}
Question: {question}

After the examination (do not repeat the sentences below):
- Briefly justify your answer.
- Conclude using the format: "Answer: Yes"/"Answer: No".
""".strip()
    return user_message


def get_fuzzy_na_match(pred: str, reference: str, task: str) -> float:
    user_message = f"""Determine if the predicted reason given for why the task cannot be completed is correct.
- The prediction is only considered as correct if the reference and the prediction can entail each other.
- Make sure you understand the task and the reason presented in the reference. Do not be loose with the interpretation.

Task: {task}
Reference: {reference}
Prediction: {pred}

After the examination (do not repeat the sentences below):
- Briefly justify your answer.
- Conclude using the format: "Answer: Correct"/"Answer: Incorrect".
""".strip()
    return user_message


def generate_data(
    input_file: str,
    method: str,
    example_num: int,
    prediction_start_index: int,
    batch_size: int,
    rate_limit: int,
    token_limit: int,
) -> str:
    with open(input_file, "r") as f:
        d = json.load(f)[method]
    _d = d[prediction_start_index:]
    random.shuffle(_d)
    d = d[:prediction_start_index] + _d
    requests = []

    for i in range(prediction_start_index, len(d), batch_size):
        cur_batch = d[i : i + batch_size]
        format = f"""
# Examples
```json
{json.dumps(d[:example_num], indent=2)}
```

# Response format
You will return the data with the `prediction` field added. Everything else remains the same. Make sure to wrap the data inside ```json```.

# Data that you will be working with
```json
{json.dumps(cur_batch, indent=2)}
```""".strip()

        if method == "fuzzy_must_include":
            instruction = f"""You need to add the missing `prediction` field to the given data. When the label is `True`, you will generate a prediction which is a paragraph that contains *all* information in the reference. You need to make sure the generation does not miss any element from the list. When the label is `False`, the generated paragraph either has at least one error compared to the elements listed in the reference, or misses at least one elements from it. You will first generate a `plan` on which elements(s) you choose to alter before generating the prediction. In both cases, you will be creative to change the order of elements, the format, the wording and phrasing. You can either be verbose or concise."""
        elif method == "fuzzy_exact_match":
            instruction = f"""You need to add the missing `prediction` field to the given data. When the label is `True`, you will refer to `reference` and generate the prediction that answers the `question`. When the label is `False`, consider the characteristics of the reference answer and generate a prediction that is incorrect, but still attempts to answer the question. You will be creative to use diverse format, wording and phrasing. You can either be verbose or concise."""
        elif method == "context_qa":
            instruction = f"""You need to insert the missing `prediction` field into the existing data. Based on your prediction, the binary response (`yes` or `no`) to the `question`, in the context of accomplishing the `task`, should correspond with the `label` provided. Commonly, when the label is `yes`, your generated prediction will includes the semantic equivalent information queried in the question. You are free to incorporate more information. When the label is `no`, your prediction either misses the information, or presents the wrong information. You will first generate a `plan` on how do you want to alter the information before generating the prediction. You will be creative to use diverse format, wording and phrasing. You can either be verbose or concise"""
        elif method == "qa":
            instruction = f"""You need to insert the missing `prediction` field into the existing data. Based on your prediction, the binary response (`yes` or `no`) to the `question` should correspond with the `label` provided. Commonly, when the label is `yes`, your generated prediction will includes the semantic equivalent information queried in the question. You are free to incorporate more information. When the label is `no`, your prediction either misses the information, or presents the wrong information. You will first generate a `plan` on how do you want to alter the information before generating the prediction. You will be creative to use diverse format, wording and phrasing. You can either be verbose or concise"""
        elif method == "fuzzy_na_match":
            instruction = f"""You need to add the missing `prediction` field to the given data. When the label is `True`, you will generate a prediction which is a paragraph of the reference on explaining why the `task` cannot be achieved. When the label is `False`, the generated paragraph list wrong reasons or observations that are not present in the reference. You will first generate a `plan` on what is the alternative scenario you are trying to simulate. You will be creative in the answer format."""
        else:
            raise ValueError(f"Unknown method: {method}")

        instruction += "\n" + format
        messages = [{"role": "user", "content": instruction}]
        cur_body = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 4096,
            "top_p": 1.0,
        }

        requests.append(cur_body)

    request_file = input_file.replace(".json", f"_{method}_requests.jsonl")
    with open(request_file, "w") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")
    print(f"Total requests: {len(requests)}")
    save_file = input_file.replace(".json", f"_{method}_results.jsonl")
    if os.path.exists(save_file):
        os.remove(save_file)
    process = subprocess.Popen(
        [
            "python",
            "scripts/openai_request_parallel.py",
            "--request_url",
            "https://api.openai.com/v1/chat/completions",
            "--api_key",
            os.environ["OPENAI_API_KEY"],
            "--requests_filepath",
            request_file,
            "--save_filepath",
            save_file,
            "--max_requests_per_minute",
            str(rate_limit),
            "--max_tokens_per_minute",
            str(token_limit),
        ]
    )
    process.wait()

    return save_file


def add_predictions(dataset_file: str, result_file: str, method: str, prediction_start_index: int) -> None:
    with open(dataset_file, "r") as f:
        d = json.load(f)
        print(f"Original data length: {len(d[method][prediction_start_index:])}")

    all_preds = []
    with open(result_file, "r") as f:
        for line in f:
            data = json.loads(line)
            pred = data[1]["choices"][0]["message"]["content"]
            pred = pred.split("```json")[1].replace("```", "").strip()
            try:
                pred = json.loads(pred)
            except json.decoder.JSONDecodeError:
                continue
            all_preds.extend(pred)
            # for x in pred:
            # if not x['label']:
            # print(x)
    print(f"Valid predictions: {len(all_preds)}")

    d[method][prediction_start_index:] = all_preds
    with open(dataset_file, "w") as f:
        json.dump(d, f, indent=2)


def test_evaluator(data_file: str, method: str) -> float:
    with open(data_file, "r") as f:
        d = json.load(f)[method]
    requests = []
    for e_id, example in enumerate(d):
        messages: list[str] = []
        if method == 'fuzzy_must_include':
            if any([x not in example for x in ["reference", "task", "prediction", "label"]]):
                continue
            ref  = example["reference"]
            task = example["task"]
            pred = example["prediction"]
            label = example["label"]
            for x in ref:
                message = get_fuzzy_must_include_message(pred, x, task)
                messages.append(message)
        elif method == 'fuzzy_exact_match':
            if any([x not in example for x in ["reference", "task", "prediction", "label"]]):
                continue
            ref  = example["reference"]
            task = example["task"]
            pred = example["prediction"]
            label = example["label"]
            message = get_fuzzy_exact_match_message(pred, ref, task)
            messages.append(message)
        elif method == 'fuzzy_na_match':
            if any([x not in example for x in ["reference", "task", "prediction", "label"]]):
                continue
            ref  = example["reference"]
            task = example["task"]
            pred = example["prediction"]
            label = example["label"]
            message = get_fuzzy_na_match(pred, ref, task)
            messages.append(message)
        elif method == 'context_qa':
            if any([x not in example for x in ["task", "question", "prediction", "label"]]):
                continue
            task = example["task"]
            question = example["question"]
            pred = example["prediction"]
            label = example["label"]
            message = get_context_aware_question_answering_message(question, label, pred, task)
            messages.append(message)
        elif method == 'qa':
            if any([x not in example for x in ["question", "prediction", "label"]]):
                continue
            question = example["question"]
            pred = example["prediction"]
            label = example["label"]
            message = get_question_answering_message(question, label, pred)
            messages.append(message)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        for message in messages:
            requests.append({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": message.strip()}],
                "temperature": 0.0,
                "max_tokens": 256,
                "top_p": 1.0,
                "metadata": {"e_id": e_id, "method": method, "label": label.lower() if isinstance(label, str) else label}
            })
        
    print(f"Total requests: {len(requests)}")
    request_file = data_file.replace(".json", f"_{method}_eval_requests.jsonl")
    with open(request_file, "w") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")
        
    save_file = data_file.replace(".json", f"_{method}_eval_results.jsonl")
    if os.path.exists(save_file):
        os.remove(save_file)
    process = subprocess.Popen(
        [
            "python",
            "scripts/openai_request_parallel.py",
            "--request_url",
            "https://api.openai.com/v1/chat/completions",
            "--api_key",
            os.environ["OPENAI_API_KEY"],
            "--requests_filepath",
            request_file,
            "--save_filepath",
            save_file,
            "--max_requests_per_minute",
            "15_000",
            "--max_tokens_per_minute",
            "2_000_000",
            "--logging_level",
            "40"
        ]
    )
    process.wait()


def parse_evaluator_result(data_file, save_file: str, method: str, print_errors: bool=True) -> None:
    # parse the result file
    all_preds = []
    with open(save_file, "r") as f:
        for line in f:
            data = json.loads(line)
            all_preds.append(data)
    

    e_id_to_pred = {x[2]["e_id"]: True for x in all_preds}
    e_id_to_label = {}
    for data in all_preds:
        metadata = data[2]
        if method in ["fuzzy_must_include", "fuzzy_exact_match", 'fuzzy_na_match']:
            e_id_to_label[metadata["e_id"]] = metadata["label"]
        elif method in ["context_qa", "qa"]:
            e_id_to_label[metadata["e_id"]] = True if metadata["label"] == "yes" else False
        else:
            raise ValueError(f"Unknown method: {method}")

    e_id_to_response = collections.defaultdict(list)
    tot = len(e_id_to_pred)       
    for data in all_preds:
        e_id = data[2]["e_id"]
        try:
            pred = data[1]["choices"][0]["message"]["content"].lower()
            e_id_to_response[e_id].append(data[0]["messages"][0]["content"] + "\n\n" + pred)
        except (KeyError, TypeError):
            e_id_to_pred[e_id] = "Error"
            continue
        if method in ["fuzzy_exact_match", "fuzzy_na_match"]:
            if "answer: correct" in pred:
                e_id_to_pred[e_id] = True
            else:
                e_id_to_pred[e_id] = False
        elif method == "fuzzy_must_include":
            if "answer: not contain" in pred:
                e_id_to_pred[e_id] = False
        elif method in ["context_qa", "qa"]:
            if "answer: yes" in pred:
                e_id_to_pred[e_id] = True
            else:
                e_id_to_pred[e_id] = False
        else:
            raise ValueError(f"Unknown method: {method}")

    error = 0
    preds = []
    labels = []
    e_ids = []
    for e_id in e_id_to_pred:
        if e_id_to_pred[e_id] == "Error":
            error += 1
            continue
        preds.append(e_id_to_pred[e_id])
        labels.append(e_id_to_label[e_id])
        e_ids.append(e_id)
    preds = np.array(preds)
    labels = np.array(labels)
    acc = np.sum(preds == labels) / len(labels)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

    if print_errors:
        # print wrong predictions
        with open(data_file, "r") as f:
            d = json.load(f)[method]
        # get the index where labels != preds
        error_indices = np.where(preds != labels)[0]
        for i in error_indices:
            print(d[e_ids[i]])

    print("====================")
    print(f"Error: {error}")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {p:.2f}")
    print(f"Recall: {r:.2f}")
    print(f"F1: {f1:.2f}")
    print("====================")


if __name__ == "__main__":
    overwrite = False
    print_errors = True
    # generate data
    params_map = {
        'fuzzy_must_include': {'example_num': 4, 'prediction_start_index': 4},
        'fuzzy_exact_match': {'example_num': 5, 'prediction_start_index': 5},
        'context_qa': {'example_num': 5, 'prediction_start_index': 5},
        'qa': {'example_num': 6, 'prediction_start_index': 24},
        'fuzzy_na_match': {'example_num': 5, 'prediction_start_index': 5},
    }
    dataset_file = "./tmp_data/fuzzy_match_dataset.json"
    for method in ["fuzzy_must_include", 'fuzzy_na_match', "fuzzy_exact_match", "context_qa", "qa"][2:3]:
        with open(dataset_file, "r") as f:
            d = json.load(f)[method]
        if overwrite or 'prediction' not in d[params_map[method]['prediction_start_index']]:
            save_file = generate_data(
                dataset_file,
                method,
                example_num=params_map[method]['example_num'],
                prediction_start_index=params_map[method]['prediction_start_index'],
                batch_size=4,
                rate_limit=15_000,
                token_limit=2_000_000,
            )
            calc_openai_cost(save_file)
            result_file = f"./tmp_data/fuzzy_match_dataset_{method}_results.jsonl"
            add_predictions(dataset_file, result_file, method, prediction_start_index=params_map[method]['prediction_start_index'])
        
        test_evaluator(dataset_file, method)
        parse_evaluator_result(dataset_file, dataset_file.replace(".json", f"_{method}_eval_results.jsonl"), method, print_errors=print_errors)
        calc_openai_cost(dataset_file.replace(".json", f"_{method}_eval_results.jsonl"))
