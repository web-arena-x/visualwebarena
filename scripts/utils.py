import os
import json
def calc_openai_cost(response_file: str) -> float:
    in_tok_tot = 0
    out_tok_tot = 0

    errors = 0
    if not os.path.exists(response_file):
        print(f"{response_file} does not exist")
        return 0.0

    with open(response_file, "r") as f:
        model_checked = False
        for l_idx, line in enumerate(f):
            cur_request = json.loads(line)
            if not model_checked:
                model = cur_request[0]["model"]
                match model:
                    case "gpt-3.5-turbo" | "gpt-3.5-turbo-1106":
                        in_cost = 0.001
                        out_cost = 0.002
                    case (
                        "gpt-4-1106-preview" | "vijay-gpt-4" | "gpt-4-turbo-2024-04-09"
                    ):
                        in_cost = 0.01
                        out_cost = 0.03
                    case "gpt-4":
                        in_cost = 0.03
                        out_cost = 0.06
                    case "gpt-4o":
                        in_cost = 0.005
                        out_cost = 0.015
                    case _:
                        raise ValueError(f"Unknown model: {model}")
                model_checked = True
            try:
                in_tok_tot += cur_request[1]["usage"]["prompt_tokens"]
                out_tok_tot += cur_request[1]["usage"]["completion_tokens"]
            except TypeError:
                errors += 1

    # calc the cost
    cost = in_tok_tot / 1000 * in_cost + out_tok_tot / 1000 * out_cost
    print(f"Input cost: {in_cost}, Output cost: {out_cost}")
    print(
        f"Input cost: {in_tok_tot / 1000 * in_cost}, Output cost: {out_tok_tot / 1000 * out_cost}"
    )
    print(f"Errors: {errors}")
    return cost