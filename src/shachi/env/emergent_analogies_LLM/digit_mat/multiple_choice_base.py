import builtins
import os
import re

import litellm
import numpy as np


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def split(word):
    return [char for char in word]


all_prob = np.load("./all_problems_1thru5.npz", allow_pickle=True)

kwargs = {
    "engine": "text-davinci-003",
    "temperature": 0,
    "max_tokens": 10,
    "stop": "\n",
    "echo": True,
    "logprobs": 1,
}

all_prob_types = builtins.list(all_prob["all_problems"].item().keys())
all_data_fname = "./gpt_matprob_results_1thru5.npz"
if os.path.exists(all_data_fname):
    data_exists = True
    all_data = np.load("./gpt_matprob_results_1thru5.npz", allow_pickle=True)
else:
    data_exists = False
all_MC_pred = {}
all_MC_correct_pred = {}
for p in range(len(all_prob_types)):
    prob_type = all_prob_types[p]
    if data_exists:
        all_MC_pred[prob_type] = all_data["all_MC_pred"].item()[prob_type]
        all_MC_correct_pred[prob_type] = all_data["all_MC_correct_pred"].item()[prob_type]
    else:
        all_MC_pred[prob_type] = []
        all_MC_correct_pred[prob_type] = []

N_runs = 20
for run in range(N_runs):
    print(str(run + 1) + " of " + str(N_runs) + "...")
    context = "[1] [1] [1]\n[2] [2] [2]\n[3] [3] [3]\n\n"
    for p in range(len(all_prob_types)):
        prob_type = all_prob_types[p]
        print("Problem type: " + prob_type + "...")
        perm_invariant = all_prob["all_problems"].item()[prob_type]["perm_invariant"]
        prob_type_N_prob = all_prob["all_problems"].item()[prob_type]["prob"].shape[0]
        if len(all_MC_correct_pred[prob_type]) <= run:
            prob_ind = int(np.floor(np.random.rand() * prob_type_N_prob))

            prob = all_prob["all_problems"].item()[prob_type]["prob"][prob_ind]
            answer_choices = all_prob["all_problems"].item()[prob_type]["answer_choices"][prob_ind]
            correct_ind = all_prob["all_problems"].item()[prob_type]["correct_ind"][prob_ind]
            correct_answer = answer_choices[correct_ind]

            prompt = ""
            for r in range(3):
                for c in range(3):
                    prompt += "["
                    if not (r == 2 and c == 2):
                        for i in range(len(prob[r][c])):
                            if prob[r][c][i] == -1:
                                prompt += " "
                            else:
                                prompt += str(prob[r][c][i])
                            if i < len(prob[r][c]) - 1:
                                prompt += " "
                        prompt += "]"
                        if c < 2:
                            prompt += " "
                        else:
                            prompt += "\n"

            context_prompt = context + prompt

            prompt += "\nHere are the 8 possible answer choices:\n"
            for i, ans in enumerate(answer_choices):
                choice_str = np.array(split(str(ans)))
                choice_str = "".join(choice_str[choice_str != ","])
                prompt += f"{i}: {choice_str}\n"
            prompt += "\nPlease answer with the single number (0-7) you think is correct.\n"

            context_prompt = context + prompt

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI that only outputs the single number (0-7) for the correct choice.",
                },
                {"role": "user", "content": context_prompt},
            ]
            plan_completion = litellm.completion(
                messages=messages,
                model="openai/gpt-3.5-turbo",
                temperature=0,
            )

            model_response = plan_completion.choices[0].message.content
            match = re.search(r"\b([0-7])\b", model_response)
            if match:
                model_choice = int(match.group(1))
            else:
                model_choice = 0
            print("-" * 80)
            print(
                f"Model choice: {model_choice}, correct_ind: {correct_ind}, choice_str: {choice_str}, model_response: {model_response}, answer_choices: {answer_choices}"
            )
            print(f"prompt: {prompt}")
            all_MC_pred[prob_type].append(model_choice)
            MC_correct = model_choice == correct_ind
            all_MC_correct_pred[prob_type].append(MC_correct)

            eval_fname = "./gpt_matprob_results_1thru5.npz"
            np.savez(
                eval_fname,
                all_MC_pred=all_MC_pred,
                all_MC_correct_pred=all_MC_correct_pred,
                allow_pickle=True,
            )
            # Raw output
            gen_data_dir = "./gpt_matprob_results_1thru5/"
            check_path(gen_data_dir)
            gen_data_fname = gen_data_dir + str(run) + ".txt"
            gen_data_fid = open(gen_data_fname, "w")
            gen_data_fid.write(context)
            gen_data_fid.close()

        else:
            gen_data_dir = "./gpt_matprob_results_1thru5/"
            gen_data_fname = gen_data_dir + str(run) + ".txt"
            gen_data_fid = open(gen_data_fname, "r")
            lines = gen_data_fid.readlines()
            context = " ".join(lines)
            # Remove spaces
            context = context.split("\n")
            new_context = context[0]
            for c in range(1, len(context)):
                new_context += "\n"
                new_context += context[c][1:]
            context = new_context
