import builtins
import os

import litellm
import numpy as np


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


# Split word into characters
def split(word):
    return [char for char in word]


# Load all problems
all_prob = np.load("./all_problems_1thru5.npz", allow_pickle=True)

# Loop through all problem types
all_prob_types = builtins.list(all_prob["all_problems"].item().keys())
# Load data if it already exists
data_exists = False
# Create data structure for storing results
all_gen_pred = {}
all_gen_correct_pred = {}
for p in range(len(all_prob_types)):
    # Problem type
    prob_type = all_prob_types[p]
    # Load data
    if data_exists:
        all_gen_pred[prob_type] = all_data["all_gen_pred"].item()[prob_type]
        all_gen_correct_pred[prob_type] = all_data["all_gen_correct_pred"].item()[prob_type]
    # Create data structure
    else:
        all_gen_pred[prob_type] = []
        all_gen_correct_pred[prob_type] = []
# Loop over all problem indices
N_runs = 20
result_list = []
for run in range(N_runs):
    print(str(run + 1) + " of " + str(N_runs) + "...")
    # Initialize context with task instructions
    context = "[1] [1] [1]\n[2] [2] [2]\n[3] [3] [3]\n\n"
    # Loop over all problem types

    counter = 0
    for p in range(len(all_prob_types)):
        # Problem type
        prob_type = all_prob_types[p]
        count = all_prob["all_problems"].item()[prob_type]["prob"].shape[0]
        print(f"Problem type: {prob_type}... {count}/{len(all_prob_types)}")
        counter += count
    print(f"Total problems: {counter}")

    for p in range(len(all_prob_types)):
        # Problem type
        prob_type = all_prob_types[p]
        print("Problem type: " + prob_type + "...")
        perm_invariant = all_prob["all_problems"].item()[prob_type]["perm_invariant"]
        prob_type_N_prob = all_prob["all_problems"].item()[prob_type]["prob"].shape[0]
        if len(all_gen_correct_pred[prob_type]) <= run:
            # Sample problem index
            prob_ind = int(np.floor(np.random.rand() * prob_type_N_prob))

            # Problem
            prob = all_prob["all_problems"].item()[prob_type]["prob"][prob_ind]
            answer_choices = all_prob["all_problems"].item()[prob_type]["answer_choices"][prob_ind]
            correct_ind = all_prob["all_problems"].item()[prob_type]["correct_ind"][prob_ind]
            correct_answer = answer_choices[correct_ind]

            # Generate prompt
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
            # Add context
            context_prompt = context + prompt

            print("Prompt: " + context_prompt)

            # Get response
            fits_window = False
            response = []
            while not fits_window:
                try:
                    response = litellm.completion(
                        messages=[{"role": "user", "content": context_prompt}],
                        model="openai/gpt-4",
                        temperature=0,
                        max_tokens=10,
                        stop="\n",
                    )
                except:
                    print("deleting problem from context...")
                    context_prompt = context_prompt.split("\n\n")[1:]
                    new_context_prompt = ""
                    for i in range(len(context_prompt)):
                        new_context_prompt += context_prompt[i]
                        if i < (len(context_prompt) - 1):
                            new_context_prompt += "\n\n"
                    context_prompt = new_context_prompt
                if response.choices:
                    fits_window = True
            response_text = response.choices[0].message.content
            # Find portion of response corresponding to prediction
            prediction = response_text
            all_gen_pred[prob_type].append(prediction)
            # Get prediction set
            pred_set = []
            invalid_char = False
            closing_bracket = False
            for i in range(len(split(prediction))):
                if prediction[i] != " ":
                    if prediction[i].isdigit():
                        pred_set.append(int(prediction[i]))
                    elif prediction[i] == "[":
                        continue
                    elif prediction[i] == "]":
                        closing_bracket = True
                        break
                    else:
                        invalid_char = True
                        break
            # Sort answer if problem type is permutation invariant
            if perm_invariant:
                correct_answer = np.sort(correct_answer)
                pred_set = np.sort(pred_set)
            # Determine whether prediction is correct
            correct_pred = False
            if not invalid_char and len(pred_set) == len(correct_answer):
                if np.all(pred_set == correct_answer):
                    correct_pred = True
            all_gen_correct_pred[prob_type].append(correct_pred)
            print(
                f"correct_pred: {correct_pred}, pred_set: {pred_set}, correct_answer: {correct_answer}, response_text: {response_text}"
            )

            result_list.append(
                {
                    "prob_type": prob_type,
                    "prob_ind": prob_ind,
                    "prediction": pred_set,
                    "correct_answer": correct_answer,
                    "correct": correct_pred,
                }
            )


from datetime import datetime

import pandas as pd

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
results_directory = os.path.join("results", "base", timestamp)
os.makedirs(results_directory, exist_ok=True)
df = pd.DataFrame(result_list)
df.to_csv(os.path.join(results_directory, "results.csv"), index=False)
acc_by_type = df.groupby("prob_type")["correct"].mean()
acc_by_type.to_csv(os.path.join(results_directory, "acc_by_type.csv"))

df_two = df[df["prob_type"].str.contains("two", case=False)]
df_three = df[df["prob_type"].str.contains("three", case=False)]
df_four = df[df["prob_type"].str.contains("four", case=False)]
df_five = df[df["prob_type"].str.contains("five", case=False)]
df_else = df[~df["prob_type"].str.contains("two|three|four|five", case=False)]

acc_two = df_two["correct"].mean() if not df_two.empty else np.nan
acc_three = df_three["correct"].mean() if not df_three.empty else np.nan
acc_four = df_four["correct"].mean() if not df_four.empty else np.nan
acc_five = df_five["correct"].mean() if not df_five.empty else np.nan
acc_else = df_else["correct"].mean() if not df_else.empty else np.nan

sub_cat_acc_dict = {
    "one": float(acc_else),
    "two": float(acc_two),
    "three": float(acc_three),
    "four": float(acc_four),
    "five": float(acc_five),
}
sub_cat_acc_df = pd.DataFrame.from_dict(sub_cat_acc_dict, orient="index", columns=["accuracy"])
sub_cat_acc_df.to_csv(os.path.join(results_directory, "sub_cat_acc.csv"))