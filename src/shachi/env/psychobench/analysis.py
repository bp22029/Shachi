import copy
import json
from typing import Dict, List, Tuple

import requests  # type: ignore[import-untyped]
import scipy.stats as stats
from pydantic import BaseModel

from .data_types import PsychoBenchTestName, Result


def hypothesis_testing(
    result1: Result, result2: Result, significance_level: float, model: str, crowd_name: str
) -> Tuple[str, str]:
    output_list = ""
    output_text = f"### Compare with {crowd_name}\n"

    # Extract the mean, std and size for both data sets
    mean1, std1, n1 = result1
    mean2, std2, n2 = result2
    output_list += f"{mean2:.1f} $\\pm$ {std2:.1f}"

    # Add an epsilon to prevent the zero standard deviarion
    epsilon = 1e-8
    std1 += epsilon
    std2 += epsilon

    output_text += "\n- **Statistic**:\n"
    output_text += f"{model}:\tmean1 = {mean1:.1f},\tstd1 = {std1:.1f},\tn1 = {n1}\n"
    output_text += f"{crowd_name}:\tmean2 = {mean2:.1f},\tstd2 = {std2:.1f},\tn2 = {n2}\n"

    # Perform F-test
    output_text += "\n- **F-Test:**\n\n"

    if std1 > std2:
        f_value = std1**2 / std2**2
        df1, df2 = n1 - 1, n2 - 1
    else:
        f_value = std2**2 / std1**2
        df1, df2 = n2 - 1, n1 - 1

    p_value = (1 - stats.f.cdf(f_value, df1, df2)) * 2
    equal_var = True if p_value > significance_level else False

    output_text += f"\tf-value = {f_value:.4f}\t($df_1$ = {df1}, $df_2$ = {df2})\n\n"
    output_text += f"\tp-value = {p_value:.4f}\t(two-tailed test)\n\n"
    output_text += "\tNull hypothesis $H_0$ ($s_1^2$ = $s_2^2$): "

    if p_value > significance_level:
        output_text += f"\tSince p-value ({p_value:.4f}) > α ({significance_level}), $H_0$ cannot be rejected.\n\n"
        output_text += f"\t**Conclusion ($s_1^2$ = $s_2^2$):** The variance of average scores responsed by {model} is statistically equal to that responsed by {crowd_name} in this category.\n\n"
    else:
        output_text += (
            f"\tSince p-value ({p_value:.4f}) < α ({significance_level}), $H_0$ is rejected.\n\n"
        )
        output_text += f"\t**Conclusion ($s_1^2$ ≠ $s_2^2$):** The variance of average scores responsed by {model} is statistically unequal to that responsed by {crowd_name} in this category.\n\n"

    # Performing T-test
    output_text += (
        "- **Two Sample T-Test (Equal Variance):**\n\n"
        if equal_var
        else "- **Two Sample T-test (Welch's T-Test):**\n\n"
    )

    df = (
        n1 + n2 - 2
        if equal_var
        else ((std1**2 / n1 + std2**2 / n2) ** 2)
        / ((std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1))
    )
    t_value, p_value = stats.ttest_ind_from_stats(
        mean1, std1, n1, mean2, std2, n2, equal_var=equal_var
    )

    output_text += f"\tt-value = {t_value:.4f}\t($df$ = {df:.1f})\n\n"
    output_text += f"\tp-value = {p_value:.4f}\t(two-tailed test)\n\n"

    output_text += "\tNull hypothesis $H_0$ ($µ_1$ = $µ_2$): "
    if p_value > significance_level:
        output_text += f"\tSince p-value ({p_value:.4f}) > α ({significance_level}), $H_0$ cannot be rejected.\n\n"
        output_text += f"\t**Conclusion ($µ_1$ = $µ_2$):** The average scores of {model} is assumed to be equal to the average scores of {crowd_name} in this category.\n\n"
        # output_list += f' ( $-$ )'

    else:
        output_text += (
            f"Since p-value ({p_value:.4f}) < α ({significance_level}), $H_0$ is rejected.\n\n"
        )
        if t_value > 0:
            output_text += "\tAlternative hypothesis $H_1$ ($µ_1$ > $µ_2$): "
            output_text += f"\tSince p-value ({(1 - p_value / 2):.1f}) > α ({significance_level}), $H_1$ cannot be rejected.\n\n"
            output_text += f"\t**Conclusion ($µ_1$ > $µ_2$):** The average scores of {model} is assumed to be larger than the average scores of {crowd_name} in this category.\n\n"
        else:
            output_text += "\tAlternative hypothesis $H_1$ ($µ_1$ < $µ_2$): "
            output_text += f"\tSince p-value ({(1 - p_value / 2):.1f}) > α ({significance_level}), $H_1$ cannot be rejected.\n\n"
            output_text += f"\t**Conclusion ($µ_1$ < $µ_2$):** The average scores of {model} is assumed to be smaller than the average scores of {crowd_name} in this category.\n\n"

    output_list += " | "
    return (output_text, output_list)


payload_template = {
    "questions": [
        {"text": "You regularly make new friends.", "answer": None},
        {
            "text": "You spend a lot of your free time exploring various random topics that pique your interest.",
            "answer": None,
        },
        {
            "text": "Seeing other people cry can easily make you feel like you want to cry too.",
            "answer": None,
        },
        {"text": "You often make a backup plan for a backup plan.", "answer": None},
        {"text": "You usually stay calm, even under a lot of pressure.", "answer": None},
        {
            "text": "At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know.",
            "answer": None,
        },
        {
            "text": "You prefer to completely finish one project before starting another.",
            "answer": None,
        },
        {"text": "You are very sentimental.", "answer": None},
        {"text": "You like to use organizing tools like schedules and lists.", "answer": None},
        {
            "text": "Even a small mistake can cause you to doubt your overall abilities and knowledge.",
            "answer": None,
        },
        {
            "text": "You feel comfortable just walking up to someone you find interesting and striking up a conversation.",
            "answer": None,
        },
        {
            "text": "You are not too interested in discussing various interpretations and analyses of creative works.",
            "answer": None,
        },
        {"text": "You are more inclined to follow your head than your heart.", "answer": None},
        {
            "text": "You usually prefer just doing what you feel like at any given moment instead of planning a particular daily routine.",
            "answer": None,
        },
        {
            "text": "You rarely worry about whether you make a good impression on people you meet.",
            "answer": None,
        },
        {"text": "You enjoy participating in group activities.", "answer": None},
        {
            "text": "You like books and movies that make you come up with your own interpretation of the ending.",
            "answer": None,
        },
        {
            "text": "Your happiness comes more from helping others accomplish things than your own accomplishments.",
            "answer": None,
        },
        {
            "text": "You are interested in so many things that you find it difficult to choose what to try next.",
            "answer": None,
        },
        {
            "text": "You are prone to worrying that things will take a turn for the worse.",
            "answer": None,
        },
        {"text": "You avoid leadership roles in group settings.", "answer": None},
        {"text": "You are definitely not an artistic type of person.", "answer": None},
        {
            "text": "You think the world would be a better place if people relied more on rationality and less on their feelings.",
            "answer": None,
        },
        {
            "text": "You prefer to do your chores before allowing yourself to relax.",
            "answer": None,
        },
        {"text": "You enjoy watching people argue.", "answer": None},
        {"text": "You tend to avoid drawing attention to yourself.", "answer": None},
        {"text": "Your mood can change very quickly.", "answer": None},
        {"text": "You lose patience with people who are not as efficient as you.", "answer": None},
        {"text": "You often end up doing things at the last possible moment.", "answer": None},
        {
            "text": "You have always been fascinated by the question of what, if anything, happens after death.",
            "answer": None,
        },
        {
            "text": "You usually prefer to be around others rather than on your own.",
            "answer": None,
        },
        {
            "text": "You become bored or lose interest when the discussion gets highly theoretical.",
            "answer": None,
        },
        {
            "text": "You find it easy to empathize with a person whose experiences are very different from yours.",
            "answer": None,
        },
        {
            "text": "You usually postpone finalizing decisions for as long as possible.",
            "answer": None,
        },
        {"text": "You rarely second-guess the choices that you have made.", "answer": None},
        {
            "text": "After a long and exhausting week, a lively social event is just what you need.",
            "answer": None,
        },
        {"text": "You enjoy going to art museums.", "answer": None},
        {
            "text": "You often have a hard time understanding other people’s feelings.",
            "answer": None,
        },
        {"text": "You like to have a to-do list for each day.", "answer": None},
        {"text": "You rarely feel insecure.", "answer": None},
        {"text": "You avoid making phone calls.", "answer": None},
        {
            "text": "You often spend a lot of time trying to understand views that are very different from your own.",
            "answer": None,
        },
        {
            "text": "In your social circle, you are often the one who contacts your friends and initiates activities.",
            "answer": None,
        },
        {
            "text": "If your plans are interrupted, your top priority is to get back on track as soon as possible.",
            "answer": None,
        },
        {
            "text": "You are still bothered by mistakes that you made a long time ago.",
            "answer": None,
        },
        {
            "text": "You rarely contemplate the reasons for human existence or the meaning of life.",
            "answer": None,
        },
        {"text": "Your emotions control you more than you control them.", "answer": None},
        {
            "text": "You take great care not to make people look bad, even when it is completely their fault.",
            "answer": None,
        },
        {
            "text": "Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.",
            "answer": None,
        },
        {
            "text": "When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.",
            "answer": None,
        },
        {
            "text": "You would love a job that requires you to work alone most of the time.",
            "answer": None,
        },
        {
            "text": "You believe that pondering abstract philosophical questions is a waste of time.",
            "answer": None,
        },
        {
            "text": "You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places.",
            "answer": None,
        },
        {"text": "You know at first glance how someone is feeling.", "answer": None},
        {"text": "You often feel overwhelmed.", "answer": None},
        {
            "text": "You complete things methodically without skipping over any steps.",
            "answer": None,
        },
        {"text": "You are very intrigued by things labeled as controversial.", "answer": None},
        {
            "text": "You would pass along a good opportunity if you thought someone else needed it more.",
            "answer": None,
        },
        {"text": "You struggle with deadlines.", "answer": None},
        {"text": "You feel confident that things will work out for you.", "answer": None},
    ],
    "gender": None,
    "inviteCode": "",
    "teamInviteKey": "",
    "extraData": [],
}
role_mapping = {
    "ISTJ": "Logistician",
    "ISTP": "Virtuoso",
    "ISFJ": "Defender",
    "ISFP": "Adventurer",
    "INFJ": "Advocate",
    "INFP": "Mediator",
    "INTJ": "Architect",
    "INTP": "Logician",
    "ESTP": "Entrepreneur",
    "ESTJ": "Executive",
    "ESFP": "Entertainer",
    "ESFJ": "Consul",
    "ENFP": "Campaigner",
    "ENFJ": "Protagonist",
    "ENTP": "Debater",
    "ENTJ": "Commander",
}


def parsing(score_list: List[int]) -> Tuple[str, str]:
    code = ""

    if score_list[0] >= 50:
        code = code + "E"
    else:
        code = code + "I"

    if score_list[1] >= 50:
        code = code + "N"
    else:
        code = code + "S"

    if score_list[2] >= 50:
        code = code + "T"
    else:
        code = code + "F"

    if score_list[3] >= 50:
        code = code + "J"
    else:
        code = code + "P"

    if score_list[4] >= 50:
        code = code + "-A"
    else:
        code = code + "-T"

    return code, role_mapping[code[:4]]


def query_16personalities_api(scores: List[int]) -> Tuple[str, str, List[int]]:
    payload = copy.deepcopy(payload_template)

    for index, score in enumerate(scores):
        payload["questions"][index]["answer"] = score

    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en,zh-CN;q=0.9,zh;q=0.8",
        "content-length": "5708",
        "content-type": "application/json",
        "origin": "https://www.16personalities.com",
        "referer": "https://www.16personalities.com/free-personality-test",
        "sec-ch-ua": "'Not_A Brand';v='99', 'Google Chrome';v='109', 'Chromium';v='109'",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "Windows",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36",
    }

    session = requests.session()
    _r = session.post(
        "https://www.16personalities.com/test-results", data=json.dumps(payload), headers=headers
    )

    sess_r = session.get("https://www.16personalities.com/api/session")
    scores = sess_r.json()["user"]["scores"]

    if sess_r.json()["user"]["traits"]["energy"] != "Extraverted":
        energy_value = 100 - (101 + scores[0]) // 2
    else:
        energy_value = (101 + scores[0]) // 2
    if sess_r.json()["user"]["traits"]["mind"] != "Intuitive":
        mind_value = 100 - (101 + scores[1]) // 2
    else:
        mind_value = (101 + scores[1]) // 2
    if sess_r.json()["user"]["traits"]["nature"] != "Thinking":
        nature_value = 100 - (101 + scores[2]) // 2
    else:
        nature_value = (101 + scores[2]) // 2
    if sess_r.json()["user"]["traits"]["tactics"] != "Judging":
        tactics_value = 100 - (101 + scores[3]) // 2
    else:
        tactics_value = (101 + scores[3]) // 2
    if sess_r.json()["user"]["traits"]["identity"] != "Assertive":
        identity_value = 100 - (101 + scores[4]) // 2
    else:
        identity_value = (101 + scores[4]) // 2

    code, role = parsing([energy_value, mind_value, nature_value, tactics_value, identity_value])

    return code, role, [energy_value, mind_value, nature_value, tactics_value, identity_value]


class SixteenPTestResult(BaseModel):
    personality_type: str
    role: str
    extraverted: float
    intuitive: float
    thinking: float
    judging: float
    assertive: float


class ResultFor16P(BaseModel):
    questionnaire_name: PsychoBenchTestName

    results: List[SixteenPTestResult]
    aggregated_result: SixteenPTestResult


def analysis_personality(test_data: List[Dict]) -> ResultFor16P:
    all_data = []
    cat = [
        "personality_type",
        "role",
        "extraverted",
        "intuitive",
        "thinking",
        "judging",
        "assertive",
    ]

    test_results = []
    for case in test_data:
        ordered_list = [case[key] - 4 for key in sorted(case.keys())]
        all_data.append(ordered_list)
        result = query_16personalities_api(ordered_list)
        result = result[:2] + tuple(result[2])

        kwargs = dict()
        for c, r in zip(cat, result):
            kwargs[c] = r

        test_results.append(SixteenPTestResult(**kwargs))

    column_sums = [sum(col) for col in zip(*all_data)]
    avg_data = [int(sum / len(all_data)) for sum in column_sums]
    avg_result = query_16personalities_api(avg_data)
    avg_result = avg_result[:2] + tuple(avg_result[2])
    kwargs = dict()
    for c, r in zip(cat, avg_result):
        kwargs[c] = r
    avg_result = SixteenPTestResult(**kwargs)

    return ResultFor16P(
        questionnaire_name="16P", results=test_results, aggregated_result=avg_result
    )
