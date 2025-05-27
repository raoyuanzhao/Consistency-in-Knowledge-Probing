import json
import argparse
import lm_utils
import metrics
import random
from tqdm import tqdm
import pickle
import os

prefix = ["""""",
"""Question: Who sings 'Here Comes the Sun'?
A: Led Zeppelin
B: Queen
C: Pink Floyd
D: The Beatles
Choose one answer from the above choices. The answer is D\nThe above answer is:\nA. True\nB. False\nThe answer isA
""",
"""Question: What is 2+2?
A: 3
B: 4
C: 5
D: 6
Choose one answer from the above choices. The answer is B\nThe above answer is:\nA. True\nB. False\nThe answer isA
""",
"""Question: What is the capital of France?
A: Berlin
B: Madrid
C: Paris
D: Rome
Choose one answer from the above choices. The answer is C\nThe above answer is:\nA. True\nB. False\nThe answer isA
""",
"""Question: What is the chemical symbol for water?
A: H2O
B: CO2
C: NaCl
D: O2
Choose one answer from the above choices. The answer is A\nThe above answer is:\nA. True\nB. False\nThe answer isA
""",
"""Question: When the lights went out during the storm, they
A: started watching a movie.
B: lit some candles.
C: opened the refrigerator.
D: went swimming in the river.
Choose one answer from the above choices. The answer is B
""",
"""Question: After the baby started crying, the mother
A: picked up the baby to comfort it.
B: paint the ceiling with a toothbrush.
C: whispered to the toaster.
D: opened an umbrella indoors.
Choose one answer from the above choices. The answer is A
""",
"""Question: As the sun set over the horizon, the sky turned
A: white.
B: completely green.
C: a mix of orange and pink.
D: into a checkerboard pattern.
Choose one answer from the above choices. The answer is C
""",
"""Question: When the doorbell rang, I went to the door and
A: closed the windows.
B: started cooking dinner.
C: went to bed.
D: opened it to see who was there.
Choose one answer from the above choices. The answer is D
"""
            ]

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use: \"mistral\", \"llama2_7/13/70b\", \"chatgpt\"")
    argParser.add_argument("-d", "--dataset", help="which dataset in data/: \"mmlu\", \"knowledge_crosswords\", \"hellaswag\", \"propaganda\", \"ambigqa\", \"electionqa23\"")
    argParser.add_argument("-o", "--portion", default = 1.0, help="portion of the dataset to use")
    argParser.add_argument("-i", "--index", default = 0, help="index of prompts")

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    portion = args.portion
    prompt_index = int(args.index)

    lm_utils.llm_init(model_name)

    correct_flags = []
    abstain_flags = []
    abstain_scores = []
    test_prediction = []
    gold_answer = []

    with open("data/" + dataset + ".json", "r") as f:
        data = json.load(f)

        data["dev"] = data["dev"][:int(len(data["dev"])*float(portion))]
        data["test"] = data["test"][:int(len(data["test"])*float(portion))]

        for d in tqdm(data["test"]):
            original_prompt = prefix[prompt_index] + "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            response = lm_utils.llm_response(original_prompt, model_name, probs=False)
            # print(response)
            # print(lm_utils.answer_parsing(response))
            test_prediction.append(lm_utils.answer_parsing(response))
            gold_answer.append(d["answer"])
            if lm_utils.answer_parsing(response) == d["answer"]:
                correct_flags.append(1)
            else:
                correct_flags.append(0)
        
            response += "\nThe above answer is:\nA. True\nB. False\nThe answer is"
            response, probs = lm_utils.llm_response(response, model_name, probs=True)
            # print(response)
            if lm_utils.answer_parsing(response) == "A":
                abstain_flags.append(0)
            else:
                abstain_flags.append(1)

            option_to_ind = {"a": 0, "b": 1, "true": 0, "false": 1}

            try:
                for token in probs.keys():
                    if token.strip().lower() in option_to_ind.keys():
                        if option_to_ind[token.strip().lower()] == 0:
                            abstain_scores.append(1 - probs[token])
                            break
                        else:
                            abstain_scores.append(probs[token])
                            break
            except:
                print("option probs failed, uniform assignment")
                chosen_option = random.choice(["A", "B"])
                abstain_scores.append(0.5)

    print("------------------")
    print("Approach: reflect")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
    print("------------------")
    results = []
    for i in range(len(abstain_scores)):
        result = {
            "question_idx": i,
            "decision": abstain_flags[i],
            "prediction": test_prediction[i],
            "gold_answer": gold_answer[i]
        }
        results.append(result)
    folder_path = f'Probing_Uncertainy/result/{model_name}_{dataset}'

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    output_file = f'Probing_Uncertainy/result/{model_name}_{dataset}/reflect_{model_name}_{dataset}_setup_{prompt_index}.json'

    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)

    with open(f'Probing_Uncertainy/pickle_result/reflect_{model_name}_{dataset}_setup_{prompt_index}_result.pkl', 'wb') as f:
        pickle.dump(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores), f)