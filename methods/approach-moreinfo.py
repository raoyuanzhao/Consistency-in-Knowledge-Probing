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
Choose one answer from the above choices. The answer is D
Do you need more information to answer this question? (Yes or No)No
""",
"""Question: What is 2+2?
A: 3
B: 4
C: 5
D: 6
Choose one answer from the above choices. The answer is B
Do you need more information to answer this question? (Yes or No)No
""",
"""Question: What is the capital of France?
A: Berlin
B: Madrid
C: Paris
D: Rome
Choose one answer from the above choices. The answer is C
Do you need more information to answer this question? (Yes or No)No
""",
"""Question: What is the chemical symbol for water?
A: H2O
B: CO2
C: NaCl
D: O2
Choose one answer from the above choices. The answer is A
Do you need more information to answer this question? (Yes or No)No
""",
"""Question: When the lights went out during the storm, they
A: started watching a movie.
B: lit some candles.
C: opened the refrigerator.
D: went swimming in the river.
Choose one answer from the above choices. The answer is B
Do you need more information to answer this question? (Yes or No)No
""",
"""Question: After the baby started crying, the mother
A: picked up the baby to comfort it.
B: paint the ceiling with a toothbrush.
C: whispered to the toaster.
D: opened an umbrella indoors.
Choose one answer from the above choices. The answer is A
Do you need more information to answer this question? (Yes or No)No
""",
"""Question: As the sun set over the horizon, the sky turned
A: white.
B: completely green.
C: a mix of orange and pink.
D: into a checkerboard pattern.
Choose one answer from the above choices. The answer is C
Do you need more information to answer this question? (Yes or No)No
""",
"""Question: When the doorbell rang, I went to the door and
A: closed the windows.
B: started cooking dinner.
C: went to bed.
D: opened it to see who was there.
Choose one answer from the above choices. The answer is D
Do you need more information to answer this question? (Yes or No)No
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

        # obtain correct_flags
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

        # obtain abstain_flags
        for d in tqdm(data["test"]):
            original_prompt = prefix[prompt_index] + "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Do you need more information to answer this question? (Yes or No)"
            response, token_probs = lm_utils.llm_response(original_prompt, model_name, probs=True)
            # print(response)
            if "yes" in response.lower():
                abstain_flags.append(1)
            else:
                abstain_flags.append(0)
            try:
                for token in token_probs.keys():
                    if token.strip().lower() == "yes":
                        abstain_scores.append(token_probs[token])
                        break
                    elif token.strip().lower() == "no":
                        abstain_scores.append(1-token_probs[token])
                        break
                    else:
                        abstain_scores.append(0.5)
                        break
                    
            
            except:
                print("yes/no probs failed, uniform assignment")
                abstain_scores.append(0.5)

    print("------------------")
    print("Approach: moreinfo")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print("debug",len(correct_flags),len(abstain_scores))
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
        # Define folder path
    folder_path = f'Probing_Uncertainy/result/{model_name}_{dataset}'

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Define the full file path
    #file_path = os.path.join(folder_path, f'prompt_{prompt_index}_result.pkl')

    output_file = f'Probing_Uncertainy/result/{model_name}_{dataset}/moreinfo_{model_name}_{dataset}_setup_{prompt_index}.json'

    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)

    with open(f'Probing_Uncertainy/pickle_result/moreinfo_{model_name}_{dataset}_setup_{prompt_index}_result.pkl', 'wb') as f:
        pickle.dump(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores), f)