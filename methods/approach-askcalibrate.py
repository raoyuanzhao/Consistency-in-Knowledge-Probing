import json
import argparse
import lm_utils
import metrics
import random
import re
from tqdm import tqdm
from datetime import datetime
import os
import pickle

prefix = ["""""",
"""Question: Who sings 'Here Comes the Sun'?
A: Led Zeppelin
B: Queen
C: Pink Floyd
D: The Beatles
Choose one answer from the above choices. The answer is D
""",
"""Question: What is 2+2?
A: 3
B: 4
C: 5
D: 6
Choose one answer from the above choices. The answer is B
""",
"""Question: What is the capital of France?
A: Berlin
B: Madrid
C: Paris
D: Rome
Choose one answer from the above choices. The answer is C
""",
"""Question: What is the chemical symbol for water?
A: H2O
B: CO2
C: NaCl
D: O2
Choose one answer from the above choices. The answer is A
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
prompts_2 = [
"""""",
"""The question is:
Who sings 'Here Comes the Sun'?
choices: A: Led Zeppelin, B: Queen, C: Pink Floyd, D: The Beatles
Choose one answer from the above choices. Guess: D
""",
"""The question is:
What is 2+2?
Choices: A: 3, B: 4, C: 5, D: 6
Choose one answer from the above choices. Guess: B
""",
"""The question is:
What is the capital of France?
Choices: A: Berlin, B: Madrid, C: Paris, D: Rome
Choose one answer from the above choices. Guess: C
""",
"""The question is:
What is the chemical symbol for water?
Choices: A: H2O, B: CO2, C: NaCl, D: O2
Choose one answer from the above choices. Guess: A
""",
"""The question is: What did they do when the lights went out during the storm?
choices: A: started watching a movie, B: lit some candles, C: opened the refrigerator, D: went swimming in the river
Choose one answer from the above choices. Guess: B
""",
"""The question is: What did the mother do after the baby started crying?
choices: A: picked up the baby to comfort it, B: paint the ceiling with a toothbrush, C: whispered to the toaster, D: opened an umbrella indoors
Choose one answer from the above choices. Guess: A
""",
"""The question is: What color did the sky turn as the sun set over the horizon?
choices: A: white, B: completely green, C: a mix of orange and pink, D: into a checkerboard pattern
Choose one answer from the above choices. Guess: C
""",
"""The question is: What did I do when the doorbell rang?
choices: A: closed the windows, B: started cooking dinner, C: went to bed, D: opened it to see who was there
Choose one answer from the above choices. Guess: D
"""
]
probability_question = "Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\n\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\nProbability:"

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

    with open("data/" + dataset + ".json", "r") as f:
        data = json.load(f)

        # portion
        data["dev"] = data["dev"][:int(len(data["dev"])*float(portion))]
        data["test"] = data["test"][:int(len(data["test"])*float(portion))]
        
        # obtain correct flags for test set
        test_prediction = []
        gold_answer = []
        for d in tqdm(data["test"]):
            original_prompt = prefix[prompt_index] + "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            response = lm_utils.llm_response(original_prompt, model_name, probs=False)

            test_prediction.append(lm_utils.answer_parsing(response))
            gold_answer.append(d["answer"])
            if lm_utils.answer_parsing(response) == d["answer"]:
                correct_flags.append(1)
            else:
                correct_flags.append(0)

        # obtain threshold of asked-for probability for abstain

        correct_labels_dev = []
        probabilities_dev = []
        threshold_optimal = 0

        for d in tqdm(data["dev"]):
            correct_answer = d["answer"]
            original_prompt = prefix[prompt_index] + "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            response = lm_utils.llm_response(original_prompt, model_name)
            if lm_utils.answer_parsing(response) == d["answer"]:
                correct_labels_dev.append(1)
            else:
                correct_labels_dev.append(0)

        for d in tqdm(data["dev"]):
            correct_answer = d["answer"]
            if len(prompts_2[prompt_index])>5:
                one_shot_prompt = "Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n\n"+prompts_2[prompt_index]+probability_question
                one_shot_prompt+="1.0\n"
            else:
                one_shot_prompt = ""
            original_prompt = one_shot_prompt+"Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n\nThe question is: "
            original_prompt += (d["question"] + "\n")
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. Guess:"
            response = lm_utils.llm_response(original_prompt, model_name, probs=False)
            original_prompt += " " + response + "\n"
            original_prompt += "Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\n\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\nProbability:"

            
            response = lm_utils.llm_response(original_prompt, model_name, probs=False)

            prob = None
            num_list = re.findall("\d+\.\d+", response) + ["0" + a for a in re.findall(".\d+", response)]
            for num in num_list:
                try:
                    temp = float(num)
                except:
                    continue
                if temp >= 0 and temp <= 1:
                    prob = temp
                    break
            if prob == None:
                print("prob is not found!")
                prob = 0.5
            assert prob >= 0 and prob <= 1
            probabilities_dev.append(prob)
        min_error = 1e6
        best_threshold = 0
        for threshold in range(1, 100):
            error = 0
            for i in range(len(correct_labels_dev)):
                if probabilities_dev[i] < float(threshold/100.0):
                    if correct_labels_dev[i] == 1:
                        error += 1
                else:
                    if correct_labels_dev[i] == 0:
                        error += 1
            if error < min_error:
                min_error = error
                best_threshold = float(threshold/100.0)
        print("best threshold:", best_threshold)
        threshold_optimal = best_threshold
            

        for d in tqdm(data["test"]):
            correct_answer = d["answer"]
            if len(prompts_2[prompt_index])>5:
                one_shot_prompt = "Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n\n"+prompts_2[prompt_index]+probability_question
                one_shot_prompt+="1.0\n"
            else:
                one_shot_prompt = ""
            original_prompt = one_shot_prompt+"Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n\n"+prompts_2[prompt_index]+"The question is: "
            original_prompt += (d["question"] + "\n")
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. Guess: "
            response = lm_utils.llm_response(original_prompt, model_name, probs=False)
            # print("------------------")
            # print(response)
            original_prompt += " " + response + "\n"
            original_prompt += "Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\n\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\nProbability: "
            response = lm_utils.llm_response(original_prompt, model_name, probs=False)
            prob = None
            num_list = re.findall("\d+\.\d+", response)
            for num in num_list:
                try:
                    temp = float(num)
                except:
                    continue
                if temp >= 0 and temp <= 1:
                    prob = temp
                    break
            if prob == None:
                print("prob is not found!")
                prob = 0.5
            assert prob >= 0 and prob <= 1
            if prob < best_threshold:
                abstain_flags.append(1)
            else:
                abstain_flags.append(0)
            abstain_scores.append(1-prob)

    print("------------------")
    print("Approach: askcalibrate")
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
            "gold_answer": gold_answer[i],
            "threshold": best_threshold
        }
        results.append(result)
    folder_path = f'Probing_Uncertainy/result/{model_name}_{dataset}'

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    output_file = f'Probing_Uncertainy/{model_name}_{dataset}/calibration_{model_name}_{dataset}_setup_{prompt_index}.json'

    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)

    with open(f'Probing_Uncertainy/pickle_result/calibration_{model_name}_{dataset}_setup_{prompt_index}_result.pkl', 'wb') as f:
        pickle.dump(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores), f)
