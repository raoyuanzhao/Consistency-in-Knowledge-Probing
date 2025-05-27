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
E: None of the above
Choose one answer from the above choices. The answer is D
""",
"""Question: What is 2+2?
A: 3
B: 4
C: 5
D: 6
E: None of the above
Choose one answer from the above choices. The answer is B
""",
"""Question: What is the capital of France?
A: Berlin
B: Madrid
C: Paris
D: Rome
E: None of the above
Choose one answer from the above choices. The answer is C
""",
"""Question: What is the chemical symbol for water?
A: H2O
B: CO2
C: NaCl
D: O2
E: None of the above
Choose one answer from the above choices. The answer is A
""",
"""Question: When the lights went out during the storm, they
A: started watching a movie.
B: lit some candles.
C: opened the refrigerator.
D: went swimming in the river.
E: None of the above
Choose one answer from the above choices. The answer is B
""",
"""Question: After the baby started crying, the mother
A: picked up the baby to comfort it.
B: paint the ceiling with a toothbrush.
C: whispered to the toaster.
D: opened an umbrella indoors.
E: None of the above
Choose one answer from the above choices. The answer is A
""",
"""Question: As the sun set over the horizon, the sky turned
A: white.
B: completely green.
C: a mix of orange and pink.
D: into a checkerboard pattern.
E: None of the above
Choose one answer from the above choices. The answer is C
""",
"""Question: When the doorbell rang, I went to the door and
A: closed the windows.
B: started cooking dinner.
C: went to bed.
D: opened it to see who was there.
E: None of the above
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

            options = []
            for key in d["choices"].keys():
                options.append(d["choices"][key])
            options.append("None of the above")
            # shuffle options
            random.shuffle(options)
            problem = {}
            symbols = ["A", "B", "C", "D", "E"]
            for i in range(len(options)):
                problem[symbols[i]] = options[i]
            # find out which is none of the above
            nota_answer = ""
            for key in problem.keys():
                if problem[key] == "None of the above":
                    nota_answer = key
            prompt = prefix[prompt_index] + "Question: " + d["question"] + "\n"
            for key in problem.keys():
                prompt += (key + ": " + problem[key] + "\n")
            prompt += "Choose one answer from the above choices. The answer is"
            response, probs = lm_utils.llm_response(prompt, model_name, probs=True)
            if lm_utils.answer_parsing(response) == nota_answer:
                abstain_flags.append(1)
            else:
                abstain_flags.append(0)

            try:
                for key in probs.keys():
                    found = False
                    for symbol in symbols:
                        if key.strip().lower() == symbol.lower():
                            if symbol == nota_answer:
                                abstain_scores.append(probs[key])
                                found = True
                                break
                            else:
                                abstain_scores.append(1 - probs[key])
                                found = True
                                break
                    if found:
                        break
            except:
                abstain_scores.append(0.5)
    
    print("------------------")
    print("Approach: NOTA")
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
    output_file = f'Probing_Uncertainy/result/{model_name}_{dataset}/nota_{model_name}_{dataset}_setup_{prompt_index}.json'

    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)

    with open(f'Probing_Uncertainy/pickle_result/nota_{model_name}_{dataset}_setup_{prompt_index}_result.pkl', 'wb') as f:
        pickle.dump(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores), f)