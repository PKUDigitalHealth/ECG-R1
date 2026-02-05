import os
import json
import random
random.seed(42)
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score,hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

def extract(text):
    if text in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        return text
    for char_dot in ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H."]:
        if char_dot in text:
            return char_dot[0]
    if "The correct option is " in text:
        predict_char = text.split("The correct option is ")[-1][0]
        if predict_char in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            return predict_char
        else:
            return None
    if "Answer:" in text:
        answer = text.split("Answer:")[-1].strip()
        # if answer in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        #   return answer
        for char in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            if char in answer:
                return char
        else:
            return None
    else:
        return None
    
def compute_f1_auc(y_pred, y_true):
    # Binarize labels
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    # print(y_true)
    # print(y_true_bin)
    hl = hamming_loss(y_true_bin, y_pred_bin)
    
    f1_scores_all = []
    # Compute the F1 score
    f1_scores = f1_score(y_true_bin, y_pred_bin, average=None)
    for idx, cls in enumerate(mlb.classes_):
        # print(f'F1 score for class {cls}: {f1_scores[idx]}')
        f1_scores_all.append(f1_scores[idx])
    
    # Compute the AUC score
    auc_scores = []
    for i in range(y_true_bin.shape[1]):
        try:
            auc = roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i])
        except ValueError:
            auc = np.nan    # If AUC cannot be calculated, NaN is returned
        auc_scores.append(auc)
        # print(f'AUC score for class {mlb.classes_[i]}: {auc}')    
    # print("f1 all",np.mean(f1_scores_all), "auc all", np.mean(auc_scores))
    return np.mean(f1_scores_all), np.mean(auc_scores), hl
    
def eval_mmmu(dir):
    print("====mmmu====")
    with open("", "r", encoding='utf-8') as f:
        data = json.load(f)
        answer_dict = {item["No"]: item["conversations"][1]["value"] for item in data}
    
    score_dict = {}
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jsonl"):
                predict_list = []
                golden_list = []
                file_path = os.path.join(root, file)
                if "mmmu-ecg" not in file_path:
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "question_id" in item:
                            qid = item["question_id"]
                        elif "id" in item:
                            qid = item["id"]
                        if "text" in item:
                            predict = extract(item["text"])
                        elif "response" in item:
                            predict = extract(item["response"])
                        if predict is None:
                            predict = random.choice(["A", "B", "C", "D"])
                        # print(predict)
                        golden = answer_dict[qid]
                        predict_list.append(predict)
                        golden_list.append(golden)
                # print(predict_list)
                if len(predict_list) != 200:
                    continue

                accuracy = accuracy_score(golden_list, predict_list)
                
                print(file, f"Accuracy: {accuracy}")
                if "step" in file:
                    if file.split("-")[-1].split(".")[0] == "final":
                        step_num = 99999
                    else:
                        step_num = int(file.split("-")[-1].split(".")[0])
                else:
                    step_num = file.split("_")[0]
                # step_num = int(file.split("-")[-1].split(".")[0])
                score_dict[step_num] = accuracy
    for step_num in sorted(score_dict):
        if step_num == 99999:
            print(f"Model final Accuracy: {score_dict[step_num]:.4f}")
        else:
            print(f"Model {step_num} Accuracy: {score_dict[step_num]:.4f}")

def eval_ptb_test(dir):
    print("====ptb test====")
    label_space = ["NORM","MI","STTC","CD","HYP"]
    golden_data_path = "download from https://huggingface.co/datasets/LANSG/ECG-Grounding/tree/main/ecg_bench/ptb-test.json"
    golden_label = {}
    with open(golden_data_path, "r", encoding='utf-8') as f:
        golden_data = json.load(f)
        for item in golden_data:
            qid = item["id"]
            golden_label[qid] = [label for label in label_space if label in item["conversations"][1]["value"]]
        
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                if "ptb-test" not in file_path or "report" in file_path:
                    continue
                predict_list = []
                golden_list = []
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "question_id" in item:
                            qid = item["question_id"]
                        elif "id" in item:
                            qid = item["id"]
                        if "text" in item:
                            predict = [label for label in label_space if label in item["text"]]
                        elif "response" in item:
                            predict = [label for label in label_space if label in item["response"]]
                        
                        true = golden_label[qid]
                        predict_list.append(predict)
                        golden_list.append(true)
                f1, auc, hl = compute_f1_auc(predict_list, golden_list)
                print(file, "f1", round(f1*100, 1), "auc", round(auc*100, 1), "hl", round(hl*100, 1))
                return {"PTB-XL_Super AUC": round(auc*100, 1), "PTB-XL_Super F1": round(f1*100, 1), "PTB-XL_Super HL": round(hl*100, 1)}
    return {}

                
def eval_cpsc_test(dir):
    print("====cpsc test====")
    label_space = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
    golden_data_path = "download from https://huggingface.co/datasets/LANSG/ECG-Grounding/tree/main/ecg_bench//cpsc-test.json"
    golden_label = {}
    with open(golden_data_path, "r", encoding='utf-8') as f:
        golden_data = json.load(f)
        for item in golden_data:
            qid = item["id"]
            golden_label[qid] = [label for label in label_space if label in item["conversations"][1]["value"]]
    
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                if "cpsc-test" not in file_path:
                    continue
                predict_list = []
                golden_list = []
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "question_id" in item:
                            qid = item["question_id"]
                        elif "id" in item:
                            qid = item["id"]
                        if "text" in item:
                            predict = [label for label in label_space if label in item["text"]]
                        elif "response" in item:
                            predict = [label for label in label_space if label in item["response"]]
                        true = golden_label[qid]
                        predict_list.append(predict)
                        golden_list.append(true)
                        
                f1, auc, hl = compute_f1_auc(predict_list, golden_list)
                print(file, "f1", round(f1*100, 1), "auc", round(auc*100, 1), "hl", round(hl*100, 1))
                
                return {"CPSC_2018_AUC": round(auc*100, 1), "CPSC_2018_F1": round(f1*100, 1), "CPSC_2018_HL": round(hl*100, 1)}
    return {}

def eval_ecgqa_test(dir):
    print("====ecgqa test====")
    golden_data_path = "" # your path to ecgqa test
    golden_label = {}
    with open(golden_data_path, "r", encoding='utf-8') as f:
        golden_data = json.load(f)
        for item in golden_data:
            qid = item["id"]
            golden_label[qid] = item["conversations"][1]["value"]
            
    score_dict = {}
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                if "ecgqa-test" not in file_path:
                    continue
                
                pass_list = []
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "question_id" in item:
                            qid = item["question_id"]
                        elif "id" in item:
                            qid = item["id"]
                        
                        if "prompt" in item:
                            if isinstance(item["prompt"], dict):
                                candidates = [i.strip() for i in item["prompt"]["prompt"].split("Options:")[-1].replace("Only answer based on the given Options without any explanation.","").split(",")]
                            else:
                                candidates = [i.strip() for i in item["prompt"].split("Options:")[-1].replace("Only answer based on the given Options without any explanation.","").split(",")]
                        
                        if "text" in item:
                            predict = [i for i in candidates if i.lower() in item["text"].lower()]
                        elif "response" in item:
                            predict = [i for i in candidates if i in item["response"].lower()]
                        if isinstance(predict, list):
                            predict_str = ''.join(predict)
                        else:
                            predict_str = predict
                        if set(predict_str) == set(golden_label[qid]):
                            pass_list.append(1)
                        else:
                            pass_list.append(0)
                accuracy = sum(pass_list) / len(pass_list)
                print(file,"accuracy",accuracy)
                if "step" in file:
                    if file.split("-")[-1].split(".")[0] == "final":
                        step_num = 99999
                    else:
                        step_num = int(file.split("-")[-1].split(".")[0])
                else:
                    step_num = file.split("_")[0]
                
                score_dict[step_num] = accuracy
    for step_num in sorted(score_dict):
        if step_num == 99999:
            print(f"Model final Accuracy: {score_dict[step_num]:.4f}")
        else:
            print(f"Model {step_num} Accuracy: {score_dict[step_num]:.4f}")
                            
def eval_code15_test(dir):
    print("====code15 test====")
    label_space = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
    golden_data_path = "download from https://huggingface.co/datasets/LANSG/ECG-Grounding/tree/main/ecg_bench//code15-test.json"
    golden_label = {}
    with open(golden_data_path, "r", encoding='utf-8') as f:
        golden_data = json.load(f)
        for item in golden_data:
            qid = item["id"]
            if item["conversations"][1]["value"] == "NORM":
                golden_label[qid] = ["NORM"]
            elif item["conversations"][1]["value"] == "ABNORMAL":
                golden_label[qid] = ["ABNORMAL"]
            else:
                golden_label[qid] = [label for label in label_space if label in item["conversations"][1]["value"]]
        
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                if "code15-test" not in file_path:
                    continue
                predict_list = []
                golden_list = []
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "question_id" in item:
                            qid = item["question_id"]
                        elif "id" in item:
                            qid = item["id"]
                        if "text" in item:
                            if "Answer:" in item["text"]:
                                item["text"] = item["text"].split("Answer:")[-1]
                            if "NORM" in item["text"] and "ABNORMAL" not in item["text"]:
                                predict = ["NORM"] + [label for label in label_space if label in item["text"]]
                            elif "ABNORMAL" in item["text"]:
                                predict = ["ABNORMAL"] + [label for label in label_space if label in item["text"]]
                            else:
                                predict = [label for label in label_space if label in item["text"]]
                        elif "response" in item:
                            if "Answer:" in item["response"]:
                                item["response"] = item["response"].split("Answer:")[-1]
                            if "NORM" in item["response"] and "ABNORMAL" not in item["response"]:
                                predict = ["NORM"] + [label for label in label_space if label in item["response"]]
                            elif "ABNORMAL" in item["response"]:
                                predict = ["ABNORMAL"] + [label for label in label_space if label in item["response"]]
                            else:
                                predict = [label for label in label_space if label in item["response"]]
                    
                        true = golden_label[qid]
                        predict_list.append(predict)
                        golden_list.append(true)
                f1, auc, hl = compute_f1_auc(predict_list, golden_list)
                print(file, "f1", round(f1*100, 1), "auc", round(auc*100, 1), "hl", round(hl*100, 1))
                
                return {"CODE-15%_AUC": round(auc*100, 1), "CODE-15%_F1": round(f1*100, 1), "CODE-15%_HL": round(hl*100, 1)}
    return {}

        
def eval_csn_test(dir):
    print("====csn test====")
    with open("download from https://huggingface.co/datasets/LANSG/ECG-Grounding/tree/main/ecg_bench//csn-test-no-cot.json", "r", encoding='utf-8') as f:
        data = json.load(f)
        answer_dict = {item["id"]: item["conversations"][1]["value"][0] for item in data}
    
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                if "csn-test" not in file_path:
                    continue
            
                predict_list = []
                golden_list = []
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "question_id" in item:
                            qid = item["question_id"]
                        elif "id" in item:
                            qid = item["id"]
                        if "text" in item:
                            predict = extract(item["text"])
                        elif "response" in item:
                            predict = extract(item["response"])
                        if predict is None:
                            predict = random.choice(["A", "B", "C", "D", "E", "F", "G", "H"])
                        
                        golden = answer_dict[qid]
                        predict_list.append(predict)
                        golden_list.append(golden)
                
                if len(predict_list) != 1611:
                    continue

                accuracy = accuracy_score(golden_list, predict_list)
                print(file, f"Accuracy: {accuracy}")
                return {"CSN_Accuracy": round(accuracy*100, 1)}
    return {}

        
def eval_g12_test(dir):
    print("====g12 test====")
    with open("download from https://huggingface.co/datasets/LANSG/ECG-Grounding/tree/main/ecg_bench//g12-test-no-cot.json", "r", encoding='utf-8') as f:
        data = json.load(f)
        answer_dict = {item["id"]: item["conversations"][1]["value"][0] for item in data}
    
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                if "g12-test" not in file_path:
                    continue
                predict_list = []
                golden_list = []
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "question_id" in item:
                            qid = item["question_id"]
                        elif "id" in item:
                            qid = item["id"]
                        if "text" in item:
                            predict = extract(item["text"])
                        elif "response" in item:
                            predict = extract(item["response"])
                        if predict is None:
                            predict = random.choice(["A", "B", "C", "D", "E", "F", "G", "H"])
                        
                        golden = answer_dict[qid]
                        predict_list.append(predict)
                        golden_list.append(golden)
                
                if len(predict_list) != 2026:
                    continue

                accuracy = accuracy_score(golden_list, predict_list)
                print(file, f"Accuracy: {accuracy}")
                return {"G12EC_Accuracy": round(accuracy*100, 1)}
    return {}


if __name__ == "__main__":
    MODEL_NAME="ECG-R1-8B-RL"
    root = f"scripts/evaluation/ecgbench/result_processed_for_eval/{MODEL_NAME}"
    
    all_results = {}
    all_results.update(eval_ptb_test(root))
    all_results.update(eval_cpsc_test(root))
    all_results.update(eval_csn_test(root))
    all_results.update(eval_g12_test(root))
    all_results.update(eval_code15_test(root))
    # all_results.update(eval_ecgqa_test(root))
    print("\n--- All evaluations completed, preparing output ---")

    columns_tuples = [
        ('PTB-XL Super', 'AUC'),
        ('PTB-XL Super', 'F1'),
        ('PTB-XL Super', 'HL'),
        ('CODE-15%', 'AUC'),
        ('CODE-15%', 'F1'),
        ('CODE-15%', 'HL'),
        ('CPSC 2018', 'AUC'),
        ('CPSC 2018', 'F1'),
        ('CPSC 2018', 'HL'),
        ('CSN', 'Accuracy'),
        ('G12EC', 'Accuracy')
    ]
    columns_index = pd.MultiIndex.from_tuples(columns_tuples, names=['Datasets', 'Metric'])
    
    data_row = {
        ('PTB-XL Super', 'AUC'): all_results.get("PTB-XL_Super AUC"),
        ('PTB-XL Super', 'F1'): all_results.get("PTB-XL_Super F1"),
        ('PTB-XL Super', 'HL'): all_results.get("PTB-XL_Super HL"),
        ('CODE-15%', 'AUC'): all_results.get("CODE-15%_AUC"),
        ('CODE-15%', 'F1'): all_results.get("CODE-15%_F1"),
        ('CODE-15%', 'HL'): all_results.get("CODE-15%_HL"),
        ('CPSC 2018', 'AUC'): all_results.get("CPSC_2018_AUC"),
        ('CPSC 2018', 'F1'): all_results.get("CPSC_2018_F1"),
        ('CPSC 2018', 'HL'): all_results.get("CPSC_2018_HL"),
        ('CSN', 'Accuracy'): all_results.get("CSN_Accuracy"),
        ('G12EC', 'Accuracy'): all_results.get("G12EC_Accuracy")
    }

    df = pd.DataFrame([data_row], index=[MODEL_NAME], columns=columns_index)
    df.index.name = "Model"

    output_excel_path = "scripts/evalutaion/ecgbench/results.xlsx"
    
    try:
        if os.path.exists(output_excel_path):
            print(f"Existing file detected: {output_excel_path}")
            existing_df = pd.read_excel(output_excel_path, header=[0, 1], index_col=0)
            if MODEL_NAME in existing_df.index:
                print(f" -> Updating model: {MODEL_NAME}")
                existing_df = existing_df.drop(MODEL_NAME)
            else:
                print(f" -> Adding new model: {MODEL_NAME}")
            
            combined_df = pd.concat([existing_df, df])
            combined_df.to_excel(output_excel_path)
            
            print("\n==============================================")
            print(f"Success! Results were *updated* at: {output_excel_path}")
            print("==============================================")

        else:
            print(f"No existing file found, creating: {output_excel_path}")
            df.to_excel(output_excel_path)
            print("\n==============================================")
            print(f"Success! Results were *created* at: {output_excel_path}")
            print("==============================================")
            
    except Exception as e:
        print(f"\n==============================================")
        print("Error: Failed to save to Excel.")
        print("Please ensure 'pandas' and 'openpyxl' are installed (run: pip install pandas openpyxl)")
        print(f"Error details: {e}")
        print("==============================================")
