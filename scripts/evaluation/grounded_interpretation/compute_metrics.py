import os
import re
import json
import pandas as pd
import json_repair

jsonl_file_path = "path/to/xxxxx-xxxxx-eval_results.jsonl"
excel_file_path = "scripts/evaluation/grounded_interpretation/result.xlsx"
name = "ECG-R1"


eval_llm = "deepseek_v3.1terminus_volengine"

def load_jsonl_to_dict(file_path):
    data_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if item.get('error'):
                        print(f"Skipping error item: {item.get('custom_id', 'unknown')}")
                        continue
                    record_id = item.get('custom_id', "")
                    content = item['response']['body']['choices'][0]['message']['content']
                    data_dict[record_id] = content
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    print(f"Error parsing line: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    return data_dict
print(f"Loading data from {jsonl_file_path}...")

if jsonl_file_path.endswith('.jsonl'):
    json_data = load_jsonl_to_dict(jsonl_file_path)
elif jsonl_file_path.endswith('.json'):
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
print(f"Loaded {len(json_data)} records.")


expected_keys = [
    'DiagnosisAccuracy',
    'AnalysisCompleteness',
    'AnalysisRelevance',
    'LeadEvidenceValidity',
    'GroundedECGUnderstanding','ECGFeatureGrounding',
    'EvidenceBasedReasoning', 'Evidence-BasedReasoning',
    'RealisticDiagnosticProcess','ClinicalDiagnosticFidelity'
]

pattern = re.compile(r'\"(?P<key>{})\":\s*(?P<content>\[.*?\])'.format('|'.join(expected_keys)), re.DOTALL)

result = {}


def fix_unterminated_string(content):
    quote_count = len(re.findall(r'(?<!\\)"', content))
    if quote_count % 2 == 1:
        content = re.sub(r'(\s*[}\]])', r'"\1', content, count=1)
    return content

def escape_inner_quotes_in_explanation(content):
    def replacer(match):
        explanation = match.group(1)
        fixed = re.sub(r'(?<!\\)"', r'\\"', explanation)
        return f'"Explanation": "{fixed}"'
    return re.sub(r'"Explanation":\s*"([^"]*?)"', replacer, content)

def remove_extra_quotes(content):
    content = re.sub(r'""+', '"', content)
    return content

def fix_unmatched_brackets(content):
    def replacer(match):
        explanation = match.group(1)
        fixed = re.sub(r'[\[\]]', '', explanation)
        return f'"Explanation": "{fixed}"'
    return re.sub(r'"Explanation":\s*"([^"]*?)"', replacer, content)


def fix_missing_commas(content):
    content = re.sub(r'(\})(\s*\{)', r'\1,\2', content)
    return content

def safe_eval(match):
    try:
        return str(eval(match.group(1)))
    except:
        return match.group(1)  # Return the original string if eval fails

def manual_fix_string(text):
    text = re.sub(r'\[\s*"Score":', r'[{"Score":', text)
    text = re.sub(r'("Score":\s*\d+(\.\d+)?)\s*"Explanation"', r'\1, "Explanation"', text)
    text = re.sub(r'\}\s*"Score"', r'}, {"Score"', text)
    
    return text

for id, content in json_data.items():
    
    if isinstance(content, dict):
        result[id] = content
        continue
    
    if isinstance(content, str):
        json_content = content.strip('```json\n').strip('\n```')
        result[id] = {}

        matches = pattern.finditer(json_content)
        
        for match in matches:
            key = match.group('key')
            content = match.group('content')
            content = manual_fix_string(content)

            # Original cleaning steps
            content = content.replace("\"", '"').replace("“", '"').replace("”", '"')
            content = re.sub(r'//.*', '', content)
            content = re.sub(r',\s*([}\]])', r'\1', content)
            content = re.sub(r'"\s*"', ' ', content)
            content = re.sub(r'\+(\d)', r'\1', content)
            content = re.sub(r'(\d+[\d\s\*\+\-\/]+\d+)', safe_eval, content)
            content = content.replace('");', '"')
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'\n|\r', ' ', content)

            # Remove unmatched trailing characters after quote
            content = re.sub(r'"\s*[\]\)]', '"', content)
            # Fix missing commas between JSON objects in arrays
            content = re.sub(r'\}\s*\{', '},{', content)
            content = re.sub(r'("Explanation":\s*".*?)(?<!\\)"\s*,?\s*\{', r'\1"}, {', content)

            # Additional cleaning steps
            content = fix_unterminated_string(content)
            content = escape_inner_quotes_in_explanation(content)
            content = remove_extra_quotes(content)
            content = fix_unmatched_brackets(content)
            content = fix_missing_commas(content)

            open_braces = content.count('{')
            close_braces = content.count('}')
            if open_braces > close_braces:
                content += '}' * (open_braces - close_braces)
            
            open_brackets = content.count('[')
            close_brackets = content.count(']')
            if open_brackets > close_brackets:
                content += ']' * (open_brackets - close_brackets)

            try:
                # content_json = json.loads(content)
                content_json = json_repair.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON decoding error for id {id}, key {key}: {e}")
                # print("Content:", content) # Optional: comment out to reduce noise
                continue

            scores = []
            explanations = []

            for item in content_json:
                if not isinstance(item, dict):
                    print(id, item)
                    # print(content_json)
                    continue
                
                try:
                    score = item.get('Score')
                    if score is None:
                        score = item.get(' Score')
                    if score is None:
                        print(id, item)
                        # print(content_json)
                        continue
                except:
                    print(id, item)
                    
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    str_val = str(score)
                    match = re.search(r'(\d+(\.\d+)?)', str_val)
                    if match:
                        score = float(match.group(1))
                    else:
                        print(id, "error")

                explanation = item.get('Explanation', '').strip()
                extra_fields = {k: v for k, v in item.items() if k not in ['Score', 'Explanation']}

                if extra_fields:
                    explanation += " Additional details: " + json.dumps(extra_fields)

                scores.append(score)
                explanations.append(explanation)

            result[id][key] = {
                'Scores': scores,
                'Explanations': explanations
            }

base_dir = os.path.dirname(jsonl_file_path)
file_name = os.path.basename(jsonl_file_path)
clean_dir = os.path.join(base_dir, "clean")
if not os.path.exists(clean_dir):
    os.makedirs(clean_dir)
name_without_ext = os.path.splitext(file_name)[0]
output_json_path = os.path.join(clean_dir, f"{name_without_ext}.json")
try:
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"\nSuccessfully saved cleaned data to: {output_json_path}")
except Exception as e:
    print(f"Failed to save JSON file: {e}")


RENAME_MAP = {
    'GroundedECGUnderstanding': 'ECGFeatureGrounding',
    'EvidenceBasedReasoning': 'Evidence-BasedReasoning',
    'RealisticDiagnosticProcess': 'ClinicalDiagnosticFidelity'
}
results = {}
for id, content in result.items():
    results[id] = {}
    for key, value in content.items():
        key = RENAME_MAP.get(key, key)
        
        if key in ['LeadEvidenceValidity', 'AnalysisCompleteness', 'AnalysisRelevance', 'ECGFeatureGrounding', 'Evidence-BasedReasoning', 'ClinicalDiagnosticFidelity']:
            average = sum(value['Scores'])
        else:
            average = len([x for x in value['Scores'] if x > 0] ) / (len(value['Scores'])) if value['Scores'] else 0
            # current_scores = value['Scores']
            # filtered_lst = [x for x in current_scores if x > 0]
            # print(filtered_lst)
            # average = sum(filtered_lst) / len(filtered_lst) if filtered_lst else 0

        results[id][key] = average

df = pd.DataFrame(results).T

if 'DiagnosisAccuracy' in df.columns:
    df['DiagnosisAccuracy'] = df['DiagnosisAccuracy'] * 100


# Add Average Column (GroundedECGUnderstanding + EvidenceBasedReasoning + RealisticDiagnosticProcess) / 3
avg_cols = ['ECGFeatureGrounding', 'Evidence-BasedReasoning', 'ClinicalDiagnosticFidelity']
# Only calculate if these columns exist to avoid errors
existing_avg_cols = [col for col in avg_cols if col in df.columns]
if len(existing_avg_cols) == 3:
    df['Average'] = df[existing_avg_cols].mean(axis=1)    
print(df.mean().round(2))

# Save to Excel
# Calculate the mean for the current run
summary_stats = df.mean().round(2).to_dict()
summary_stats['name'] = name
summary_stats['eval_llm'] = eval_llm

# Define the order of columns
final_keys_order = [
    'name', 
    'eval_llm',
    'DiagnosisAccuracy',
    'AnalysisCompleteness',
    'AnalysisRelevance',
    'LeadEvidenceValidity',
    'ECGFeatureGrounding',
    'Evidence-BasedReasoning',
    'ClinicalDiagnosticFidelity',
    'Average'
]

if "drop" in excel_file_path:
    match = re.search(r'drop(\d+)\.jsonl', jsonl_file_path)
    if match:
        drop_ratio = int(match.group(1))
        summary_stats['missing_ratio'] = drop_ratio
        print(f"Detected missing ratio: {drop_ratio}")
        
        if 'drop_rmissing_ratio' not in final_keys_order:
            final_keys_order.insert(1, 'missing_ratio')
    else:
        print("Warning: 'result_ecgdrop.xlsx' target detected but could not extract drop ratio from filename.")

# Create a DataFrame for the new row
new_row_df = pd.DataFrame([summary_stats])
# Reorder columns, filling missing ones with NaN if any
new_row_df = new_row_df.reindex(columns=final_keys_order)

try:
    if os.path.exists(excel_file_path):
        # Load existing file to match columns/append
        existing_df = pd.read_excel(excel_file_path)
        combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        combined_df.to_excel(excel_file_path, index=False)
    else:
        # Create new file
        new_row_df.to_excel(excel_file_path, index=False)
    print(f"Results appended to {excel_file_path}")
except Exception as e:
    print(f"Failed to save to Excel: {e}")
