import json
import jinja2
import os
import argparse
from tqdm import tqdm
from volcenginesdkarkruntime import Ark

class Config:
    # your volcengine online inference API KEY
    API_KEY = ""
    BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    MODEL = "deepseek-v3-1-terminus"

def construct_prompt(template_raw, generated_text, groundtruth_text):
    template = jinja2.Template(template_raw, trim_blocks=True, lstrip_blocks=True)
    return template.render(generated=generated_text, groundtruth=groundtruth_text)

def call_model_api(client, prompt):
    try:
        completion = client.chat.completions.create(
            model=Config.MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            top_p=None,
            thinking={"type": "disabled"}
        )
        return completion
    except Exception as e:
        print(f"\nAPI call failed: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Evaluation using Volcano Engine Ark API.")
    parser.add_argument("-i", "--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("-o", "--end", type=int, default=2382, help="End index (exclusive)")
    args = parser.parse_args()

    gen_path = 'ECG-R1 generate jsonl result path'
    gt_path = 'ecg-grounding-test-mimiciv_full.jsonl'
    prompt_template_path = 'scripts/evaluation/grounded_interpretation/interpretation_evaluation_prompt.txt'
    
    output_dir = f"{os.path.dirname(os.path.dirname(gen_path))}/eval_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_id = gen_path.split('/')[-1]
    output_file_path = os.path.join(output_dir, file_id)
    print("--- Starting processing ---")
    print(f"Results will be appended to: {output_file_path}")

    client = Ark(
        base_url=Config.BASE_URL,
        api_key=Config.API_KEY,
    )

    with open(prompt_template_path, "r", encoding='utf-8') as f:
        template_raw = f.read()

    with open(gt_path, 'r', encoding='utf-8') as f_gt, \
         open(gen_path, 'r', encoding='utf-8') as f_gen:
        
        all_gt_lines = f_gt.readlines()
        all_gen_lines = f_gen.readlines()
        batch_gt_lines = all_gt_lines[args.start:args.end]
        batch_gen_lines = all_gen_lines[args.start:args.end]
        print(f"Processing records {args.start} to {args.end}, total {len(batch_gt_lines)}...")

        with open(output_file_path, 'a', encoding='utf-8') as f_out:
            for gt_line, gen_line in tqdm(zip(batch_gt_lines, batch_gen_lines), total=len(batch_gt_lines)):
                try:
                    gt_record = json.loads(gt_line)
                    gen_record = json.loads(gen_line)
                    gt_id = gt_record.get("id")
                    gen_id_extracted = None
                    gen_images = gen_record.get("images", [])
                    if gen_images and len(gen_images) > 0:
                        gen_img_path = gen_images[0].get("path", "")
                        if gen_img_path:
                            filename = os.path.basename(gen_img_path)
                            gen_id_extracted = filename.split('-')[0]
                    if not gen_id_extracted:
                        gen_objects = gen_record.get("objects", {})
                        ecg_list = gen_objects.get("ecg", [])
                        if ecg_list and len(ecg_list) > 0:
                            gen_id_extracted = os.path.basename(ecg_list[0])
                            
                    if not gen_id_extracted or str(gt_id) != str(gen_id_extracted):
                        print(f"\nSkipping mismatch. GT ID: {gt_id}, extracted: {gen_id_extracted}")
                        continue
                    groundtruth_text = gt_record.get("messages", [{}, {}, {}])[-1].get("content", "")
                    generated_text = gen_record.get("response", "")
                    prompt = construct_prompt(template_raw, generated_text, groundtruth_text)
                    
                    completion = call_model_api(client, prompt)
                    
                    if completion:
                        try:
                            completion_dict = completion.model_dump() 
                        except AttributeError:
                            completion_dict = completion.to_dict() if hasattr(completion, 'to_dict') else dict(completion)

                        formatted_output = {
                            "id": completion.id,
                            "custom_id": gt_id,
                            "response": {
                                "request_id": completion.id,
                                "body": completion_dict
                            }
                        }
                        f_out.write(json.dumps(formatted_output, ensure_ascii=False) + '\n')
                        f_out.flush()
                        
                except Exception as e:
                    print(f"\nFailed to process record (GT ID: {gt_record.get('id', 'Unknown')}): {e}")
                    continue

    print(f"\nDone. Output file: {output_file_path}")
