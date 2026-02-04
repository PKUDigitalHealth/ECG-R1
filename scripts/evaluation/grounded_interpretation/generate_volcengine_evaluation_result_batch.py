import json
import jinja2
import os
import argparse
import asyncio
import sys
from datetime import datetime
from tqdm.asyncio import tqdm
from volcenginesdkarkruntime import AsyncArk

class Config:
    # your volcengine online inference API KEY
    API_KEY = ""
    MAX_CONCURRENT_TASKS = 25
    MODEL = "deepseek-v3-1-terminus"

def construct_prompt(template_raw, generated_text, groundtruth_text):
    template = jinja2.Template(template_raw, trim_blocks=True, lstrip_blocks=True)
    return template.render(generated=generated_text, groundtruth=groundtruth_text)

async def worker(
    worker_id: int,
    client: AsyncArk,
    queue: asyncio.Queue,
    file_lock: asyncio.Lock,
    output_file_path: str,
    pbar: tqdm
):
    while True:
        item = await queue.get()
        custom_id, prompt = item
        try:
            completion = await client.chat.completions.create(
                model=Config.MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                top_p=None,
                thinking={"type": "disabled"}
            )

            if completion:
                try:
                    completion_dict = completion.model_dump() 
                except AttributeError:
                    completion_dict = completion.to_dict() if hasattr(completion, 'to_dict') else dict(completion)

                formatted_output = {
                    "id": completion.id,
                    "custom_id": custom_id,
                    "response": {
                        "request_id": completion.id,
                        "body": completion_dict
                    }
                }

                async with file_lock:
                    with open(output_file_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(formatted_output, ensure_ascii=False) + '\n')
                        f.flush()

        except Exception as e:
            print(f"\n[Error] ID {custom_id}: {e}", file=sys.stderr)
        finally:
            queue.task_done()
            pbar.update(1)


async def main():
    parser = argparse.ArgumentParser(description="Async Batch Evaluation.")
    parser.add_argument("-i", "--start", type=int, default=0, help="Start index")
    parser.add_argument("-o", "--end", type=int, default=2382, help="End index")
    args = parser.parse_args()

    gen_path = 'ECG-R1 generate jsonl result path'
    gt_path = 'ecg-grounding-test-mimiciv_full.jsonl'
    prompt_template_path = 'scripts/evaluation/grounded_interpretation/interpretation_evaluation_prompt.txt'
    
    output_dir = f"{os.path.dirname(os.path.dirname(gen_path))}/eval_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_id = gen_path.split('/')[-1]
    output_file_path = os.path.join(output_dir, file_id)
    print("--- Preparing inputs ---")
    print(f"Output file: {output_file_path}")

    with open(prompt_template_path, "r", encoding='utf-8') as f:
        template_raw = f.read()

    tasks_data = []
    with open(gt_path, 'r', encoding='utf-8') as f_gt, \
         open(gen_path, 'r', encoding='utf-8') as f_gen:
        
        all_gt_lines = f_gt.readlines()
        all_gen_lines = f_gen.readlines()
        end_index = min(args.end, len(all_gt_lines))
        batch_gt = all_gt_lines[args.start:end_index]
        batch_gen = all_gen_lines[args.start:end_index]
        print(f"Parsing inputs... (Range: {args.start}-{end_index})")
        
        for gt_line, gen_line in zip(batch_gt, batch_gen):
            try:
                gt_record = json.loads(gt_line)
                gen_record = json.loads(gen_line)
                gt_id = gt_record.get("id")
                gen_id_extracted = None
                gen_images = gen_record.get("images", [])
                if gen_images:
                    path = gen_images[0].get("path", "")
                    if path: gen_id_extracted = os.path.basename(path).split('-')[0]
                if not gen_id_extracted:
                    gen_objects = gen_record.get("objects", {})
                    ecg_list = gen_objects.get("ecg", [])
                    if ecg_list: gen_id_extracted = os.path.basename(ecg_list[0])
                
                if not gen_id_extracted or str(gt_id) != str(gen_id_extracted):
                    print(f"Skipping mismatch: GT {gt_id} != GEN {gen_id_extracted}")
                    continue

                groundtruth_text = gt_record.get("messages", [{}, {}, {}])[-1].get("content", "")
                generated_text = gen_record.get("response", "")
                prompt = construct_prompt(template_raw, generated_text, groundtruth_text)
                tasks_data.append((gt_id, prompt))
                
            except Exception as e:
                continue
    print(f"Valid tasks: {len(tasks_data)}")

    client = AsyncArk(
        api_key=Config.API_KEY,
        timeout=24 * 3600,
    )
    queue = asyncio.Queue()
    file_lock = asyncio.Lock()
    for item in tasks_data:
        queue.put_nowait(item)
    start_time = datetime.now()
    pbar = tqdm(total=len(tasks_data), desc="Processing")
    workers = []
    num_workers = min(Config.MAX_CONCURRENT_TASKS, len(tasks_data))
    for i in range(num_workers):
        task = asyncio.create_task(
            worker(i, client, queue, file_lock, output_file_path, pbar)
        )
        workers.append(task)
    print(f"Started {num_workers} concurrent workers, running inference...")
    await queue.join()
    for task in workers:
        task.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    await client.close()
    pbar.close()
    end_time = datetime.now()
    print(f"\nDone. Elapsed: {end_time - start_time}")
    print(f"Results written to: {output_file_path}")

if __name__ == "__main__":
    asyncio.run(main())
