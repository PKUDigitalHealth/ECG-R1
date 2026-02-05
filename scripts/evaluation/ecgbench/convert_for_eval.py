#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
转换脚本: 将 LLM 的输出 .jsonl 注入缺失的 'id' 字段。

它通过匹配 'image_path'，从原始的 "golden_data" JSON 文件中查找 'id'，
并将 'id' 添加到 LLM 输出的 .jsonl 行中，使其可被评估脚本处理。
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# --- 辅助函数 ---

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件，返回一个字典列表。"""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def read_json(path: Path) -> Any:
    """读取一个标准 JSON 文件 (列表或字典)。"""
    with path.open("r", encoding="utf-8") as fr:
        return json.load(fr)

def write_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    """将字典列表写入 JSONL 文件。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fw:
        for obj in records:
            fw.write(json.dumps(obj, ensure_ascii=False) + "\n")

# --- 核心逻辑 ---

def create_id_lookup(golden_data_path: Path) -> Dict[str, str]:
    """
    从原始 "golden_data" JSON 文件创建一个
    {image_path: id} 的速查表。
    """
    print(f"创建ID速查表: {golden_data_path.name}")
    lookup_map = {}
    data = read_json(golden_data_path)
    
    for item in tqdm(data, desc=" ├─ 读取 Golden Data"):
        image_path = item.get("image")  # e.g., "code15/1109348-0.png"
        item_id = item.get("id")

        if image_path and item_id:
            lookup_map[image_path] = item_id
        else:
            print(f"  └─ 警告: Golden Data 中缺少 'image' 或 'id': {str(item)[:100]}")
            
    print(f"  └─ 速查表创建完毕 ({len(lookup_map)} 条目)")
    return lookup_map

def convert_llm_output(
    llm_output_path: Path,
    golden_data_path: Path,
    converted_output_path: Path
):
    """
    执行转换，将 'id' 注入 LLM 输出。
    """
    # 1. 创建速查表
    lookup_map = create_id_lookup(golden_data_path)
    if not lookup_map:
        print(f"错误: 无法从 {golden_data_path} 创建速查表。")
        return

    # 2. 读取 LLM 输出
    print(f"读取 LLM 输出: {llm_output_path.name}")
    llm_outputs = read_jsonl(llm_output_path)
    
    # 3. 转换
    converted_records = []
    missing_count = 0
    
    for item in tqdm(llm_outputs, desc=" ├─ 转换 LLM 输出"):
        # 提取 LLM 输出中的图像路径
        if "images" not in item or not item["images"]:
            print(f"  └─ 警告: LLM 输出中缺少 'images' 字段，跳过。")
            missing_count += 1
            continue
            
        image_path = item["images"][0].get("path", "")
        if not image_path:
            print(f"  └─ 警告: 'images' 字段中缺少 'path'，跳过。")
            missing_count += 1
            continue
            
        # 4. 查找 ID 并注入
        if image_path in lookup_map:
            found_id = lookup_map[image_path]
            # 注入 'id' 字段
            item["id"] = found_id
            converted_records.append(item)
        else:
            print(f"  └─ 警告: 无法在速查表中匹配图像路径 '{image_path}'，跳过。")
            missing_count += 1
    
    if missing_count > 0:
        print(f"  └─ 转换期间共跳过了 {missing_count} 条记录。")

    # 5. 写入新文件
    print(f" ├─ 正在写入 {len(converted_records)} 条转换后的记录...")
    write_jsonl(converted_records, converted_output_path)
    print(f"└─ 转换成功! -> {converted_output_path}")

# --- 命令行入口 ---

def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 'id' 注入 LLM 输出 .jsonl 文件，使其可被评估。"
    )
    parser.add_argument(
        "--llm_output_file", 
        type=Path, 
        required=True,
        help="LLM 输出的 .jsonl 文件路径 (e.g., .../result/.../code15-test.jsonl)"
    )
    parser.add_argument(
        "--golden_data_file", 
        type=Path, 
        required=True,
        help="原始的 .json 数据集文件路径 (e.g., .../golden_data_original/code15-test.json)"
    )
    parser.add_argument(
        "--converted_output_file", 
        type=Path, 
        required=True,
        help="转换后用于评估的 .jsonl 文件的输出路径"
    )
    
    args = parser.parse_args()

    print(f"--- 开始转换 {args.llm_output_file.name} ---")
    convert_llm_output(
        llm_output_path=args.llm_output_file,
        golden_data_path=args.golden_data_file,
        converted_output_path=args.converted_output_file
    )
    print("----------------------------------\n")

if __name__ == "__main__":
    main()