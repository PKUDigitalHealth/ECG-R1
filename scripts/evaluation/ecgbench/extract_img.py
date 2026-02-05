import pandas as pd
import os

def extract_and_save_images(parquet_file_path: str, output_base_folder: str):
    print(f"正在加载数据: {parquet_file_path}")
    try:
        df = pd.read_parquet(parquet_file_path)
    except Exception as e:
        print(f"加载文件失败: {e}")
        print("请确保已安装 'pandas' 和 'pyarrow'")
        return

    if df.empty:
        print("数据为空，没有图片可以提取。")
        return

    print(f"检测到 {len(df)} 条记录，开始提取并保存图片到 '{output_base_folder}'...")

    for index, row in df.iterrows():
        image_path_in_df = row['image_path']
        full_output_image_path = os.path.join(output_base_folder, image_path_in_df)
        os.makedirs(os.path.dirname(full_output_image_path), exist_ok=True)
        
        try:
            image_bytes = row['image']['bytes']
            with open(full_output_image_path, 'wb') as f:
                f.write(image_bytes) 
        except Exception as e:
            print(f"处理行 {index} 的图片时出错 ({image_path_in_df}): {e}")

    print(f"\n所有图片提取和保存完成到 '{output_base_folder}'。")

if __name__ == "__main__":
    # e.g. g12-test-no-cot/test-00000-of-00001.parquet
    parquet_file = ".parquet file from https://huggingface.co/datasets/PULSE-ECG/ECGBench"
    output_image_base_folder = "Where ECGBench images saved"
    os.makedirs(output_image_base_folder, exist_ok=True)
    extract_and_save_images(parquet_file, output_image_base_folder)

    print("\n--- 验证部分 ---")
    df_check = pd.read_parquet(parquet_file)
    if not df_check.empty:
        first_image_path_in_df = df_check.iloc[0]['image_path']
        expected_image_file = os.path.join(output_image_base_folder, first_image_path_in_df)
        if os.path.exists(expected_image_file):
            print(f"验证: 第一个图片文件 '{expected_image_file}' 已存在。")
        else:
            print(f"验证失败: 第一个图片文件 '{expected_image_file}' 不存在。")
    else:
        print("无法验证，因为数据为空。")
