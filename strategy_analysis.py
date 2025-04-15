import os
import pandas as pd
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from function_library import load_tensor, load_pickle
import rule_library_username as rule_library


class RuleEvaluator:
    def __init__(
        self,
        file_path,
        combined_rule_info,
        output_folder_name,
        rule_index,
        lonshort,
        device="cpu",
    ):
        self.df = pd.read_csv(file_path, encoding="utf-8-sig")
        self.output_folder_name = output_folder_name
        self.rule_index = rule_index
        self.lonshort = lonshort
        self.device = device  # 設置運算設備 (cpu/cuda)
        self.evaluate_combined_rule(self.df, combined_rule_info)

    def evaluate_combined_rule(self, df, combined_rule_info):
        rule_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rule")
        measure_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "measure"
        )

        # 從結構中提取數據
        rule_use = combined_rule_info["rule_use"]
        rule_weight = combined_rule_info["rule_weight"]
        rule_describe = combined_rule_info["rule_describe"]

        print(f"分析合併規則 {self.rule_index}: {rule_use}")

        # 讀取並合併所有的規則數據
        obj_rule = rule_library.Rule_Library()
        tensors = []
        max_length = df.shape[0]

        for rule, weight in zip(rule_use, rule_weight):
            rule_method = getattr(obj_rule, rule, None)
            if rule_method:
                rule_method()  # 如果需要參數，可以從 rule 中提取

                tensor = load_tensor(
                    os.path.join(rule_path, rule + ".pt"), method="hdf5"
                )

                # 將張量轉換為 PyTorch 張量，並移動到指定設備
                tensor = torch.tensor(tensor, device=self.device, dtype=torch.float32)

                # 調整張量形狀
                if len(tensor.shape) == 1:
                    tensor = tensor.unsqueeze(1)  # 將一維張量擴展為二維

                # 對齊張量大小
                if tensor.shape[0] < -max_length:
                    padding = torch.zeros(
                        (max_length - tensor.shape[0], tensor.shape[1]),
                        device=self.device,
                    )
                    tensor = torch.cat([tensor, padding], dim=0)
                elif tensor.shape[0] > max_length:
                    tensor = tensor[:max_length]

                tensor = tensor * weight  # 使用 weight 調整張量
                tensors.append(tensor)
                print(f"{rule} tensor shape: {tensor.shape}")

        # 使用 AND 條件來合併張量（基於 PyTorch）
        combined_tensor = torch.stack(tensors, dim=0).sum(axis=1)
        rule_condition = (combined_tensor > (0.3)) * 1  # 超過 0.8 是 1 分
        rule_condition = rule_condition.flatten()

        # 修正錯誤原因：確認 `combined_tensor` 為一維陣列
        combined_tensor = combined_tensor.cpu().numpy().flatten()  # 確保結果為一維陣列

        # 將結果存入 DataFrame
        tensor_df = pd.DataFrame({f"combined_rule_{self.rule_index}": combined_tensor})

        # 準備數據
        measure_name = obj_rule.cross_section_data_file_name[0]
        date_list = load_pickle(
            os.path.join(measure_path, measure_name, measure_name + "_date_list.pkl")
        )

        df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d")
        date_df = pd.DataFrame({"Date": date_list})
        merged_df = pd.merge(date_df, df, on="Date", how="left")
        merged_df[["波段低點區間", "波段高點區間"]] = merged_df[
            ["波段低點區間", "波段高點區間"]
        ].fillna(0)

        # 將 tensor_df 合併到 merged_df
        merged_df = pd.concat([merged_df, tensor_df], axis=1)

        # 設置 y_pred 和 y_true
        y_pred = merged_df[f"combined_rule_{self.rule_index}"] > 0
        y_true = (
            merged_df["波段低點區間"]
            if self.lonshort == "long"
            else merged_df["波段高點區間"]
        )

        # 計算評估指標
        results = {
            "measure": measure_name,
            "combined_rule": rule_use,
            "rule_descriptions": rule_describe,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        # 創建對應的輸出資料夾，保持編號一致
        output_folder_path = os.path.join(
            self.output_folder_name, f"output_{self.lonshort}_{self.rule_index}"
        )
        os.makedirs(output_folder_path, exist_ok=True)

        # 將結果保存為 .json 檔案
        json_path = os.path.join(
            output_folder_path, f"output_{self.lonshort}_{self.rule_index}.json"
        )
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)

        # 將結果保存為 .csv 檔案
        csv_path = os.path.join(
            output_folder_path, f"output_{self.lonshort}_{self.rule_index}.csv"
        )
        pd.DataFrame([results]).to_csv(csv_path, encoding="utf-8-sig", index=False)

        print(f"結果已保存至: {output_folder_path}")


# 主程式部分
def main():
    file_path = "拐點標註檔.csv"
    output_folder = "output"

    combine_rule_path = "combine_rule"

    for prefix in ["combine_rule_long_", "combine_rule_short_"]:
        lonshort = "long" if "long" in prefix else "short"

        subfolders = [
            folder
            for folder in os.listdir(combine_rule_path)
            if folder.startswith(prefix)
        ]

        for folder_name in subfolders:
            folder_path = os.path.join(combine_rule_path, folder_name)

            if os.path.exists(folder_path):
                try:
                    rule_index = int(folder_name.split("_")[-1])
                except ValueError:
                    print(f"錯誤: 無法從 {folder_name} 提取編號")
                    continue

                json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
                if json_files:
                    json_path = os.path.join(folder_path, json_files[0])
                    with open(json_path, "r", encoding="utf-8") as file:
                        combined_rule_info = json.load(file)

                    if not isinstance(combined_rule_info, dict) or not all(
                        key in combined_rule_info
                        for key in ["rule_use", "rule_weight", "rule_describe"]
                    ):
                        print(f"錯誤: {json_path} 中的規則格式不正確，請確認格式。")
                        continue

                    print(f"處理 {folder_name}/{json_files[0]} 規則組合")
                    RuleEvaluator(
                        file_path=file_path,
                        combined_rule_info=combined_rule_info,
                        output_folder_name=output_folder,
                        rule_index=rule_index,
                        lonshort=lonshort,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )


if __name__ == "__main__":
    main()
