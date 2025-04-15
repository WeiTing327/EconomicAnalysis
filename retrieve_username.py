import pandas as pd
import os


def load_data(file_path):
    """
    從 CSV 檔案中讀取並選取指定的欄位，並設置 'index' 欄位對應到 'rule' 編號
    :param file_path: CSV 檔案路徑
    :return: 包含選定欄位和更新後 'index' 欄位的 DataFrame
    """
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        selected_columns = [
            "measure",
            "rule",
            "rule_describe",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ]
        sample_data = df[selected_columns]

        # 提取 'rule' 中的編號，並設置到 'index' 欄位
        sample_data["index"] = sample_data["rule"].str.extract(r"(\d+)").astype(int)

        # 將 'index' 欄位放到最前面
        sample_data = sample_data[["index"] + selected_columns]
        return sample_data
    except FileNotFoundError:
        raise Exception(f"檔案 {file_path} 不存在")
    except KeyError as e:
        raise Exception(f"缺少必要欄位: {e}")


def save_data(df, filename):
    """
    儲存 DataFrame 至相同的相對路徑
    :param df: 要儲存的 DataFrame
    :param filename: 輸出的 CSV 檔案名稱
    """
    # 設定檔案儲存的相對路徑
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"{filename} 已儲存至 {save_path}")


if __name__ == "__main__":
    # 指定資料來源
    long_file_path = "bigtable_long.csv"
    short_file_path = "bigtable_short.csv"

    # 提取長資料並儲存
    retrieve_long = load_data(long_file_path)
    save_data(retrieve_long, "retrieve_long.csv")

    # 提取短資料並儲存
    retrieve_short = load_data(short_file_path)
    save_data(retrieve_short, "retrieve_short.csv")
