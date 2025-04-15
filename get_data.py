import os
import numpy as np
from full_fred.fred import Fred
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta

load_dotenv()


class Crawl_data:
    def __init__(
        self,
        data=None,
        name=None,
        start_date=None,
        end_date=None,
        change_freq=None,
        FREQ=None,
        days=None,
        columns_name=None,
        observation_start=None,
        realtime_start=None,
        Fred_file=None,
        Fred_path=None,
    ):

        self.name = name
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.change_freq = change_freq
        self.FREQ = FREQ
        self.days = days
        self.columns_name = columns_name
        self.observation_date = observation_start
        self.realtime_start = observation_start
        self.Fred_file = Fred_file
        self.Fred_path = Fred_path
        self.error_data = []

    def get_fred_data(self):

        FRED = Fred(api_key_file=None)  # 先不指定文件路徑 API 從load_dotenv()讀取

        try:
            data_df = FRED.get_series_df(
                self.name,
                observation_start=self.observation_date,
                realtime_start=self.realtime_start,
            )  # , observation_start="2005-01-01", realtime_start="1998-01-01"#問題在這裡

        except Exception as e:
            print(f"第一次抓{self.name}失敗，自動啟動第二次試抓")
            try:
                data_df = FRED.get_series_df(
                    self.name, observation_start=self.observation_date
                )  # 可能是價格資料
                print(f"第二次抓{self.name}成功")
            except Exception as e:

                print(f"Error occurred while fetching data for {self.name}: {e}")
                self.error_data = {
                    "備註": [f"Error occurred while fetching data for {self.name}: {e}"]
                }

                return self.error_data

        return data_df

    def get_error_data(self):
        return self.error_data


if __name__ == "__main__":

    if 0:
        obj = Crawl_data(
            name="GDPC1",
            observation_start="2005-01-01",
            realtime_start="1999-01-01",
            days=1,
        )
        df = obj.get_fred_data()
        print(df)
    if 0:
        obj = Crawl_data(
            name="T10YFF",
            observation_start="2005-01-01",
            realtime_start="1999-01-01",
            days=1,
        )
        df = obj.get_fred_data()
        print(df)

    if 1:
        symbol_list = [
            "GDPC1",
            "T10YFF",
            "CPILFESL",
            "EFFR",
            "UNRATE",
            "UMCSENT",
            "MORTGAGE30US",
            "PAYEMS",
            "PPIACO",
            "CPILFESL",
            "WM2NS",
            "INDPRO",
            "DGS10",
            "OBFRVOL",
        ]
        symbol_list = ["OBFRVOL"]
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")
        for symbol in symbol_list:
            obj = Crawl_data(
                name=symbol,
                observation_start="2005-01-01",
                realtime_start="1999-01-01",
                days=1,
            )
            df = obj.get_fred_data()
            df.to_csv(os.path.join(save_path, f"{symbol}.csv"), encoding="utf-8-sig")
