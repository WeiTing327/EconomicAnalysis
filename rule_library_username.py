class Rule_Library:
    def __init__(self): ...

    def get_cross_section_data(self):
        # 如果變數存在
        if hasattr(self, "cross_section_data"):
            return self.cross_section_data
        else:
            return []

    def get_rule_dataframe(self):
        return self.df

    def get_rule_list(self):
        return [
            method
            for method in dir(eval(type(self).__qualname__))
            if not method.startswith("__")
        ]

    def get_use_data(self):
        return self.use_data

    def get_rule_command(self):
        return self.rule_command

    def get_process_variable(self):
        return self.process_variable

    def get_use_data_by_cross_section_data(self):
        return self.use_data

    def get_cross_section_data_file_name(self):
        return self.cross_section_data_file_name

    def get_rule_description(self):
        return self.description

    # torch.diagonal(filled_tensor, dim1=0, dim2=2).T -> 當天看到的最新的GDP(過了就過了)
    """ rule dmoe 

    """

    def rule_1(self):
        # 描述
        self.description = "靜態美國GDP原始數字比250天前看到的減少"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "diff_record_daily_value_change(GDPC1,windows = 250) < -  0 "
        )

    def rule_2(self):
        # 描述
        self.description = "靜態美國GDP原始數字比250天前看到的增加"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "diff_record_daily_value_change(GDPC1,windows = 250) > 0  "
        )

    def rule_3(self):
        # 描述
        self.description = "動態美國GDP原始數字比250天前看到的減少"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "diff_track_adjusted_values_change(GDPC1,windows = 60) < - 0 "
        )

    def rule_4(self):
        # 描述
        self.description = "動態美國GDP原始數字比250天前看到的增加"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "diff_track_adjusted_values_change(GDPC1,windows = 250) > 0  "
        )

    def rule_5(self):
        # 描述
        self.description = "動態美國GDP原始數字比250天前低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "percent_change_track_adjusted_values_change(GDPC1,windows = 250) < - 0.01 "
        )

    def rule_6(self):
        # 描述
        self.description = "動態美國GDP原始數字比250天前低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "percent_change_track_adjusted_values_change(GDPC1,windows = 250) < - 0.05 "
        )

    def rule_7(self):
        # 描述
        self.description = "動態美國GDP原始數字比250天前低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "percent_change_track_adjusted_values_change(GDPC1,windows = 250) < - 0.1 "
        )

    def rule_8(self):
        # 描述
        self.description = "動態美國GDP原始數字比250天前增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "percent_change_track_adjusted_values_change(GDPC1,windows = 250) > 0.01 "
        )

    def rule_9(self):
        # 描述
        self.description = "動態美國GDP原始數字比250天前增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "percent_change_track_adjusted_values_change(GDPC1,windows = 250) > 0.05 "
        )

    def rule_10(self):
        # 描述
        self.description = "動態美國GDP原始數字比250天前增加10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "percent_change_track_adjusted_values_change(GDPC1,windows = 250) > 0.1 "
        )

    def rule_11(self):
        # 描述
        self.description = "靜態美國GDP原始數字比250天前低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "percent_change_record_daily_value_change(GDPC1,windows = 250) < - 0.01 "
        )

    def rule_12(self):
        # 描述
        self.description = "靜態美國GDP原始數字比250天前低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "percent_change_record_daily_value_change(GDPC1,windows = 250) < - 0.05 "
        )

    def rule_13(self):
        # 描述
        self.description = "靜態美國GDP原始數字比250天前低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "percent_change_record_daily_value_change(GDPC1,windows = 250) < - 0.1 "
        )

    def rule_14(self):
        # 描述
        self.description = "靜態美國GDP原始數字比250天前增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "percent_change_record_daily_value_change(GDPC1,windows = 250) > 0.01 "
        )

    def rule_15(self):
        # 描述
        self.description = "靜態美國GDP原始數字比250天前增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "percent_change_record_daily_value_change(GDPC1,windows = 250) > 0.05 "
        )

    def rule_16(self):
        # 描述
        self.description = "靜態美國GDP原始數字比250天前增加10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數
        self.process_variable = (
            "percent_change_record_daily_value_change(GDPC1,windows = 250) > 0.1 "
        )

    def rule_17(self):
        # 描述
        self.description = "靜態美國GDP原始數字MOM成長1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數：利用函數計算MOM增長
        self.process_variable = (
            "percent_change_record_daily_value_change(GDPC1,windows = 30) > 0.01"
        )

    def rule_18(self):
        # 描述
        self.description = "動態美國GDP原始數字MOM成長1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數：利用函數計算MOM增長
        self.process_variable = (
            "percent_change_track_adjusted_values_change(GDPC1,windows = 30) > 0.01"
        )

    def rule_19(self):
        # 描述
        self.description = "動態美國GDP原始數字MOM成長5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數：利用函數計算MOM增長
        self.process_variable = (
            "percent_change_track_adjusted_values_change(GDPC1,windows = 30) > 0.05"
        )

    def rule_20(self):
        # 描述
        self.description = "動態美國GDP原始數字MOM成長10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數：利用函數計算MOM增長
        self.process_variable = (
            "percent_change_track_adjusted_values_change(GDPC1,windows = 30) > 0.10"
        )

    def rule_21(self):
        # 描述
        self.description = "動態美國GDP原始數字MOM低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數：利用函數計算MOM增長
        self.process_variable = (
            "percent_change_track_adjusted_values_change(GDPC1,windows = 30) < - 0.01"
        )

    def rule_22(self):
        # 描述
        self.description = "動態美國GDP原始數字MOM低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數：利用函數計算MOM增長
        self.process_variable = (
            "percent_change_track_adjusted_values_change(GDPC1,windows = 30) < - 0.05"
        )

    def rule_23(self):
        # 描述
        self.description = "動態美國GDP原始數字MOM低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1"]

        # 處理變數：利用函數計算MOM增長
        self.process_variable = (
            "percent_change_track_adjusted_values_change(GDPC1,windows = 30) < - 0.1"
        )

    def rule_24(self):
        # 描述
        self.description = "靜態美國失業率比250天前增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UNRATE"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UNRATE"]

        # 處理變數：靜態失業率250天內的變化是否增加1%
        self.process_variable = (
            "percent_change_record_daily_value_change(UNRATE,windows = 250) > 0.01"
        )

    def rule_25(self):
        # 描述
        self.description = "靜態美國失業率比250天前增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UNRATE"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UNRATE"]

        # 處理變數：靜態失業率250天內的變化是否增加5%
        self.process_variable = (
            "percent_change_record_daily_value_change(UNRATE,windows = 250) > 0.5"
        )

    def rule_26(self):
        # 描述
        self.description = "靜態美國失業率比250天前增加10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UNRATE"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UNRATE"]

        # 處理變數：靜態失業率250天內的變化是否增加10%
        self.process_variable = (
            "percent_change_record_daily_value_change(UNRATE,windows = 250) > 0.1"
        )

    def rule_27(self):
        # 描述
        self.description = "靜態美國失業率比250天前低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UNRATE"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UNRATE"]

        # 處理變數：靜態失業率250天內的變化是否低於1%
        self.process_variable = (
            "percent_change_record_daily_value_change(UNRATE,windows = 250) < - 0.01"
        )

    def rule_28(self):
        # 描述
        self.description = "靜態美國失業率比250天前低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UNRATE"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UNRATE"]

        # 處理變數：靜態失業率250天內的變化是否低於5%
        self.process_variable = (
            "percent_change_record_daily_value_change(UNRATE,windows = 250) < - 0.05"
        )

    def rule_29(self):
        # 描述
        self.description = "靜態美國失業率比250天前低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UNRATE"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UNRATE"]

        # 處理變數：靜態失業率250天內的變化是否低於10%
        self.process_variable = (
            "percent_change_record_daily_value_change(UNRATE,windows = 250) < - 0.1"
        )

    def rule_30(self):
        # 描述
        self.description = "動態美國失業率比250天前增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UNRATE"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UNRATE"]

        # 處理變數：動態失業率250天內的變化是否增加1%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(UNRATE,windows = 250) > 0.01"
        )

    def rule_31(self):
        # 描述
        self.description = "動態美國失業率比250天前增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UNRATE"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UNRATE"]

        # 處理變數：動態失業率250天內的變化是否增加5%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(UNRATE,windows = 250) > 0.1"
        )

    def rule_32(self):
        # 描述
        self.description = "動態美國失業率比250天前增加10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UNRATE"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UNRATE"]

        # 處理變數：動態失業率250天內的變化是否增加10%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(UNRATE,windows = 250) > 0.1"
        )

    def rule_33(self):
        # 描述
        self.description = "動態美國失業率比250天前低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UNRATE"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UNRATE"]

        # 處理變數：動態失業率250天內的變化是否低於1%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(UNRATE,windows = 250) < - 0.01"
        )

    def rule_34(self):
        # 描述
        self.description = "動態美國失業率比250天前低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UNRATE"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UNRATE"]

        # 處理變數：動態失業率250天內的變化是否低於5%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(UNRATE,windows = 250) < - 0.5"
        )

    def rule_35(self):
        # 描述
        self.description = "動態美國失業率比250天前低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UNRATE"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UNRATE"]

        # 處理變數：動態失業率250天內的變化是否低於10%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(UNRATE,windows = 250) < - 0.1"
        )

    def rule_36(self):
        # 描述
        self.description = "靜態核心CPI比120天前增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["CPILFESL"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["CPILFESL"]

        # 處理變數：靜態核心CPI在120天內是否增長超過1%
        self.process_variable = (
            "percent_change_record_daily_value_change(CPILFESL,windows = 120) > 0.01"
        )

    def rule_37(self):
        # 描述
        self.description = "靜態核心CPI比120天前增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["CPILFESL"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["CPILFESL"]

        # 處理變數：靜態核心CPI在120天內是否增長超過5%
        self.process_variable = (
            "percent_change_record_daily_value_change(CPILFESL,windows = 120) > 0.05"
        )

    def rule_38(self):
        # 描述
        self.description = "靜態核心CPI比120天前增加10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["CPILFESL"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["CPILFESL"]

        # 處理變數：靜態核心CPI在120天內是否增長超過10%
        self.process_variable = (
            "percent_change_record_daily_value_change(CPILFESL,windows = 120) > 0.1"
        )

    def rule_39(self):
        # 描述
        self.description = "靜態核心CPI比120天前低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["CPILFESL"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["CPILFESL"]

        # 處理變數：靜態核心CPI在120天內是否低於超過1%
        self.process_variable = (
            "percent_change_record_daily_value_change(CPILFESL,windows = 120) < - 0.01"
        )

    def rule_40(self):
        # 描述
        self.description = "靜態核心CPI比120天前低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["CPILFESL"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["CPILFESL"]

        # 處理變數：靜態核心CPI在120天內是否低於超過5%
        self.process_variable = (
            "percent_change_record_daily_value_change(CPILFESL,windows = 120) < - 0.5"
        )

    def rule_41(self):
        # 描述
        self.description = "靜態核心CPI比120天前低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["CPILFESL"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["CPILFESL"]

        # 處理變數：靜態核心CPI在120天內是否低於超過10%
        self.process_variable = (
            "percent_change_record_daily_value_change(CPILFESL,windows = 120) < - 0.1"
        )

    def rule_42(self):
        # 描述
        self.description = "動態核心CPI比120天前增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["CPILFESL"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["CPILFESL"]

        # 處理變數：動態核心CPI在120天內是否增長超過1%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(CPILFESL,windows = 120) > 0.01"
        )

    def rule_43(self):
        # 描述
        self.description = "動態核心CPI比120天前增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["CPILFESL"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["CPILFESL"]

        # 處理變數：動態核心CPI在120天內是否增長超過5%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(CPILFESL,windows = 120) > 0.05"
        )

    def rule_44(self):
        # 描述
        self.description = "動態核心CPI比120天前增加10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["CPILFESL"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["CPILFESL"]

        # 處理變數：動態核心CPI在120天內是否增長超過10%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(CPILFESL,windows = 120) > 0.1"
        )

    def rule_45(self):
        # 描述
        self.description = "動態核心CPI比120天前低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["CPILFESL"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["CPILFESL"]

        # 處理變數：動態核心CPI在120天內是否低於超過1%
        self.process_variable = "percent_change_track_adjusted_values_change(CPILFESL,windows = 120) < - 0.01"

    def rule_46(self):
        # 描述
        self.description = "動態核心CPI比120天前低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["CPILFESL"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["CPILFESL"]

        # 處理變數：動態核心CPI在120天內是否低於超過5%
        self.process_variable = "percent_change_track_adjusted_values_change(CPILFESL,windows = 120) < - 0.5"

    def rule_47(self):
        # 描述
        self.description = "動態核心CPI比120天前低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["CPILFESL"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["CPILFESL"]

        # 處理變數：動態核心CPI在120天內是否低於超過10%
        self.process_variable = "percent_change_track_adjusted_values_change(CPILFESL,windows = 120) < - 0.1"

    def rule_48(self):
        # 描述
        self.description = (
            "動態美國GDP原始數字比250天前增加1%且動態美國失業率比250天前低於1%"
        )

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["GDPC1", "UNRATE"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["GDPC1", "UNRATE"]

        # 處理變數：動態 GDP 比 250 天前增加 1%，且動態失業率比 250 天前低於 1%
        gdp_condition = (
            "percent_change_track_adjusted_values_change(GDPC1, windows=250) > 0.01"
        )
        unemployment_condition = (
            "percent_change_track_adjusted_values_change(UNRATE, windows=250) < - 0.01"
        )

        # 使用逐元素邏輯運算符來結合條件
        self.process_variable = f"({gdp_condition}) & ({unemployment_condition})"

    def rule_49(self):
        # 描述
        self.description = "靜態生產者物價指數比250天前增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PPIACO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PPIACO"]

        # 處理變數：靜態工業生產數據比250天前增加1%
        self.process_variable = (
            "percent_change_record_daily_value_change(PPIACO,windows = 250) > 0.01"
        )

    def rule_50(self):
        # 描述
        self.description = "靜態生產者物價指數比250天前增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PPIACO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PPIACO"]

        # 處理變數：靜態工業生產數據比250天前增加5%
        self.process_variable = (
            "percent_change_record_daily_value_change(PPIACO,windows = 250) > 0.05"
        )

    def rule_51(self):
        # 描述
        self.description = "靜態生產者物價指數比250天前增加10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PPIACO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PPIACO"]

        # 處理變數：靜態工業生產數據比250天前增加10%
        self.process_variable = (
            "percent_change_record_daily_value_change(PPIACO,windows = 250) > 0.1"
        )

    def rule_52(self):
        # 描述
        self.description = "靜態生產者物價指數比250天前低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PPIACO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PPIACO"]

        # 處理變數：靜態工業生產數據比250天前低於1%
        self.process_variable = (
            "percent_change_record_daily_value_change(PPIACO,windows = 250) < - 0.01"
        )

    def rule_53(self):
        # 描述
        self.description = "靜態生產者物價指數比250天前低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PPIACO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PPIACO"]

        # 處理變數：靜態工業生產數據比250天前低於5%
        self.process_variable = (
            "percent_change_record_daily_value_change(PPIACO,windows = 250) < - 0.05"
        )

    def rule_54(self):
        # 描述
        self.description = "靜態生產者物價指數比250天前低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PPIACO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PPIACO"]

        # 處理變數：靜態工業生產數據比250天前低於10%
        self.process_variable = (
            "percent_change_record_daily_value_change(PPIACO,windows = 250) < - 0.1"
        )

    def rule_55(self):
        # 描述
        self.description = "動態生產者物價指數比250天前增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PPIACO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PPIACO"]

        # 處理變數：工業生產數據250天前增加1%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(PPIACO,windows = 250) > 0.01"
        )

    def rule_56(self):
        # 描述
        self.description = "動態生產者物價指數比250天前增加2%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PPIACO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PPIACO"]

        # 處理變數：動態工業生產數據250天前增加2%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(PPIACO,windows = 250) > 0.02"
        )

    def rule_57(self):
        # 描述
        self.description = "動態生產者物價指數比250天前增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PPIACO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PPIACO"]

        # 處理變數：動態工業生產數據250天前增加5%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(PPIACO,windows = 250) > 0.05"
        )

    def rule_58(self):
        # 描述
        self.description = "動態生產者物價指數比250天前增加10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PPIACO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PPIACO"]

        # 處理變數：動態工業生產數據250天前增加10%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(PPIACO,windows = 250) > 0.1"
        )

    def rule_59(self):
        # 描述
        self.description = "動態生產者物價指數比250天前低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PPIACO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PPIACO"]

        # 處理變數：工業生產數據250天前低於1%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(PPIACO,windows = 250) < - 0.01"
        )

    def rule_60(self):
        # 描述
        self.description = "動態生產者物價指數比250天前低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PPIACO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PPIACO"]

        # 處理變數：動態工業生產數據250天前低於5%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(PPIACO,windows = 250) < - 0.05"
        )

    def rule_61(self):
        # 描述
        self.description = "動態生產者物價指數比250天前低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PPIACO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PPIACO"]

        # 處理變數：動態工業生產數據250天前低於10%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(PPIACO,windows = 250) < - 0.1"
        )

    def rule_62(self):
        # 描述
        self.description = "靜態美國30年期固定利率抵押貸款平均值在過去30天增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：靜態房屋價格指數30天增加1%
        self.process_variable = (
            "percent_change_record_daily_value_change(MORTGAGE30US, windows=30) > 0.01"
        )

    def rule_63(self):
        # 描述
        self.description = "靜態美國30年期固定利率抵押貸款平均值在過去30天增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：靜態房屋價格指數30天增加5%
        self.process_variable = (
            "percent_change_record_daily_value_change(MORTGAGE30US, windows=30) > 0.05"
        )

    def rule_64(self):
        # 描述
        self.description = "靜態美國30年期固定利率抵押貸款平均值在過去30天增加0%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：靜態房屋價格指數30天增加10%
        self.process_variable = (
            "percent_change_record_daily_value_change(MORTGAGE30US, windows=30) > 0.1"
        )

    def rule_65(self):
        # 描述
        self.description = "靜態美國30年期固定利率抵押貸款平均值在過去250天增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：靜態房屋價格指數250天增加1%
        self.process_variable = (
            "percent_change_record_daily_value_change(MORTGAGE30US, windows=250) > 0.01"
        )

    def rule_66(self):
        # 描述
        self.description = "靜態美國30年期固定利率抵押貸款平均值在過去250天增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：靜態房屋價格指數250天增加5%
        self.process_variable = (
            "percent_change_record_daily_value_change(MORTGAGE30US, windows=250) > 0.05"
        )

    def rule_67(self):
        # 描述
        self.description = "靜態美國30年期固定利率抵押貸款平均值在過去250天增加0%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：靜態房屋價格指數250內增加10%
        self.process_variable = (
            "percent_change_record_daily_value_change(MORTGAGE30US, windows=250) > 0.1"
        )

    def rule_68(self):
        # 描述
        self.description = "靜態美國30年期固定利率抵押貸款平均值在過去30天低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：靜態房屋價格指數30天低於1%
        self.process_variable = "percent_change_record_daily_value_change(MORTGAGE30US, windows=30) < - 0.01"

    def rule_69(self):
        # 描述
        self.description = "靜態美國30年期固定利率抵押貸款平均值在過去30天低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：靜態房屋價格指數30天低於5%
        self.process_variable = "percent_change_record_daily_value_change(MORTGAGE30US, windows=30) < - 0.05"

    def rule_70(self):
        # 描述
        self.description = "靜態美國30年期固定利率抵押貸款平均值在過去30天低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：靜態房屋價格指數30天低於10%
        self.process_variable = (
            "percent_change_record_daily_value_change(MORTGAGE30US, windows=30) < - 0.1"
        )

    def rule_71(self):
        # 描述
        self.description = "靜態美國30年期固定利率抵押貸款平均值在過去250天低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：靜態房屋價格指數250天低於1%
        self.process_variable = "percent_change_record_daily_value_change(MORTGAGE30US, windows=250) < - 0.01"

    def rule_72(self):
        # 描述
        self.description = "靜態美國30年期固定利率抵押貸款平均值在過去250天低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：靜態房屋價格指數250天低於5%
        self.process_variable = "percent_change_record_daily_value_change(MORTGAGE30US, windows=250) < - 0.05"

    def rule_73(self):
        # 描述
        self.description = "靜態美國30年期固定利率抵押貸款平均值在過去250天低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：靜態房屋價格指數250低於10%
        self.process_variable = "percent_change_record_daily_value_change(MORTGAGE30US, windows=250) < - 0.1"

    def rule_74(self):
        # 描述
        self.description = "動態美國30年期固定利率抵押貸款平均值在過去30天增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：動態房屋價格指數30天增加1%
        self.process_variable = "percent_change_track_adjusted_values_change(MORTGAGE30US, windows=30) > 0.01"

    def rule_75(self):
        # 描述
        self.description = "動態美國30年期固定利率抵押貸款平均值在過去30天增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：動態房屋價格指數30天增加5%
        self.process_variable = "percent_change_track_adjusted_values_change(MORTGAGE30US, windows=30) > 0.05"

    def rule_76(self):
        # 描述
        self.description = "動態美國30年期固定利率抵押貸款平均值在過去30天增加0%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：動態房屋價格指數30天增加10%
        self.process_variable = "percent_change_track_adjusted_values_change(MORTGAGE30US, windows=30) > 0.1"

    def rule_77(self):
        # 描述
        self.description = "動態美國30年期固定利率抵押貸款平均值在過去250天增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：動態房屋價格指數250天增加1%
        self.process_variable = "percent_change_track_adjusted_values_change(MORTGAGE30US, windows=250) > 0.01"

    def rule_78(self):
        # 描述
        self.description = "動態美國30年期固定利率抵押貸款平均值在過去250天增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：動態房屋價格指數250天增加5%
        self.process_variable = "percent_change_track_adjusted_values_change(MORTGAGE30US, windows=250) > 0.05"

    def rule_79(self):
        # 描述
        self.description = "動態美國30年期固定利率抵押貸款平均值在過去250天增加0%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：動態房屋價格指數250內增加10%
        self.process_variable = "percent_change_track_adjusted_values_change(MORTGAGE30US, windows=250) > 0.1"

    def rule_80(self):
        # 描述
        self.description = "動態美國30年期固定利率抵押貸款平均值在過去30天低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：動態房屋價格指數30天低於1%
        self.process_variable = "percent_change_track_adjusted_values_change(MORTGAGE30US, windows=30) < - 0.01"

    def rule_81(self):
        # 描述
        self.description = "動態美國30年期固定利率抵押貸款平均值在過去30天低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：動態房屋價格指數30天低於5%
        self.process_variable = "percent_change_track_adjusted_values_change(MORTGAGE30US, windows=30) < - 0.05"

    def rule_82(self):
        # 描述
        self.description = "動態美國30年期固定利率抵押貸款平均值在過去30天低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：動態房屋價格指數30天低於10%
        self.process_variable = "percent_change_track_adjusted_values_change(MORTGAGE30US, windows=30) < - 0.1"

    def rule_83(self):
        # 描述
        self.description = "動態美國30年期固定利率抵押貸款平均值在過去250天低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：動態房屋價格指數250天低於1%
        self.process_variable = "percent_change_track_adjusted_values_change(MORTGAGE30US, windows=250) < - 0.01"

    def rule_84(self):
        # 描述
        self.description = "動態美國30年期固定利率抵押貸款平均值在過去250天低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：動態房屋價格指數250天低於5%
        self.process_variable = "percent_change_track_adjusted_values_change(MORTGAGE30US, windows=250) < - 0.05"

    def rule_85(self):
        # 描述
        self.description = "動態美國30年期固定利率抵押貸款平均值在過去250天低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["MORTGAGE30US"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["MORTGAGE30US"]

        # 處理變數：動態房屋價格指數250低於10%
        self.process_variable = "percent_change_track_adjusted_values_change(MORTGAGE30US, windows=250) < - 0.1"

    def rule_86(self):
        # 描述
        self.description = "靜態美國消費者信心指數在過去250天增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UMCSENT"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UMCSENT"]

        # 處理變數：靜態消費者信心指數250天增加1%
        self.process_variable = (
            "percent_change_record_daily_value_change(UMCSENT, windows=250) > 0.01"
        )

    def rule_87(self):
        # 描述
        self.description = "靜態美國消費者信心指數在過去250天增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UMCSENT"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UMCSENT"]

        # 處理變數：靜態消費者信心指數250天增加5%
        self.process_variable = (
            "percent_change_record_daily_value_change(UMCSENT, windows=250) > 0.05"
        )

    def rule_88(self):
        # 描述
        self.description = "靜態美國消費者信心指數在過去250天增加10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UMCSENT"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UMCSENT"]

        # 處理變數：靜態消費者信心指數250天增加10%
        self.process_variable = (
            "percent_change_record_daily_value_change(UMCSENT, windows=250) > 0.1"
        )

    def rule_89(self):
        # 描述
        self.description = "靜態美國消費者信心指數在過去250天低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UMCSENT"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UMCSENT"]

        # 處理變數：靜態消費者信心指數250天低於1%
        self.process_variable = (
            "percent_change_record_daily_value_change(UMCSENT, windows=250) < - 0.01"
        )

    def rule_90(self):
        # 描述
        self.description = "靜態美國消費者信心指數在過去250天低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UMCSENT"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UMCSENT"]

        # 處理變數：靜態消費者信心指數250天低於5%
        self.process_variable = (
            "percent_change_record_daily_value_change(UMCSENT, windows=250) < - 0.05"
        )

    def rule_91(self):
        # 描述
        self.description = "靜態美國消費者信心指數在過去250天低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UMCSENT"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UMCSENT"]

        # 處理變數：靜態消費者信心指數250天低於10%
        self.process_variable = (
            "percent_change_record_daily_value_change(UMCSENT, windows=250) < - 0.1"
        )

    def rule_92(self):
        # 描述
        self.description = "動態美國消費者信心指數在過去250天增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UMCSENT"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UMCSENT"]

        # 處理變數：動態消費者信心指數250天增加1%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(UMCSENT, windows=250) > 0.01"
        )

    def rule_93(self):
        # 描述
        self.description = "動態美國消費者信心指數在過去250天增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UMCSENT"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UMCSENT"]

        # 處理變數：動態消費者信心指數250天增加5%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(UMCSENT, windows=250) > 0.05"
        )

    def rule_94(self):
        # 描述
        self.description = "動態美國消費者信心指數在過去250天增加10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UMCSENT"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UMCSENT"]

        # 處理變數：動態消費者信心指數250天增加10%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(UMCSENT, windows=250) > 0.1"
        )

    def rule_95(self):
        # 描述
        self.description = "動態美國消費者信心指數在過去250天低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UMCSENT"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UMCSENT"]

        # 處理變數：動態消費者信心指數250天低於1%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(UMCSENT, windows=250) < - 0.01"
        )

    def rule_96(self):
        # 描述
        self.description = "動態美國消費者信心指數在過去250天低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UMCSENT"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UMCSENT"]

        # 處理變數：動態消費者信心指數250天低於5%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(UMCSENT, windows=250) < - 0.05"
        )

    def rule_97(self):
        # 描述
        self.description = "動態美國消費者信心指數在過去250天低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["UMCSENT"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["UMCSENT"]

        # 處理變數：動態消費者信心指數250天低於10%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(UMCSENT, windows=250) < - 0.1"
        )

    def rule_98(self):
        # 描述
        self.description = "靜態美國非農就業數據在過去250天增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數：靜態非農就業數據250天增加1%
        self.process_variable = (
            "percent_change_record_daily_value_change(PAYEMS, windows=250) > 0.01"
        )

    def rule_99(self):
        # 描述
        self.description = "靜態美國非農就業數據在過去250天增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數：靜態非農就業數據250天增加5%
        self.process_variable = (
            "percent_change_record_daily_value_change(PAYEMS, windows=250) > 0.05"
        )

    def rule_100(self):
        # 描述
        self.description = "靜態美國非農就業數據在過去250天增加10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數：靜態非農就業數據250天增加10%
        self.process_variable = (
            "percent_change_record_daily_value_change(PAYEMS, windows=250) > 0.1"
        )

    def rule_101(self):
        # 描述
        self.description = "靜態美國非農就業數據在過去250天低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數：靜態非農就業數據250天低於1%
        self.process_variable = (
            "percent_change_record_daily_value_change(PAYEMS, windows=250) < - 0.01"
        )

    def rule_102(self):
        # 描述
        self.description = "靜態美國非農就業數據在過去250天低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數：靜態非農就業數據250天低於5%
        self.process_variable = (
            "percent_change_record_daily_value_change(PAYEMS, windows=250) < - 0.05"
        )

    def rule_103(self):
        # 描述
        self.description = "靜態美國非農就業數據在過去250天低於10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數：靜態非農就業數據250天低於10%
        self.process_variable = (
            "percent_change_record_daily_value_change(PAYEMS, windows=250) < - 0.1"
        )

    def rule_104(self):
        # 描述
        self.description = "動態美國非農就業數據在過去250天增加1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數：動態非農就業數據250天增加1%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(PAYEMS, windows=250) > 0.01"
        )

    def rule_105(self):
        # 描述
        self.description = "動態美國非農就業數據在過去250天增加5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數：動態非農就業數據250天增加5%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(PAYEMS, windows=250) > 0.05"
        )

    def rule_106(self):
        # 描述
        self.description = "動態美國非農就業數據在過去250天增加10%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數：動態非農就業數據250天增加10%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(PAYEMS, windows=250) > 0.1"
        )

    def rule_107(self):
        # 描述
        self.description = "動態美國非農就業數據在過去250天低於1%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數：動態非農就業數據250天低於1%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(PAYEMS, windows=250) < - 0.01"
        )

    def rule_108(self):
        # 描述
        self.description = "動態美國非農就業數據在過去250天低於5%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數：動態非農就業數據250天低於5%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(PAYEMS, windows=250) < - 0.05"
        )

    def rule_109(self):
        # 描述
        self.description = "動態美國非農就業數據在過去250天低於1000%"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數：動態非農就業數據250天低於1000%
        self.process_variable = (
            "percent_change_track_adjusted_values_change(PAYEMS, windows=250) < - 100"
        )

    def rule_110(self):
        # 描述
        self.description = "靜態美國非農就業數據原始數字比250天前看到的增加"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數
        self.process_variable = (
            "diff_record_daily_value_change(PAYEMS,windows = 250) >  0 "
        )

    def rule_111(self):
        # 描述
        self.description = "靜態美國非農就業數據原始數字比250天前看到的低於"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數
        self.process_variable = (
            "diff_record_daily_value_change(PAYEMS,windows = 250) < -  0 "
        )

    def rule_112(self):
        # 描述
        self.description = "動態美國非農就業數據原始數字比250天前看到的增加"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數
        self.process_variable = (
            "diff_track_adjusted_values_change(PAYEMS,windows = 250) >  0 "
        )

    def rule_113(self):
        # 描述
        self.description = "動態美國非農就業數據原始數字比250天前看到的低於"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["PAYEMS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["PAYEMS"]

        # 處理變數
        self.process_variable = (
            "diff_track_adjusted_values_change(PAYEMS,windows = 250) < -  0 "
        )

    def rule_114(self):
        # 描述
        self.description = "靜態美國M2原始數字比250天前看到的減少"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["WM2NS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["WM2NS"]

        # 處理變數
        self.process_variable = (
            "diff_record_daily_value_change(WM2NS,windows = 250) < -  0 "
        )

    def rule_115(self):
        # 描述
        self.description = "靜態美國M2原始數字比250天前看到的增加"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["WM2NS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["WM2NS"]

        # 處理變數
        self.process_variable = (
            "diff_record_daily_value_change(WM2NS,windows = 250) > 0  "
        )

    def rule_116(self):
        # 描述
        self.description = "動態美國M2的數字比250天前看到的低於"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["WM2NS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["WM2NS"]

        # 處理變數
        self.process_variable = (
            "diff_track_adjusted_values_change(WM2NS,windows = 60) < - 0 "
        )

    def rule_117(self):
        # 描述
        self.description = "動態美國M2的數字比250天前看到的增加"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["WM2NS"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["WM2NS"]

        # 處理變數
        self.process_variable = (
            "diff_track_adjusted_values_change(WM2NS,windows = 250) > 0  "
        )

    def rule_118(self):
        # 描述
        self.description = "靜態美國工業生產指數原始數字比250天前看到的增加"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["INDPRO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["INDPRO"]

        # 處理變數
        self.process_variable = (
            "percent_change_record_daily_value_change(INDPRO,windows = 250) >  0 "
        )

    def rule_119(self):
        # 描述
        self.description = "動態美國工業生產指數原始數字比250天前看到的低於"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["INDPRO"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["INDPRO"]

        # 處理變數
        self.process_variable = (
            "percent_change_track_adjusted_values_change(INDPRO,windows = 250) < -  0 "
        )

    def rule_120(self):
        # 描述
        self.description = (
            "靜態美國10年期固定期限美國公債市場報酬率原始數字比250天前看到的增加"
        )

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["DGS10"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["DGS10"]

        # 處理變數
        self.process_variable = (
            "percent_change_record_daily_value_change(DGS10,windows = 250) >  0 "
        )

    def rule_121(self):
        # 描述
        self.description = (
            "動態10年期固定期限美國公債市場報酬率原始數字比250天前看到的低於"
        )

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["DGS10"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["DGS10"]

        # 處理變數
        self.process_variable = (
            "percent_change_track_adjusted_values_change(DGS10,windows = 250) <   0 "
        )

    def rule_122(self):
        # 描述
        self.description = "靜態美國銀行隔夜融資量原始數字比250天前看到的增加"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["OBFRVOL"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["OBFRVOL"]

        # 處理變數
        self.process_variable = (
            "diff_record_daily_value_change(OBFRVOL,windows = 250) >  0 "
        )

    def rule_123(self):
        # 描述
        self.description = "動態美國銀行隔夜融資量原始數字比250天前看到的低於"

        # 資料庫的欄位名稱
        self.use_data = [""]

        # measure檔案名稱
        self.cross_section_data_file_name = ["OBFRVOL"]

        # measure檔案名稱對應的變數名稱
        self.cross_section_data = ["OBFRVOL"]

        # 處理變數
        self.process_variable = (
            "diff_track_adjusted_values_change(OBFRVOL,windows = 250) < -  0 "
        )
