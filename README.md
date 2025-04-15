目的和結果
1. 總經、經濟數據會更改過去數字 參考GDPC1 (FRED)，靜態數字為當時訂下的數值，動態為隨時間調整的數值
2. 以往量化分析以二維方式處理資料，因經濟數據有會調整的特性，透過三維方法找出市場做多、做空訊號策略
3. 策略實作內容有以下兩者，一、單一策略市場預測，二、完全隨機組合多策略市場預測
4. fred的資料來源會有兩個get_series_df 、 get_series_df ()
5. 找出單一個代號的資料
   ```
    import torch
    from function_library import load_tensor,load_pickle
    df = pd.DataFrame(load_tensor(os.path.join("measure", "GDPC1", "GDPC1.pt"),method = 'hdf5')[:,0,:])
    date_list = load_pickle(os.path.join("measure","GDPC1","GDPC1_date_list" + ".pkl"))
    df.index = date_list  # 設置 index
    df.columns = date_list  # 設置 columns
    df.to_csv("GDPC1.csv",encoding = 'utf-8-sig')
    ```

---

執行步驟
1. 取得FRED的API然後取代.env的API
2. 執行get_data.py -> 目的 : 從FRED抓資料
3. 執行measure_build.py -> 目的 : 建立measure.pt
4. 執行rule_build.py -> 目的 : 建立rule.pt
5. 執行analysis.py -> 目的 : 分析結果 產生bigtable.csv
6. 把rule_library_username.py的檔名名中的username改成你的名子
7. 參考rule_build.py新增rule# -
