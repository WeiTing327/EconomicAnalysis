# 注意事項:
1. 經濟數字會更改過去數字 參考GDPC1 (FRED)
2. fred的資料來源會有兩個get_series_df 、 get_series_df ()
3. 找出單一個代號的資料
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

# 執行步驟

## 階段一
1. 取得FRED的API然後取代.env的API
2. 執行get_data.py -> 目的 : 從FRED抓資料
3. 執行measure_build.py -> 目的 : 建立measure.pt
4. 執行rule_build.py -> 目的 : 建立rule.pt
5. 執行analysis.py -> 目的 : 分析結果 產生bigtable.csv


## 階段二
1. 把rule_library_username.py的檔名名中的username改成你的名子
2. 參考rule_build.py新增rule