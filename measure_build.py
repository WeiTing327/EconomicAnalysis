import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
from function_library import * 
import os

def ffill_tensor(tensor_input,method = 1):
    # 確保 tensor_input 是浮點型態，才能處理 NaN
    tensor_input = tensor_input.float()

    mask = torch.isnan(tensor_input)  
    if method == 2:
        for k in range(1, tensor_input.shape[0]):  
            tensor_input[k, :, :] = torch.where(mask[k, :, :], tensor_input[k - 1, :, ], tensor_input[k, :, :])
    elif method ==1:

        for k in range(1, tensor_input.shape[0]):  
            tensor_input[:, :, k] = torch.where(mask[:, :, k], tensor_input[:, :, k-1], tensor_input[:, :, k])

    if 1: # 把未來資料變成nan 這樣避免以後的問題 # 做成下三角矩陣
        i_indices = torch.arange(tensor_input.shape[0]).view(-1, 1, 1)  # 形狀為 [i, 1, 1]
        k_indices = torch.arange(tensor_input.shape[2]).view(1, 1, -1)  # 形狀為 [1, 1, k]
        mask = k_indices > i_indices  # 形狀為 [i, 1, k] 建立一個遮罩來標記所有 k > i 的位置
        mask = mask.expand(tensor_input.shape[0], tensor_input.shape[1], -1)  # 將遮罩擴展
        # 使用遮罩將符合條件的位置填充為 NaN
        tensor_input[mask] = float('nan')
        # tensor_input[mask] = float(99999)
        
    # tensor_input = tensor_input.permute(2, 1, 0) # 把第一個維度和第三個維度互換 這樣操作比較合乎直覺  # 不互換前面程式都要改
    return tensor_input

def create_tensor(symbol_files, data_path="", save_path=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for symbol_idx, file in enumerate(symbol_files):
        print("-"*50)
        df = pd.read_csv(os.path.join(data_path,file),encoding = 'utf-8-sig')
        df = df.replace(".",0)
        # 將所有錯誤的日期轉換為Nan
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['realtime_start'] = pd.to_datetime(df['realtime_start'], errors='coerce')
        df['realtime_end'] = pd.to_datetime(df['realtime_end'], errors='coerce') # 全部都是9999-12-31會print錯誤但不影響
        today = pd.Timestamp(datetime.now().date())  # 取得今天的日期
        
        
        
        # 將 Nan 的值替換為今天的日期
        df['date'].fillna(today, inplace=True)
        df['realtime_start'].fillna(today, inplace=True)
        df['realtime_end'].fillna(today, inplace=True)

        # 排序資料
        df = df.sort_values(by=['date', 'realtime_start', 'realtime_end'])
        max_date = max([df['date'].max(), df['realtime_start'].max(), df['realtime_end'].max()])
        min_date = min([df['date'].min(), df['realtime_start'].min(), df['realtime_end'].min()])
        # 建立日期範圍
        date_list = pd.date_range(start=min_date, end=max_date)
        print(f"max_date:{max_date} | min_date:{min_date} |  df.shape {df.shape} | file {file}")
        is_daily = False
        if df['realtime_start'].equals(df['realtime_end']): # 如果是日資料
            if len(set(df['date'])) == (len(df['date'])):
                is_daily = True
            else:
                print(f"要確認 !!! symbol {file} 是不是日資料")
        if not is_daily and (df['realtime_start'][0] - df['date'][0] ).days > 500 : # 很晚才新增資料 才會有這麼大的差距 就採用自動修正
        # if (df['realtime_start'][0] - df['date'][0] ).days > 500 : # 很晚才新增資料 才會有這麼大的差距 就採用自動修正
            print(f"進入自動新增資料 symbol {file}")
            print(df.head(5))
            if 0:# 最簡單 但是在非日資料不合理
                # T10YFF
                df['realtime_start'] = df['date']
            if 1: # 修正非日資料但是又很晚(ex .2024) 才加入資料庫 ex(CPILFESL)
                check = df.groupby('date').first()
                check = check.reset_index()
                check['diff'] = check['realtime_start'] - check['date']
                
                adj_realtime_start_date = check.iloc[-1]['diff'].days # 用最後一筆判斷資料會遞延多久
                # 直接使用布林條件篩選符合條件的行
                condition = (check['realtime_start'] - check['date']).dt.days < - adj_realtime_start_date
                print(f"symbol {file} 在資料日後{adj_realtime_start_date}公布資料")
                # 新增的資料計算
                new_realtime_start = check.loc[condition, 'date'] + timedelta(days=adj_realtime_start_date)
                new_realtime_end = new_realtime_start - timedelta(days=1)
                
                new_realtime_start = new_realtime_start.apply(lambda x: min(x, max_date)) #確定不超過today
                new_realtime_end = new_realtime_end.apply(lambda x: min(x, max_date)) #確定不超過today

                new_realtime_value = check.loc[condition, 'value']
                # 建立新的 DataFrame 並標記為 '自動新增'
                new_rows = pd.DataFrame({
                    'realtime_start': new_realtime_start,
                    'realtime_end': new_realtime_end,
                    'date': new_realtime_end,
                    '自動新增': True,
                    'value': new_realtime_value
                })
                # 合併新資料
                print(new_rows)
                df = pd.concat([df, new_rows], ignore_index=True)

                # 填補 '新自動新增增' 欄位中的 NaN
                df['自動新增'] = df['自動新增'].fillna(False)
        else:
            df['自動新增'] = False

        if 1:
            tensor = torch.full((len(date_list), 1, len(date_list)), np.nan)
            date_tensor = torch.full((len(date_list), 1 , len(date_list)), np.nan, dtype=torch.float64)
        # print(f"開始find_position | symbol_idx {symbol_idx} | file {file} |time {datetime.now()}")
        # 定義一個內部函數來查找位置
        def find_position(date):
            return date_list.get_loc(date)  # 使用 get_loc 方法查找索引
        
        # 為每個日期列創建新的位置列
        date_columns = ['realtime_start', 'realtime_end', 'date']
        for col in date_columns:
            df[f'{col}_position'] = df[col].apply(find_position)
            # 將資料填入 tensor 中
        # print(f"開始for loop | symbol_idx {symbol_idx} | file {file} |time {datetime.now()}")
        for _, row in df.iterrows():
            # print(is_daily)
            if is_daily:
                start_pos = row['date_position'] # 日資料
            else:
                start_pos = row['realtime_start_position']
                # start_pos = row['realtime_end_position']
            # start_pos = row['realtime_start_position']
            end_pos = row['realtime_end_position']
            date_pos = row['date_position']
            value = (row['value'])
            datadate = row['date']
            value = float(value)
        
            try:
                value = float(value)
            except ValueError:
                value = np.nan  # 使用 NaN 來替代無效值
            if 1:
                # 將指定範圍內的值填入 tensor
                tensor[start_pos:(end_pos+1), 0, date_pos] = value # 填值
                date_tensor[start_pos:(end_pos+1), 0, date_pos] = int(datadate.strftime('%Y%m%d')) *  0.0001 # 轉八碼填載小數位
                fillmethod = 1
            if 0:
                # 將指定範圍內的值填入 tensor
                tensor[date_pos, 0, start_pos:(end_pos+1)] = value # 填值
                date_tensor[date_pos, 0, start_pos:(end_pos+1)] = int(datadate.strftime('%Y%m%d')) *  0.0001 # 轉八碼填載小數位
                fillmethod = 2
                
        # print(f"開始filled_tensor | symbol_idx {symbol_idx} | file {file} |time {datetime.now()}")     
        filled_tensor = ffill_tensor(tensor.to(device),method = fillmethod).to('cpu') # [:,0,:] 會是一個下三角的矩陣 但不會是從第一筆開始
        # print(f"開始filled_tensor_date | symbol_idx {symbol_idx} | file {file} |time {datetime.now()}")     
        filled_tensor_date = ffill_tensor(date_tensor.to(device),method = fillmethod).to('cpu') # [:,0,:] 會是一個下三角的矩陣 但不會是從第一筆開始
        # print(f"結束filled_tensor_date | symbol_idx {symbol_idx} | file {file} |time {datetime.now()}")     
        # 檢查用 耗時
        # torch.equal(torch.nan_to_num(ffill_tensor(tensor,method = fillmethod),0.00), torch.nan_to_num(ffill_tensor(tensor,method = 3),0.00))
        
        # save file
        os.makedirs(os.path.join(save_path,file.split(".")[0]), exist_ok=True)
        save_tensor(filled_tensor,os.path.join(save_path,file.split(".")[0],file.split(".")[0]+ '.pt'),method='hdf5')
        save_tensor(filled_tensor_date,os.path.join(save_path,file.split(".")[0],file.split(".")[0]+ '_date.pt'),method='hdf5')
        save_pickle(date_list,os.path.join(save_path,file.split(".")[0],file.split(".")[0]+ '_date_list.pkl'))
        pd.DataFrame(date_list, columns=["date"]).to_csv(os.path.join(save_path, file.split(".")[0], file.split(".")[0] + '_date_list.csv'),index=False, encoding='utf-8-sig')

        # pd.DataFrame(date_list, columns=["Date"]).to_csv(os.path.join(save_path, file.split(".")[0] + '_date_list.csv'),index=False,encoding='utf-8-sig')

        if 0:
            df_sub = pd.DataFrame(filled_tensor[:, 0, :])
            # 設定 DataFrame 的索引和列名為 date_list
            df_sub.index = date_list  # 設置 index
            df_sub.columns = date_list  # 設置 columns
            output_path = os.path.join(save_path, file.split(".")[0], file.split(".")[0] + '.csv')
            df_sub.to_csv(output_path,encoding='utf-8-sig')
        if 1:
            output_path = os.path.join(save_path, file.split(".")[0], file.split(".")[0] + '_raw_data_自動修正後.csv')
            df.to_csv(output_path,encoding='utf-8-sig')
            
if __name__ == '__main__':
    # 使用示例
    
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'raw_data')
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'measure')
    symbol_files_list = ['GDPC1.csv','T10YFF.csv','EFFR.csv','CPILFESL.csv','UNRATE.csv','UMCSENT.csv','MORTGAGE30US.csv','PAYEMS.csv','PPIACO.csv','CPILFESL.csv','WM2NS.csv','INDPRO.csv','DGS10.csv','OBFRVOL.csv']
    # symbol_files_list = [symbol_files_list [-2]]
    # symbol_files_list = ['UNRATE.csv']
    # UNRATE PAYEMS需另外確認
    if 1: # 製作GDPC1.pt
        create_tensor(symbol_files_list,data_path = data_path,save_path = save_path)
        # df = pd.DataFrame(data_tensor[:,0,:]) # 看這個比較好理解資料的樣子 第二個維度是因為只有一個symbol

