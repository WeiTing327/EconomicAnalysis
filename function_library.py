import gzip
import h5py
import torch
import pickle

def place_data_on_three_dim_main_diagonal(two_dim_tensor):
    """
    將一個二維張量的數據放置在三維張量的主對角線上，其餘位置為 0。

    參數:
    two_dim_tensor (torch.Tensor): 形狀為 [N, M] 的二維張量。

    返回:
    torch.Tensor: 形狀為 [N, M, N] 的三維張量，主對角線為原始張量的數據。
    """
    # 獲取原始張量的形狀
    n, m = two_dim_tensor.shape

    # 初始化一個新的三維張量，其中非對角線元素為 0
    three_dim_tensor = torch.zeros((n, m, n))

    # 使用 PyTorch 的 advanced indexing 來填充主對角線
    indices = torch.arange(n)
    three_dim_tensor[indices, :, indices] = two_dim_tensor

    return three_dim_tensor

def signal_count_check_input_command(operationn, **variables):
    ''' 輸入rule指令 '''
    result = rule_calculate(operationn, **variables)
    count = (torch.nan_to_num(result, nan=0.0) !=0).sum().item()
    print(f"訊號出現 {count} 次 總共用 {result.shape[0]} 天 | 比例為 {(count / result.shape[0] * 100):.2f}%")

def signal_count_check_input_result(result, **variables):
    ''' 輸入結果 '''
    count = (torch.nan_to_num(result, nan=0.0) !=0).sum().item()
    print(f"訊號出現 {count} 次 總共用 {result.shape[0]} 天 | 比例為 {(count / result.shape[0] * 100):.2f}%")
    

def diff_track_adjusted_values_change(tensor,windows = 60):
    ''' 每天看到所有最新數值(調整後 調整的時候會調過去)的變動 '''
    if tensor.dim() != 3:
        raise ValueError("The input tensor must be 3-dimensional.")
    diff = torch.cat((torch.zeros(windows, tensor.size(1)), (get_diagonal_values(tensor)[windows:] -get_diagonal_values(tensor)[:-windows])), dim=0)
    result = place_data_on_three_dim_main_diagonal(diff)

    return result

def diff_record_daily_value_change(tensor,windows = 60):
    ''' 每天看到數值的變動(看到就記錄下來) '''
    if tensor.dim() != 3:
        raise ValueError("The input tensor must be 3-dimensional.")
    diff = torch.cat((torch.zeros((tensor.size(0), tensor.size(1), windows)), tensor[:, :, windows:] - tensor[:, :, :-windows]), dim=2)

    return diff


def percent_change_track_adjusted_values_change(tensor,windows = 60):
    ''' 每天看到所有最新數值(調整後 調整的時候會調過去)的變動  動態'''

    if tensor.dim() != 3:
        raise ValueError("The input tensor must be 3-dimensional.")
        
    # 獲取 tensor 的主對角線值
    diagonal_values = get_diagonal_values(tensor)
    
    # 計算調整後的百分比變動
    past_values = diagonal_values[:-windows]
    current_values = diagonal_values[windows:]
    
    # 避免分母為零
    percent_change = (current_values - past_values) / (past_values + 1e-8)
    diff = torch.cat((torch.zeros(windows, tensor.size(1)), percent_change), dim=0)
    
    result = place_data_on_three_dim_main_diagonal(diff)
    return result

def percent_change_record_daily_value_change(tensor, windows=60):
    ''' 每天看到數值的百分比變動 (看到就記錄下來) 靜態 '''
    if tensor.dim() != 3:
        raise ValueError("The input tensor must be 3-dimensional.")
    
    # 計算百分比變動
    past_values = tensor[:, :, :-windows]
    current_values = tensor[:, :, windows:]
    
    # 避免分母為零
    percent_change = (current_values - past_values) / (past_values + 1e-8)
    diff = torch.cat((torch.zeros((tensor.size(0), tensor.size(1), windows)), percent_change), dim=2)
    
    return diff


def fillna(tensor, value=0.0):
    ''' 將 NaN 值填充為指定值 '''
    return torch.nan_to_num(tensor, nan=value)

def rule_calculate(operation, **variables):
    ''' operation : 文字 Rule
        tensors : 輸入的tensor (至少一個可以是多個)
        輸出維度 [day, factor] 
    '''

    result = eval(operation, globals(), variables) * 1# 使用空的全局和局部變數字典來避免潛在的安全問題

    if result.dim() == 3:
        result = torch.diagonal(result, dim1=0, dim2=2).T # 取出主對角線的資料
    elif result.dim() == 2:
        pass # 格式正確 甚麼都不做
    elif result.dim() == 1:
        result = result.unsqueeze(1) # 加入第二個維度
    return result

def get_diagonal_values(tensor):
    ''' 找出主對角線的值 '''
    return torch.diagonal(tensor, dim1=0, dim2=2).T
def load_tensor(path,method = 'basic'):
    ''' 讀取tensor '''
    if method.lower() == 'basic':
        return torch.load(path)
    elif method.lower() == 'gzip':
        return load_tensor_gzip(path)
    elif method.lower() == 'hdf5':
        return load_tensor_hdf5(path)
def save_tensor(tensor, path,method = 'basic'):
    ''' 儲存tensor '''
    if method.lower() == 'basic':
        torch.save(tensor, path, pickle_protocol=5)
    elif method.lower() == 'gzip':
        save_tensor_gzip(tensor, path)
    elif method.lower() == 'hdf5':
        save_tensor_hdf5(tensor, path)
        
def save_tensor_hdf5(tensor, path):
    ''' hdf5附檔名結尾 ''' 
    try:
        with h5py.File(path, 'w') as f:
            f.create_dataset('tensor', data=tensor.numpy(), compression='gzip')
        print(f"Tensor saved to {path}")
    except Exception as e:
        print(f"Failed to save tensor to {path}: {e}")

def load_tensor_hdf5(path):
    ''' hdf5附檔名結尾 ''' 
    try:
        with h5py.File(path, 'r') as f:
            data = f['tensor'][:]
        tensor = torch.tensor(data)
        print(f"Tensor loaded from {path}")
        return tensor
    except Exception as e:
        print(f"Failed to load tensor from {path}: {e}")
        
def save_tensor_gzip(tensor, path):
    # .pt.gz 結尾
    try:
        with gzip.open(path, 'wb') as f:
            torch.save(tensor, f)
        print(f"Tensor saved to {path}")
    except Exception as e:
        print(f"Failed to save tensor to {path}: {e}")
    
def load_tensor_gzip(path):
    # .pt.gz 結尾
    try:
        with gzip.open(path, 'rb') as f:
            tensor = torch.load(f)
        print(f"Tensor loaded from {path}")
        return tensor
    except Exception as e:
        print(f"Failed to load tensor from {path}: {e}")
def pct_change(numerator, denominator,day = 60):
    ''' 
    numerator 分子
    denominator : 分母
    day : 往前補齊天數
    '''
    # 計算百分比變化
    value = (numerator /  denominator) - 1
    return torch.cat((torch.zeros(day, numerator.size(1)),value), dim=0)

def save_pickle(file,path):
    ''' 儲存pickle檔案 '''
    with open(path, 'wb') as f:
        pickle.dump(file, f)
def load_pickle(path):
    ''' 讀取pickle檔案 '''
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file