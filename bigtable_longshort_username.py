import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from function_library import load_tensor, load_pickle
import os
import rule_library_username as rule_library

class BuildBigtable():
    def __init__(self, file_path="拐點標註檔.csv", rule_list=['rule_1', 'rule_2']):
        self.df = pd.read_csv(file_path, encoding='utf-8-sig')
        self.build(self.df, rule_list)

    @staticmethod
    def build(df, rule_list):
        rule_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rule')
        measure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'measure')

        results_list = []
        long_results_list = []
        short_results_list = []

        for rule_name in rule_list:
            obj_rule = rule_library.Rule_Library()
            exec(f"obj_rule.{rule_name}()")
            measure_name = obj_rule.cross_section_data_file_name[0]
            print(f"分析規則 {rule_name} | {measure_name}")

            tensor = load_tensor(os.path.join(rule_path, rule_name + ".pt"), method='hdf5')
            date_list = load_pickle(os.path.join(measure_path, measure_name, measure_name + "_date_list.pkl"))

            df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
            date_df = pd.DataFrame({'Date': date_list})
            merged_df = pd.merge(date_df, df, on='Date', how='left')
            merged_df[['波段低點區間', '波段高點區間']] = merged_df[['波段低點區間', '波段高點區間']].fillna(0)

            assert tensor.shape[0] == len(date_list), "日期數量不一致"
            tensor_df = pd.DataFrame(tensor, columns=[rule_name])
            merged_df = pd.concat([merged_df, tensor_df], axis=1)

            assert tensor.shape[0] == merged_df.shape[0], "合併後日期大小改變"

            for lonshort in ['long', 'short']:
                result_dictionary_sub = {}
                if lonshort.lower() == 'long':        
                    y_true = merged_df['波段低點區間']
                elif lonshort.lower() == 'short':
                    y_true = merged_df['波段高點區間']
                    
                y_pred = merged_df[rule_name]
                confusion_matrix_result = confusion_matrix(y_true, y_pred)
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)

                result_dictionary_sub['start_date'] = merged_df['Date'].max()
                result_dictionary_sub['end_date'] = merged_df['Date'].min()
                result_dictionary_sub['LongShort'] = lonshort
                result_dictionary_sub['measure'] = obj_rule.cross_section_data_file_name
                result_dictionary_sub['rule'] = rule_name
                result_dictionary_sub['rule_describe'] = obj_rule.description
                result_dictionary_sub['accuracy'] = accuracy
                result_dictionary_sub['precision'] = precision
                result_dictionary_sub['recall'] = recall
                result_dictionary_sub['f1_score'] = f1
                result_dictionary_sub['signal_count_0'] = (y_pred == 0).sum()
                result_dictionary_sub['signal_count_1'] = (y_pred == 1).sum()

                results_list.append(result_dictionary_sub)

                if lonshort.lower() == 'long'.lower():
                    long_results_list.append(result_dictionary_sub)
                elif lonshort.lower() == 'short'.lower():
                    short_results_list.append(result_dictionary_sub)

        # 生成bigtable表格
        df_results = pd.DataFrame(results_list)
        df_results.index.name = 'index'
        df_results.to_csv('bigtable.csv', encoding='utf-8-sig')

        # 生成long和short的表格並按照precision排序
        long_df = pd.DataFrame(long_results_list).sort_values(by='precision', ascending=False)
        short_df = pd.DataFrame(short_results_list).sort_values(by='precision', ascending=False)

        long_df.index.name = 'index'
        short_df.index.name = 'index'

        long_df.to_csv('bigtable_long.csv', encoding='utf-8-sig')
        short_df.to_csv('bigtable_short.csv', encoding='utf-8-sig')

        print("bigtable, long and short tables have been saved, sorted by precision.")

if __name__ == '__main__':
    file_path = "拐點標註檔.csv"
    rule_list = ['rule_1','rule_2','rule_3','rule_4','rule_5','rule_6','rule_7','rule_8','rule_9','rule_10',
                 'rule_11','rule_12','rule_13','rule_14','rule_15','rule_16','rule_17','rule_18','rule_19','rule_20',
                 'rule_21','rule_22','rule_23','rule_24','rule_25','rule_26','rule_27','rule_28','rule_29','rule_30',
                 'rule_31','rule_32','rule_33','rule_34','rule_35','rule_36','rule_37','rule_38','rule_39','rule_40',
                 'rule_41','rule_42','rule_43','rule_44','rule_45','rule_46','rule_47','rule_48','rule_49','rule_50',
                 'rule_51','rule_52','rule_53','rule_54','rule_55','rule_56','rule_57','rule_58','rule_59','rule_60',
                 'rule_61','rule_62','rule_63','rule_64','rule_65','rule_66','rule_67','rule_68','rule_69','rule_70',
                 'rule_71','rule_72','rule_73','rule_74','rule_75','rule_76','rule_77','rule_78','rule_79','rule_80',
                 'rule_81','rule_82','rule_83','rule_84','rule_85','rule_86','rule_87','rule_88','rule_89','rule_90',
                 'rule_91','rule_92','rule_93','rule_94','rule_95','rule_96','rule_97','rule_98','rule_99','rule_100',
                 'rule_101','rule_102','rule_103','rule_104','rule_105','rule_106','rule_107','rule_108','rule_109','rule_110',
                 'rule_111','rule_112','rule_113','rule_114','rule_115','rule_116','rule_117','rule_118','rule_119','rule_120',
                 'rule_121','rule_122','rule_123']
    obj_build_bigtable = BuildBigtable(file_path=file_path, rule_list=rule_list)

