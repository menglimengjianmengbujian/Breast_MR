
import pandas as pd

def merge_csv_files(file1, file2, file3, output_file):
    """
    合并三个标准化和归一化后的CSV文件，保留姓名列，并为每个文件的特征列加上后缀区分。

    参数:
    file1, file2, file3: str
        输入CSV文件的路径。
    output_file: str
        输出合并后CSV文件的路径。
    """
    # 读取文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    # 确保第一列是姓名列
    df1 = df1.rename(columns={df1.columns[0]: 'Name'})
    df2 = df2.rename(columns={df2.columns[0]: 'Name'})
    df3 = df3.rename(columns={df3.columns[0]: 'Name'})

    # 将特征列加上后缀
    df1 = df1.add_suffix('_1')
    df2 = df2.add_suffix('_2')
    df3 = df3.add_suffix('_3')

    # 保留姓名列，不加后缀
    df1 = df1.rename(columns={'Name_1': 'Name'})
    df2 = df2.rename(columns={'Name_2': 'Name'})
    df3 = df3.rename(columns={'Name_3': 'Name'})

    # 合并数据
    merged_df = df1.merge(df2, on='Name').merge(df3, on='Name')

    # 保存合并后的数据
    merged_df.to_csv(output_file, index=False)

    print(f"合并后的文件已保存至 {output_file}")

# 示例调用，保存良性的三个CSV文件合并
file1 = r"C:\\Users\\Administrator\\Desktop\\Breast\\benignDCE_Stand_MinMaxScaler.csv"
file2 = r"C:\\Users\\Administrator\\Desktop\\Breast\\benignT1_Stand_MinMaxScaler.csv"
file3 = r"C:\\Users\\Administrator\\Desktop\\Breast\\benignT2_Stand_MinMaxScaler.csv"
output_file = r"C:\\Users\\Administrator\\Desktop\\Breast\\benign_merged.csv"

merge_csv_files(file1, file2, file3, output_file)
print("三个良性cvs文件保存完毕。")
#/ 示例调用，保存良恶的三个CSV文件合并
file1 = r"C:\Users\Administrator\Desktop\Breast\malignantDCE_Stand_MinMaxScaler.csv"
file2 = r"C:\\Users\\Administrator\\Desktop\\Breast\\malignantT1_Stand_MinMaxScaler.csv"
file3 = r"C:\\Users\\Administrator\\Desktop\\Breast\\malignantT2_Stand_MinMaxScaler.csv"
output_file = r"C:\\Users\\Administrator\\Desktop\\Breast\\malignant_merged.csv"

merge_csv_files(file1, file2, file3, output_file)
print("三个恶性cvs文件保存完毕。")