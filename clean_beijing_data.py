import pandas as pd
import re

def clean_data():
    source_file = 'd:/pythonfinal/cleaned_house_data_beijing.csv'
    target_file = 'd:/pythonfinal/cleaned_house_data_beijing_final.csv'
    
    # 读取源文件
    df = pd.read_csv(source_file)
    
    # 过滤掉 publish_date 为空的数据
    original_count = len(df)
    df = df.dropna(subset=['publish_date'])
    # 确保去除空白字符串
    df = df[df['publish_date'].astype(str).str.strip() != '']
    print(f"Filtered out {original_count - len(df)} rows with missing publish_date")
    
    # 创建新的 DataFrame
    new_df = pd.DataFrame()
    
    # 1. house_id: 从 link 中提取数字 ID
    def extract_id(link):
        match = re.search(r'/(\d+)\.html', str(link))
        if match:
            return match.group(1)
        return str(link)
    
    new_df['house_id'] = df['link'].apply(extract_id)
    
    # 2. title
    new_df['title'] = df['title']
    
    # 3. district: 使用 district_name (中文)
    new_df['district'] = df['district_name']
    
    # 4. district_area: 组合 district_name 和 region，格式如 "西城-牛街"
    # 如果 region 为空，直接用 district_name
    new_df['district_area'] = df.apply(
        lambda row: f"{row['district_name']}-{row['region']}" if pd.notna(row['region']) else row['district_name'], 
        axis=1
    )
    
    # 5. layout
    new_df['layout'] = df['layout']
    
    # 6. area: 将 "平米" 替换为 "㎡"
    new_df['area'] = df['area'].astype(str).str.replace('平米', '㎡')
    
    # 7. floor: 使用 floor_info
    new_df['floor'] = df['floor_info']
    
    # 8. direction
    new_df['direction'] = df['direction']
    
    # 9. deal_price: 使用 total_price (总价)
    new_df['deal_price'] = df['total_price']
    
    # 10. deal_unit_price: 使用 unit_price
    # 源数据可能是 "64,072元/平"，目标数据可能为空或只是数字
    # 这里保留源数据格式，或者清洗为纯数字字符串？
    # 模型定义是 CharField，且目标 CSV 示例为空。
    # 我们保留原始值，或者去掉 "元/平" 和逗号？
    # 为了保险，保留原值，或者做简单清洗。
    # 用户要求：与目标列名相同形式。
    new_df['deal_unit_price'] = df['unit_price']
    
    # 11. deal_date: 使用 publish_date，只取日期部分 YYYY-MM-DD
    # publish_date 格式如 "2025-12-14 16:50:40" (或空)
    def format_date(date_str):
        if pd.isna(date_str):
            return None
        try:
            return str(date_str).split(' ')[0]
        except:
            return date_str
            
    new_df['deal_date'] = df['publish_date'].apply(format_date)
    
    # 保存结果
    new_df.to_csv(target_file, index=False, encoding='utf-8-sig')
    print(f"Successfully cleaned data to {target_file}")
    print("Columns:", new_df.columns.tolist())
    print("First 5 rows:")
    print(new_df.head().to_string())

if __name__ == '__main__':
    clean_data()
