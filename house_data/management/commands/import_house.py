import os
import pandas as pd
from django.core.management.base import BaseCommand
from django.db import transaction
from house_data.models import HouseDeal

def clean_area(area_str):
    if pd.isna(area_str) or area_str == '':
        return None
    try:
        area_num = ''.join([c for c in str(area_str) if c.isdigit() or c == '.'])
        return float(area_num)
    except Exception as e:
        print(f"面积清洗失败：{area_str}，错误：{e}")
        return None

def clean_deal_price(price_str):
    if pd.isna(price_str) or price_str == '':
        return None
    try:
        price_num = ''.join([c for c in str(price_str) if c.isdigit() or c == '.'])
        return int(float(price_num))
    except Exception as e:
        print(f"价格清洗失败：{price_str}，错误：{e}")
        return None

def clean_deal_date(date_str):
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        return pd.to_datetime(date_str).date()
    except Exception as e:
        print(f"日期清洗失败：{date_str}，错误：{e}")
        return None

def import_single_csv(csv_path):
    """
    仅导入/更新CSV中「密云」相关的数据
    """
    print(f"\n开始处理文件：{os.path.basename(csv_path)}")
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='gbk')
        except Exception as e:
            print(f"读取文件{os.path.basename(csv_path)}失败，错误：{e}")
            return 0

    # 检查必要字段
    required_fields = ['house_id', 'title', 'district', 'district_area', 'layout', 'area', 'floor', 'direction', 'deal_price', 'deal_unit_price', 'deal_date']
    missing_fields = [f for f in required_fields if f not in df.columns]
    if missing_fields:
        print(f"文件{os.path.basename(csv_path)}缺少必要字段：{missing_fields}，跳过导入")
        return 0

    # 关键修改1：筛选出「密云」数据（根据实际字段值调整，如district或district_area为“密云”）
    # 先确认你的CSV中密云的字段值（是“密云”/“密云区”，对应district或district_area，此处以district为例，可按需修改）
    miyun_df = df[df['district'].str.contains('密云', na=False)]  # 模糊匹配，兼容“密云”“密云区”
    if miyun_df.empty:
        print(f"文件{os.path.basename(csv_path)}中无密云相关数据，跳过")
        return 0

    # 数据清洗（仅清洗密云数据）
    miyun_df['area'] = miyun_df['area'].apply(clean_area)
    miyun_df['deal_price'] = miyun_df['deal_price'].apply(clean_deal_price)
    miyun_df['deal_date'] = miyun_df['deal_date'].apply(clean_deal_date)
    str_fields = ['floor', 'direction', 'deal_unit_price']
    miyun_df[str_fields] = miyun_df[str_fields].fillna('')

    # 批量更新/创建密云数据
    success_count = 0
    fail_count = 0
    with transaction.atomic():
        for index, row in miyun_df.iterrows():
            try:
                # 关键修改2：先查询是否存在，存在则更新，不存在则创建
                defaults = {
                    'district': row['district'],
                    'district_area': row['district_area'],
                    'layout': row['layout'],
                    'area': row['area'],
                    'floor': row['floor'],
                    'direction': row['direction'],
                    'deal_price': row['deal_price'],
                    'deal_unit_price': row['deal_unit_price'] if row['deal_unit_price'] != '' else None
                }
                # 按唯一标识查询
                obj, created = HouseDeal.objects.update_or_create(
                    house_id=row['house_id'],
                    title=row['title'],
                    deal_date=row['deal_date'],
                    defaults=defaults
                )
                # created为True表示新增，False表示更新
                success_count += 1
                if (success_count + fail_count) % 100 == 0:
                    print(f"已处理密云数据{success_count + fail_count}条，成功{success_count}条，失败{fail_count}条")
            except Exception as e:
                fail_count += 1
                print(f"第{index+1}条密云数据导入失败：{row['title']}，错误：{e}")

    print(f"文件{os.path.basename(csv_path)}密云数据处理完成：成功{success_count}条（新增/更新），失败{fail_count}条")
    return success_count

class Command(BaseCommand):
    help = '仅更新D:\\pythonfinal\\clear_data目录下CSV中的密云房屋交易数据'

    def handle(self, *args, **options):
        csv_dir = r"D:\pythonfinal\clear_data"
        if not os.path.exists(csv_dir):
            self.stdout.write(self.style.ERROR(f"错误：目录{csv_dir}不存在"))
            return

        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if not csv_files:
            self.stdout.write(self.style.WARNING(f"目录{csv_dir}下无CSV文件"))
            return

        total_success = 0
        self.stdout.write(self.style.SUCCESS(f"共发现{len(csv_files)}个CSV文件，开始筛选并更新密云数据..."))

        for csv_file in csv_files:
            csv_path = os.path.join(csv_dir, csv_file)
            success_num = import_single_csv(csv_path)
            total_success += success_num

        self.stdout.write(self.style.SUCCESS(f"\n所有文件处理完成！总计成功更新/新增{total_success}条密云数据"))