import os
import pandas as pd
from django.core.management.base import BaseCommand  # 导入Django自定义命令基类
from django.db import transaction  # 事务支持，确保导入原子性
from house_data.models import HouseDeal  # 导入你的模型（app名是house_data）

def clean_area(area_str):
    """
    清洗面积字段：去除"㎡"符号，转换为浮点数
    处理空值和异常值
    """
    if pd.isna(area_str) or area_str == '':
        return None
    try:
        # 提取数字部分，去除㎡及其他无关字符
        area_num = ''.join([c for c in str(area_str) if c.isdigit() or c == '.'])
        return float(area_num)
    except Exception as e:
        print(f"面积清洗失败：{area_str}，错误：{e}")
        return None

def clean_deal_price(price_str):
    """
    清洗成交总价字段：去除"万"符号，转换为整数
    处理空值和异常值
    """
    if pd.isna(price_str) or price_str == '':
        return None
    try:
        # 提取数字部分，去除万及其他无关字符
        price_num = ''.join([c for c in str(price_str) if c.isdigit() or c == '.'])
        return int(float(price_num))  # 先转浮点数再转整数，兼容带小数的价格（如183.5万）
    except Exception as e:
        print(f"价格清洗失败：{price_str}，错误：{e}")
        return None

def clean_deal_date(date_str):
    """
    清洗成交日期字段：确保为合法日期格式，转换为date类型
    """
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        return pd.to_datetime(date_str).date()
    except Exception as e:
        print(f"日期清洗失败：{date_str}，错误：{e}")
        return None

def import_single_csv(csv_path):
    """
    导入单个CSV文件的数据
    使用get_or_create避免重复导入（根据house_id+title+deal_date唯一标识一条记录）
    """
    print(f"\n开始处理文件：{os.path.basename(csv_path)}")
    # 尝试用utf-8编码读取，失败则用gbk
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='gbk')
        except Exception as e:
            print(f"读取文件{os.path.basename(csv_path)}失败，不支持的编码格式，错误：{e}")
            return 0  # 返回0表示导入失败

    # 筛选有效字段（确保CSV字段与模型字段对应）
    required_fields = ['house_id', 'title', 'district', 'district_area', 'layout', 'area', 'floor', 'direction', 'deal_price', 'deal_unit_price', 'deal_date']
    missing_fields = [f for f in required_fields if f not in df.columns]
    if missing_fields:
        print(f"文件{os.path.basename(csv_path)}缺少必要字段：{missing_fields}，跳过导入")
        return 0

    # 数据清洗
    df['area'] = df['area'].apply(clean_area)
    df['deal_price'] = df['deal_price'].apply(clean_deal_price)
    df['deal_date'] = df['deal_date'].apply(clean_deal_date)
    # 填充空值
    str_fields = ['floor', 'direction', 'deal_unit_price']
    df[str_fields] = df[str_fields].fillna('')
    # 批量导入数据库（使用事务，出错时回滚）
    success_count = 0
    fail_count = 0
    with transaction.atomic():
        for index, row in df.iterrows():
            try:
                # get_or_create：存在则获取，不存在则创建，避免重复
                HouseDeal.objects.get_or_create(
                    house_id=row['house_id'],
                    title=row['title'],
                    deal_date=row['deal_date'],
                    defaults={
                        'district': row['district'],
                        'district_area': row['district_area'],
                        'layout': row['layout'],
                        'area': row['area'],
                        'floor': row['floor'],
                        'direction': row['direction'],
                        'deal_price': row['deal_price'],
                        'deal_unit_price': row['deal_unit_price'] if row['deal_unit_price'] != '' else None,
                        'city': row['city'] if  'city' in  row else '重庆',
                    }
                )
                success_count += 1
                # 每导入100条打印进度
                if (success_count + fail_count) % 100 == 0:
                    print(f"已处理{success_count + fail_count}条记录，成功{success_count}条，失败{fail_count}条")
            except Exception as e:
                fail_count += 1
                print(f"第{index+1}条记录导入失败：{row['title']}，错误：{e}")

    print(f"文件{os.path.basename(csv_path)}处理完成：成功{success_count}条，失败{fail_count}条")
    return success_count

class Command(BaseCommand):
    """
    Django自定义命令：批量导入房屋交易CSV数据
    执行命令：python manage.py import_house
    """
    # 命令帮助信息
    help = '批量导入D:\\pythonfinal\\clear_data目录下的房屋交易CSV数据到数据库'

    def handle(self, *args, **options):
        """
        命令执行的核心逻辑
        """
        csv_dir = r"D:\pythonfinal\clear_data_chongqing"  # CSV文件存放目录
        # 检查目录是否存在
        if not os.path.exists(csv_dir):
            self.stdout.write(self.style.ERROR(f"错误：目录{csv_dir}不存在，请检查路径是否正确"))
            return

        # 遍历目录下所有CSV文件
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if not csv_files:
            self.stdout.write(self.style.WARNING(f"目录{csv_dir}下未找到任何CSV文件"))
            return

        total_success = 0
        self.stdout.write(self.style.SUCCESS(f"共发现{len(csv_files)}个CSV文件，开始批量导入..."))

        for csv_file in csv_files:

            csv_path = os.path.join(csv_dir, csv_file)
            success_num = import_single_csv(csv_path)
            total_success += success_num

        self.stdout.write(self.style.SUCCESS(f"\n所有文件导入完成！总计成功导入{total_success}条有效记录"))