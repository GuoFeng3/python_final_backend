from django.db import models
from django.utils import timezone


class HouseDeal(models.Model):
    """
    房屋交易数据模型
    对应你提供的房屋交易CSV数据
    """
    # house_id：唯一标识字符串（注意：CSV中该字段有重复，不适合作为Django默认主键，故单独定义为普通字段）
    house_id = models.CharField(
        verbose_name="房屋唯一标识ID",
        max_length=50,
        db_index=True  # 建立索引，提升查询效率
    )
    # title：房屋标题
    title = models.CharField(
        verbose_name="房屋标题",
        max_length=200
    )
    city = models.CharField(
        verbose_name="城市",
        max_length=50,
        default="北京",  # 新增数据默认值为北京
        db_index=True    # 建立索引，便于按城市查询
    )
    # district：行政区（如昌平）
    district = models.CharField(
        verbose_name="行政区",
        max_length=50,
        db_index = True
    )
    # district_area：细分区域（如昌平-回龙观）
    district_area = models.CharField(
        verbose_name="细分区域",
        max_length=100,
        db_index=True
    )
    # layout：户型（如1室1厅）
    layout = models.CharField(
        verbose_name="户型",
        max_length=50,
        db_index=True
    )
    # area：房屋面积（去除㎡符号后存储为浮点数，更便于计算）
    area = models.FloatField(
        verbose_name="房屋面积（平方米）"
    )
    # floor：楼层信息（如共6层）
    floor = models.CharField(
        verbose_name="楼层信息",
        max_length=50
    )
    # direction：朝向（如南）
    direction = models.CharField(
        verbose_name="房屋朝向",
        max_length=20,
        db_index=True
    )
    # deal_price：成交总价（如183万，存储为整数，单位：万元，便于统计）
    deal_price = models.IntegerField(
        verbose_name="成交总价（万元）"
    )
    # deal_unit_price：成交单价（CSV中为空，允许为NULL）
    deal_unit_price = models.CharField(
        verbose_name="成交单价",
        max_length=50,
        blank=True,
        null=True  # 允许数据库中该字段为NULL
    )
    # deal_date：成交日期
    deal_date = models.DateField(
        verbose_name="成交日期",
        db_index=True
    )

    class Meta:
        """
        模型元数据配置
        """
        verbose_name = "房屋交易记录"
        verbose_name_plural = "房屋交易记录"  # 复数形式与单数一致，更符合中文习惯
        db_table = "house_deal"  # 数据库表名，自定义避免Django默认的app_model格式
        ordering = ["-deal_date"]  # 默认按成交日期倒序排列（最新成交在前）
        indexes = [
            # 联合索引：便于按区域+户型查询交易记录
            models.Index(fields=["district", "layout"]),
            # 联合索引：便于按区域+成交日期查询
            models.Index(fields=["district_area", "deal_date"]),
        ]

    def __str__(self):
        """
        模型实例的字符串表示，便于在admin后台和终端查看
        """
        return f"{self.title} - {self.deal_price}万 - {self.deal_date}"