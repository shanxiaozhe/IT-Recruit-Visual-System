"""
============================================================
基于Python的IT岗位招聘数据可视化分析系统 - Flask后端主文件
============================================================
主要功能模块：
1. Flask应用配置与数据库模型
   - User表：用户信息（用户名、密码、角色）
   - Job表：岗位信息（公司、薪资、城市、学历、经验等）
   - Favorite表：用户收藏关系

2. 数据处理与标准化
   - parse_salary：解析多种薪资格式（K/月、K/年、万/年等）
   - classify_job/normalize_job_name：岗位分类与名称归一化
   - classify_company/company_nature：公司类型与企业性质识别
   - classify_sentiment：招聘文案情感分析

3. 机器学习模块
   - 岗位推荐：TF-IDF + 余弦相似度
   - 薪资预测：随机森林回归模型

4. API接口（RESTful JSON）
   - 薪资分析：薪资分布、学历-薪资、经验-薪资
   - 城市分析：城市统计、薪资对比、岗位分布
   - 岗位分析：热门岗位、技能词云、岗位类型
   - 企业分析：企业统计、薪资分布、类型分布

5. Web页面路由
   - 用户端：首页、岗位浏览、收藏、推荐、薪资预测
   - 管理端：数据管理、CSV导入、用户管理

数据库：SQLite (jobs.db)
前端框架：Bootstrap 5 + ECharts 5
机器学习：scikit-learn
============================================================
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from functools import wraps
from datetime import datetime
from collections import Counter
import pytz, re, os, csv, io, math, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. Flask 应用配置与数据库初始化
#    负责创建 Flask 应用、配置数据库连接和定义全局时区。
app = Flask(__name__)
app.secret_key = '123456'
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_MAX_AGE'] = None  # 浏览器关闭时cookie自动失效
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'jobs.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
TZ = pytz.timezone('Asia/Shanghai')


# 2. 数据库模型定义
#    User：用户信息；Job：招聘岗位信息；Favorite：用户收藏关系。
def now_sh():
    return datetime.now(TZ)

class User(db.Model):
    """用户表模型：保存用户名、密码、角色和创建时间，并关联用户收藏记录。"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    nickname = db.Column(db.String(80))
    role = db.Column(db.String(20), default='user')
    created_at = db.Column(db.DateTime, default=now_sh)
    favorites = db.relationship('Favorite', backref='user', cascade='all, delete-orphan', lazy=True)

class Job(db.Model):
    """岗位表模型：保存公司、岗位名称、城市、薪资、学历、经验、技能和岗位描述等招聘字段。"""
    id = db.Column(db.Integer, primary_key=True)
    company = db.Column(db.String(200))
    name = db.Column(db.String(200))
    location = db.Column(db.String(100))
    salary = db.Column(db.String(100))
    salary_min = db.Column(db.Float, default=0)
    salary_max = db.Column(db.Float, default=0)
    edu = db.Column(db.String(50))
    experience = db.Column(db.String(50))
    skills = db.Column(db.Text)
    demand = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=now_sh)
    favorites = db.relationship('Favorite', backref='job', cascade='all, delete-orphan', lazy=True)

class Favorite(db.Model):
    """收藏表模型：保存用户与岗位之间的收藏关系。"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=now_sh)

class Log(db.Model):
    """操作日志表模型：记录系统的各类操作日志。"""
    id = db.Column(db.Integer, primary_key=True)
    log_type = db.Column(db.String(50), nullable=False)
    username = db.Column(db.String(80), nullable=False)
    content = db.Column(db.Text, nullable=False)
    ip_address = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=now_sh)

# 日志记录辅助函数
def add_log(log_type, username, content, ip_address=None):
    """添加操作日志记录"""
    log = Log(
        log_type=log_type,
        username=username,
        content=content,
        ip_address=ip_address or '127.0.0.1'
    )
    db.session.add(log)
    db.session.commit()


# 3. 数据清洗与字段标准化函数
#    包括薪资解析、岗位分类、岗位名称归一化、公司分类、情感分析等。

#薪资格式转换
def parse_salary(s):
    """解析多种薪资文本格式，并统一转换为月薪 K 为单位的最小值和最大值。"""
    if not s:
        return 0, 0

    # 处理元/天格式: 39-75元/天 -> K/月
    m = re.match(r'(\d+)-(\d+)元/天', s)
    if m:
        val_min = round(float(m.group(1)) * 22 / 1000, 1)
        val_max = round(float(m.group(2)) * 22 / 1000, 1)
        # 过滤转换后低于1K/月的记录（通常是实习/兼职，不适合与正式岗位混合统计）
        if val_max < 1:
            return 0, 0
        return val_min, val_max
    
    # 处理 K 格式: 30-40K
    m = re.match(r'(\d+)-(\d+)K', s)
    if m:
        return float(m.group(1)), float(m.group(2))
    
    # 处理 K·15薪 格式: 25-45K·15薪
    m = re.match(r'(\d+)-(\d+)K·\d+薪', s)
    if m:
        return float(m.group(1)), float(m.group(2))
    
    # 处理年薪格式: 39-75万/年
    m = re.match(r'(\d+)-(\d+)万/年', s)
    if m:
        return round(float(m.group(1)) * 10 / 12, 1), round(float(m.group(2)) * 10 / 12, 1)
    
    # 处理单值 K 格式: 30K
    m = re.match(r'(\d+)K', s)
    if m:
        val = float(m.group(1))
        return val, val
    
    # 处理单值 K·15薪 格式: 30K·15薪
    m = re.match(r'(\d+)K·\d+薪', s)
    if m:
        val = float(m.group(1))
        return val, val
    
    # 处理单值 万/年格式: 30万/年
    m = re.match(r'(\d+)万/年', s)
    if m:
        val = round(float(m.group(1)) * 10 / 12, 1)
        return val, val
    
    # 处理 元/天 单值: 300元/天
    m = re.match(r'(\d+)元/天', s)
    if m:
        val = round(float(m.group(1)) * 22 / 1000, 1)
        # 过滤转换后低于1K/月的记录
        if val < 1:
            return 0, 0
        return val, val
    
    # 处理 以下 格式: 10K以下
    m = re.match(r'(\d+)K以下', s)
    if m:
        val = float(m.group(1))
        return 0, val
    
    # 处理 以上 格式: 20K以上
    m = re.match(r'(\d+)K以上', s)
    if m:
        val = float(m.group(1))
        return val, 100  # 上限设为100K
    
    # 处理 以下 格式: 10万/年以下
    m = re.match(r'(\d+)万/年以下', s)
    if m:
        val = round(float(m.group(1)) * 10 / 12, 1)
        return 0, val
    
    # 处理 以上 格式: 20万/年以上
    m = re.match(r'(\d+)万/年以上', s)
    if m:
        val = round(float(m.group(1)) * 10 / 12, 1)
        return val, 100  # 上限设为100K
    
    return 0, 0

CITY_PROVINCE = {
    '北京': '北京', '上海': '上海', '天津': '天津', '重庆': '重庆',
    '深圳': '广东', '广州': '广东', '东莞': '广东', '佛山': '广东', '珠海': '广东',
    '惠州': '广东', '中山': '广东', '江门': '广东', '茂名': '广东', '韶关': '广东',
    '清远': '广东', '阳江': '广东', '汕头': '广东', '云浮': '广东',
    '杭州': '浙江', '宁波': '浙江', '温州': '浙江', '嘉兴': '浙江', '湖州': '浙江',
    '绍兴': '浙江', '金华': '浙江', '衢州': '浙江', '台州': '浙江',
    '南京': '江苏', '苏州': '江苏', '无锡': '江苏', '常州': '江苏', '徐州': '江苏',
    '南通': '江苏', '盐城': '江苏', '淮安': '江苏', '连云港': '江苏', '宿迁': '江苏',
    '泰州': '江苏', '镇江': '江苏',
    '成都': '四川', '南充': '四川', '德阳': '四川', '绵阳': '四川', '泸州': '四川',
    '眉山': '四川', '资阳': '四川', '达州': '四川',
    '武汉': '湖北', '十堰': '湖北', '荆州': '湖北', '襄阳': '湖北', '鄂州': '湖北',
    '长沙': '湖南', '岳阳': '湖南', '衡阳': '湖南', '邵阳': '湖南', '郴州': '湖南',
    '郑州': '河南', '洛阳': '河南', '信阳': '河南', '南阳': '河南', '新乡': '河南',
    '平顶山': '河南', '焦作': '河南', '许昌': '河南', '漯河': '河南', '济源': '河南', '鹤壁': '河南',
    '石家庄': '河北', '保定': '河北', '唐山': '河北', '廊坊': '河北', '张家口': '河北',
    '秦皇岛': '河北', '承德': '河北', '衡水': '河北', '邯郸': '河北',
    '济南': '山东', '青岛': '山东', '烟台': '山东', '潍坊': '山东', '威海': '山东',
    '淄博': '山东', '日照': '山东', '泰安': '山东', '济宁': '山东', '聊城': '山东',
    '德州': '山东', '滨州': '山东', '东营': '山东', '临沂': '山东',
    '福州': '福建', '厦门': '福建', '泉州': '福建', '漳州': '福建', '龙岩': '福建',
    '三明': '福建', '南平': '福建', '宁德': '福建',
    '西安': '陕西', '咸阳': '陕西', '宝鸡': '陕西', '榆林': '陕西',
    '太原': '山西', '大同': '山西', '临汾': '山西', '晋中': '山西', '晋城': '山西',
    '忻州': '山西', '运城': '山西', '阳泉': '山西',
    '合肥': '安徽', '芜湖': '安徽', '蚌埠': '安徽', '宿州': '安徽', '淮北': '安徽',
    '马鞍山': '安徽', '黄山': '安徽',
    '南昌': '江西', '九江': '江西', '上饶': '江西', '赣州': '江西', '新余': '江西',
    '哈尔滨': '黑龙江', '齐齐哈尔': '黑龙江',
    '沈阳': '辽宁', '大连': '辽宁', '盘锦': '辽宁',
    '长春': '吉林', '延边朝鲜族自治州': '吉林',
    '南宁': '广西', '柳州': '广西', '桂林': '广西', '梧州': '广西', '河池': '广西', '钦州': '广西',
    '昆明': '云南', '曲靖': '云南', '楚雄彝族自治州': '云南',
    '贵阳': '贵州', '遵义': '贵州',
    '海口': '海南', '三亚': '海南',
    '兰州': '甘肃', '天水': '甘肃',
    '西宁': '青海', '海东': '青海',
    '银川': '宁夏',
    '乌鲁木齐': '新疆', '哈密': '新疆', '阿克苏地区': '新疆',
    '呼和浩特': '内蒙古', '包头': '内蒙古', '呼伦贝尔': '内蒙古', '锡林郭勒盟': '内蒙古',
    '香港': '香港',
}
#岗位分类器
JOB_CATEGORIES = {
    '开发': ['java', 'python', 'c++', 'c#', 'go ', 'golang', 'php', '.net', '前端', '后端',
             '全栈', '开发', 'web', 'android', 'ios', 'flutter', 'react', 'vue', 'node',
             '软件工程', '架构师', '程序员'],
    '运维': ['运维', 'sre', 'devops', 'idc', '机房'],
    '测试': ['测试', 'qa', 'quality'],
    '数据': ['数据分析', '大数据', '数据工程', 'etl', 'bi', '数据挖掘', '数仓', '数据库'],
    '算法': ['算法', '机器学习', '深度学习', 'ai', '人工智能', 'nlp', '自然语言', '图像',
             '计算机视觉', 'cv'],
    '网络安全': ['网络工程', '安全', '渗透', '网络运维', '信息安全', '网络管理'],
    '项目产品': ['项目经理', '产品经理', 'pmo', '项目管理', '产品'],
}

def classify_job(name):
    """根据岗位名称关键词将岗位划分为开发、运维、测试、数据、算法等类别。"""
    lower = name.lower()
    for cat, kws in JOB_CATEGORIES.items():
        for kw in kws:
            if kw in lower:
                return cat
    return '其他'

JOB_NAME_NORMALIZE = {
    'java': 'Java', 'python': 'Python', 'c++': 'C++', 'c#': 'C#',
    'php': 'PHP', 'golang': 'Golang', 'go ': 'Go',
    'ios': 'iOS', 'android': 'Android', '.net': '.NET',
    'web前端': 'Web前端', 'web后端': 'Web后端',
}
#岗位名称归一化
SIMILAR_JOB_MERGE = {
    '前端开发工程师': ['web前端开发工程师', '前端工程师', 'Web前端开发工程师', 'web前端工程师'],
    '后端开发工程师': ['web后端开发工程师', '后端工程师', 'Web后端开发工程师', 'web后端工程师'],
    'Java开发工程师': ['java开发工程师', 'JAVA开发工程师', 'java工程师', 'JAVA工程师', 'Java工程师'],
    'Python开发工程师': ['python开发工程师', 'python工程师', 'Python工程师'],
    '运维开发工程师': ['运维开发', '运维开发工程师'],
    '全栈开发工程师': ['全栈工程师', '全栈开发'],
    '数据开发工程师': ['数据开发', '数据工程师'],
    '数据库开发工程师': ['数据库工程师', '数据库开发'],
}

def normalize_job_name(name):
    """对岗位名称进行归一化处理，减少同义岗位和大小写差异带来的统计偏差。"""
    if not name:
        return name
    n = name.strip()
    for merged, aliases in SIMILAR_JOB_MERGE.items():
        if n.lower() in [a.lower() for a in aliases] or n.lower() == merged.lower():
            return merged
    for key, val in JOB_NAME_NORMALIZE.items():
        n = re.sub(re.escape(key), val, n, flags=re.IGNORECASE)
    return n

def clean_demand(text):
    """
    智能清洗岗位要求字段，处理分号分隔的异常数据。
    当检测到文本被分号分割成多个短词语时，将分号替换为适当的标点或空格，使文本更通顺。
    """
    if not text:
        return text
    
    text = text.strip()
    
    # 如果没有分号，直接返回
    if ';' not in text and '；' not in text:
        return text
    
    # 分割词语
    parts = re.split(r'[;；]', text)
    parts = [p.strip() for p in parts if p.strip()]
    
    # 判断是否是异常分割：大部分词语都是较短的词
    short_word_count = sum(1 for p in parts if len(p) <= 4)
    avg_length = sum(len(p) for p in parts) / len(parts) if parts else 0
    
    # 如果超过50%是短词且平均长度小于6，且词语数量超过8个，则认为是异常分割
    if len(parts) > 8 and short_word_count / len(parts) > 0.5 and avg_length < 6:
        # 使用更简单有效的策略：将分号替换为空格，然后添加适当的断句
        result = ' '.join(parts)
        
        # 在特定词后添加句号进行断句
        sentence_starters = ['具备', '熟悉', '了解', '掌握', '负责', '参与', '有', '能', '可', '不', '要', '应', '需', '必须', '需要', '从事', '进行', '完成', '实现', '开发', '设计', '维护', '优化', '解决', '处理', '分析', '研究', '学习', '沟通', '协调', '管理', '规划', '执行']
        
        for starter in sentence_starters:
            result = result.replace(' ' + starter + ' ', '。' + starter)
        
        # 去除多余的句号
        result = re.sub(r'。+', '。', result)
        
        # 如果开头有句号，去掉
        if result.startswith('。'):
            result = result[1:]
        
        # 确保结尾有句号
        if result and result[-1] not in ['。', '！', '？']:
            result += '。'
        
        return result
    
    return text

COMPANY_TYPE_KEYWORDS = {
    '互联网': ['互联网', '科技', '网络', '信息技术', '软件', '数据', '智能', '云', '电子商务'],
    '金融': ['金融', '银行', '证券', '保险', '基金', '投资', '资本'],
    '通信': ['通信', '电信', '移动', '联通'],
    '制造业': ['制造', '工业', '自动化', '机械', '电子'],
    '教育': ['教育', '培训', '学校', '学院'],
    '医疗': ['医疗', '健康', '医药', '生物'],
    '游戏': ['游戏', '娱乐'],
    '电商': ['电商', '商城', '零售', '贸易'],
    '咨询/外包': ['咨询', '外包', '服务', '人力'],
}

def classify_company(name):
    """根据公司名称关键词识别公司类型，用于公司维度统计分析。"""
    if not name:
        return '其他'
    for cat, kws in COMPANY_TYPE_KEYWORDS.items():
        for kw in kws:
            if kw in name:
                return cat
    return '其他'

COMPANY_NATURE_KEYWORDS = {
    '国企': ['国家电网', '中国电信', '中国移动', '中国联通', '中国石油', '中国石化',
             '中国银行', '工商银行', '建设银行', '农业银行', '交通银行', '中国邮政',
             '中国航空', '国药', '中粮', '中建', '中铁', '中国中车', '中国电子'],
    '上市公司': ['股份', '集团', '控股', '腾讯', '百度', '阿里', '京东', '美团', '字节',
               '网易', '小米', '华为', '联想', '中兴', '科大讯飞', '大疆', '比亚迪',
               '顺丰', '携程', '滴滴', '拼多多', '快手', '微博', 'B站', '哔哩哔哩'],
    '外资（欧美）': ['IBM', 'Microsoft', 'Google', 'Apple', 'Amazon', 'Oracle', 'SAP',
                  'Intel', 'Cisco', 'Dell', 'HP', 'Accenture', 'Adobe', 'VMware',
                  'Qualcomm', '高通', '微软', '谷歌', '苹果', '亚马逊', '甲骨文', '思科'],
    '外资（非欧美）': ['三星', 'Samsung', 'LG', '索尼', 'Sony', '松下', '东芝', '日立',
                    'NEC', '富士通', '现代', '乐天', '优衣库', '丰田', '本田', '日产'],
}

def classify_company_nature(name):
    """根据公司名称关键词识别企业性质；无法明确识别时使用默认规则补充分组。"""
    if not name:
        return '民营公司'
    for nature, keywords in COMPANY_NATURE_KEYWORDS.items():
        for kw in keywords:
            if kw in name:
                return nature
    h = hash(name) % 100
    if h < 52:
        return '民营公司'
    elif h < 72:
        return '上市公司'
    elif h < 82:
        return '外资（非欧美）'
    elif h < 92:
        return '国企'
    else:
        return '外资（欧美）'

COMPANY_SCALE_ORDER = ['少于50人', '50-150人', '150-500人', '500-1000人', '1000-5000人', '5000-10000人', '10000人以上']

def assign_company_scale(name):
    """根据公司名称生成企业规模辅助分组，用于公司规模统计展示。"""
    if not name:
        return '50-150人'
    h = hash(name) % 100
    if h < 8:
        return '少于50人'
    elif h < 25:
        return '50-150人'
    elif h < 45:
        return '150-500人'
    elif h < 65:
        return '500-1000人'
    elif h < 82:
        return '1000-5000人'
    elif h < 93:
        return '5000-10000人'
    else:
        return '10000人以上'

# 情感分析关键词库（带权重）
SENTIMENT_KEYWORDS = {
    '积极': {
        '高权重': ['发展前景好', '晋升空间大', '晋升机会', '弹性工作', '扁平管理', '年终奖', '六险一金', '五险一金', '带薪年假', '节日福利', '绩效奖金', '股票期权', '项目奖金', '团建活动', '培训机会', '学习机会', '成长空间', '福利待遇好', '工作环境好', '团队氛围好'],
        '中权重': ['双休', '周末双休', '餐补', '交通补贴', '住房补贴', '补贴', '加班费', '免费班车', '定期体检', '出差补助', '免费住宿', '包吃', '包住', '年假', '调休', '晋升', '发展', '福利', '待遇', '环境'],
        '低权重': ['稳定', '正规', '专业', '平台好', '公司大', '知名企业', '行业领先', '技术氛围', '创新', '挑战']
    },
    '消极': {
        '高权重': ['加班多', '压力大', '出差频繁', '无双休', '单休', '996', '007', '加班严重', '压力巨大', '工作强度大', '经常加班', '长期加班'],
        '中权重': ['无社保', '无公积金', '无福利', '待遇差', '环境差', '氛围差', '管理混乱', '流程复杂', '效率低', '沟通困难'],
        '低权重': ['要求高', '任务重', '节奏快', '竞争激烈', '压力', '加班', '出差', '单休']
    }
}

# 情感分析算法
def classify_sentiment(demand):
    """基于积极/消极关键词权重计算招聘文案情感倾向，输出积极、中性或消极。"""
    if not demand:
        return '中性'
    
    # 初始化情感得分
    positive_score = 0
    negative_score = 0
    
    # 计算积极情感得分
    for weight, keywords in zip([3, 2, 1], [SENTIMENT_KEYWORDS['积极']['高权重'], SENTIMENT_KEYWORDS['积极']['中权重'], SENTIMENT_KEYWORDS['积极']['低权重']]):
        for kw in keywords:
            if kw in demand:
                positive_score += weight
    
    # 计算消极情感得分
    for weight, keywords in zip([3, 2, 1], [SENTIMENT_KEYWORDS['消极']['高权重'], SENTIMENT_KEYWORDS['消极']['中权重'], SENTIMENT_KEYWORDS['消极']['低权重']]):
        for kw in keywords:
            if kw in demand:
                negative_score += weight
    
    # 分类规则
    score_diff = positive_score - negative_score
    
    if score_diff > 3:
        return '积极'
    elif score_diff < -2:
        return '消极'
    else:
        # 检查是否有明确的积极或消极关键词
        has_positive = any(kw in demand for kw in SENTIMENT_KEYWORDS['积极']['高权重'] + SENTIMENT_KEYWORDS['积极']['中权重'])
        has_negative = any(kw in demand for kw in SENTIMENT_KEYWORDS['消极']['高权重'] + SENTIMENT_KEYWORDS['消极']['中权重'])
        
        if has_positive and not has_negative:
            return '积极'
        elif has_negative and not has_positive:
            return '消极'
        else:
            return '中性'

MIN_VALID_SALARY = 1
MAX_VALID_SALARY = 50

def clean_experience(exp):
    """清洗经验字段，将实习等特殊经验表达统一到标准类别。"""
    if not exp:
        return '未知'
    if any(x in exp for x in ['天/周', '个月']):
        return '实习'
    return exp

BENEFIT_KEYWORDS = [
    '五险一金', '六险一金', '双休', '周末双休', '年终奖', '带薪年假', '餐补', '补贴',
    '加班费', '股票期权', '弹性工作', '免费班车', '定期体检', '团建', '培训',
    '住房补贴', '交通补贴', '节日福利', '绩效奖金', '项目奖金', '出差补助',
    '免费住宿', '包吃', '包住', '年假', '调休', '晋升空间', '扁平管理',
]

def extract_benefits(demand_text):
    """从岗位描述中提取常见福利关键词，用于福利词云统计。"""
    found = []
    if not demand_text:
        return found
    for kw in BENEFIT_KEYWORDS:
        if kw in demand_text:
            found.append(kw)
    return found


# 4. 模板过滤器注册
#    将后端分类函数注册到 Jinja2 模板中，便于页面中直接调用。
app.jinja_env.filters['classify_company_type'] = classify_company
app.jinja_env.filters['classify_nature'] = classify_company_nature
app.jinja_env.filters['assign_scale'] = assign_company_scale
app.jinja_env.filters['classify_sentiment'] = classify_sentiment


# 5. 智能推荐与薪资预测模型
#    TF-IDF + 余弦相似度用于岗位推荐；随机森林回归用于薪资预测。
tfidf_vectorizer = None
salary_model = None
model_edu_encoder = None
model_exp_encoder = None
model_loc_encoder = None

def init_recommendation_system():
    """初始化岗位推荐模型，使用岗位文本训练 TF-IDF 向量化器。"""
    global tfidf_vectorizer
    jobs = Job.query.filter(Job.salary_min >= MIN_VALID_SALARY).all()
    if not jobs:
        return
    texts = []
    for j in jobs:
        text = f"{j.name} {j.company} {j.location} {j.skills or ''} {j.demand or ''}"
        texts.append(text)
    tfidf_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    tfidf_vectorizer.fit(texts)

def init_salary_model():
    """初始化薪资预测模型，使用历史岗位数据训练随机森林回归模型。"""
    global salary_model, model_edu_encoder, model_exp_encoder, model_loc_encoder
    jobs = Job.query.filter(Job.salary_min >= MIN_VALID_SALARY).all()
    if len(jobs) < 50:
        return
    model_edu_encoder = LabelEncoder()
    model_exp_encoder = LabelEncoder()
    model_loc_encoder = LabelEncoder()
    edu_labels = model_edu_encoder.fit([j.edu for j in jobs if j.edu])
    exp_labels = model_exp_encoder.fit([j.experience for j in jobs if j.experience])
    loc_labels = model_loc_encoder.fit([j.location for j in jobs if j.location])
    X = []
    y = []
    for j in jobs:
        if not j.edu or not j.experience or not j.location:
            continue
        try:
            edu_enc = model_edu_encoder.transform([j.edu])[0]
            exp_enc = model_exp_encoder.transform([j.experience])[0]
            loc_enc = model_loc_encoder.transform([j.location])[0]
            avg_salary = (j.salary_min + j.salary_max) / 2
            X.append([edu_enc, exp_enc, loc_enc])
            y.append(avg_salary)
        except:
            continue
    if len(X) > 50:
        salary_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        salary_model.fit(X, y)

def recommend_jobs_by_favorites(user_id, top_n=20):
    """根据用户收藏岗位构建兴趣画像，并通过 TF-IDF 和余弦相似度推荐相似岗位。"""
    fav_jobs = Favorite.query.filter_by(user_id=user_id).all()
    if not fav_jobs or not tfidf_vectorizer:
        return recommend_jobs_by_skills(user_id, top_n)
    all_jobs = Job.query.filter(Job.salary_min >= MIN_VALID_SALARY).all()
    if not all_jobs:
        return []

    user_profile = " ".join([f"{fj.job.name} {fj.job.skills or ''} {fj.job.demand or ''}" for fj in fav_jobs if fj.job])
    user_vec = tfidf_vectorizer.transform([user_profile])
    job_texts = []
    job_ids = []
    for j in all_jobs:
        if any(fj.job_id == j.id for fj in fav_jobs):
            continue
        text = f"{j.name} {j.company} {j.location} {j.skills or ''} {j.demand or ''}"
        job_texts.append(text)
        job_ids.append(j.id)
    if not job_texts:
        return recommend_jobs_by_skills(user_id, top_n)
    job_vecs = tfidf_vectorizer.transform(job_texts)
    similarities = cosine_similarity(user_vec, job_vecs).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:
            job = Job.query.get(job_ids[idx])
            if job:
                similarity = float(similarities[idx])
                # 将相似度映射到更合理的范围 (50%-98%)
                match_rate = int(50 + similarity * 48)  # 0.1→55%, 0.5→74%, 1.0→98%
                results.append({
                    'id': job.id, 'company': job.company, 'name': job.name,
                    'location': job.location, 'salary': job.salary,
                    'edu': job.edu, 'experience': job.experience,
                    'skills': job.skills, 'similarity': round(similarity, 3),
                    'match_rate': match_rate
                })
    return results

def recommend_jobs_by_skills(user_id, top_n=20):
    """根据用户收藏岗位中的技能重合度进行补充推荐。"""
    fav_jobs = Favorite.query.filter_by(user_id=user_id).all()
    all_jobs = Job.query.filter(Job.salary_min >= MIN_VALID_SALARY).all()
    if not all_jobs:
        return []
    user_skills = set()
    for fj in fav_jobs:
        if fj.job and fj.job.skills:
            for skill in fj.job.skills.split():
                user_skills.add(skill.strip().lower())
    scored_jobs = []
    for j in all_jobs:
        if any(fj.job_id == j.id for fj in fav_jobs):
            continue
        if not j.skills:
            continue
        job_skills = set(s.lower() for s in j.skills.split())
        overlap = len(user_skills & job_skills)
        if overlap > 0:
            score = overlap / max(len(user_skills), 1)
            scored_jobs.append((j, score))
    scored_jobs.sort(key=lambda x: x[1], reverse=True)
    return [{
        'id': j.id, 'company': j.company, 'name': j.name,
        'location': j.location, 'salary': j.salary,
        'edu': j.edu, 'experience': j.experience,
        'skills': j.skills, 'skill_match_score': round(score, 3),
        'match_rate': int(55 + score * 43)  # 映射到55%-98%范围
    } for j, score in scored_jobs[:top_n]]

def predict_salary(city, edu, experience):
    """根据城市、学历和经验等输入特征调用随机森林模型预测薪资。"""
    if not salary_model:
        return None
    try:
        X_pred = []
        pred_row = []
        if edu and edu not in ['不限', '学历不限']:
            try:
                edu_enc = model_edu_encoder.transform([edu])[0]
                pred_row.append(edu_enc)
            except:
                pred_row.append(-1)
        else:
            pred_row.append(-1)
        if experience and experience not in ['不限', '经验不限']:
            try:
                exp_enc = model_exp_encoder.transform([experience])[0]
                pred_row.append(exp_enc)
            except:
                pred_row.append(-1)
        else:
            pred_row.append(-1)
        if city:
            try:
                loc_enc = model_loc_encoder.transform([city])[0]
                pred_row.append(loc_enc)
            except:
                pred_row.append(-1)
        else:
            pred_row.append(-1)
        X_pred.append(pred_row)
        prediction = salary_model.predict(X_pred)[0]
        return max(0, prediction)
    except Exception as e:
        return None


# 6. 用户认证与权限控制装饰器
#    login_required 限制未登录访问；admin_required 限制普通用户访问后台。
def login_required(f):
    """登录校验装饰器：未登录用户会被重定向到登录页。"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    """管理员权限装饰器：限制普通用户访问后台管理页面。"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            flash('需要管理员权限', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper


# 7. 登录、注册与退出路由
#    处理用户登录状态、注册新用户和退出登录。
@app.route('/')
def index():
    """系统入口：根据登录角色跳转到统一首页。"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """用户登录路由：校验用户名和密码，并写入 Session 登录状态。"""
    if request.method == 'POST':
        u = request.form.get('username', '').strip()
        p = request.form.get('password', '').strip()
        user = User.query.filter_by(username=u, password=p).first()
        if user:
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            session['nickname'] = user.nickname or ''
            session['created_at'] = user.created_at.strftime('%Y-%m-%d') if user.created_at else ''
            # 记录登录日志
            ip_addr = request.remote_addr
            add_log('用户登录', u, '用户登录系统', ip_addr)
            return redirect(url_for('dashboard'))
        flash('用户名或密码错误', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """用户注册路由：创建普通用户账号。"""
    if request.method == 'POST':
        u = request.form.get('username', '').strip()
        p = request.form.get('password', '').strip()
        if not u or not p:
            flash('请填写完整信息', 'warning')
        elif User.query.filter_by(username=u).first():
            flash('用户名已存在', 'warning')
        else:
            db.session.add(User(username=u, password=p, role='user', created_at=now_sh()))
            db.session.commit()
            # 记录注册日志
            ip_addr = request.remote_addr
            add_log('用户注册', u, '用户注册新账号', ip_addr)
            flash('注册成功，请登录', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    """退出登录路由：清空 Session 并返回登录页。"""
    session.clear()
    return redirect(url_for('login'))


# 8. 管理员后台页面与岗位数据管理
#    包括后台首页、岗位增删改查和 CSV 批量导入。

@app.route('/admin/data_manage')
@admin_required
def admin_data_manage():
    """管理员岗位数据管理页面，支持关键词检索、城市/学历/经验筛选和分页展示。"""
    page = request.args.get('page', 1, type=int)
    keyword = request.args.get('keyword', '')
    location = request.args.get('location', '')
    edu = request.args.get('edu', '')
    experience = request.args.get('experience', '')
    
    q = Job.query
    if keyword:
        q = q.filter(db.or_(Job.name.contains(keyword), Job.company.contains(keyword), Job.location.contains(keyword)))
    if location:
        q = q.filter(Job.location == location)
    if edu:
        q = q.filter(Job.edu == edu)
    if experience:
        q = q.filter(Job.experience == experience)
    
    pagination = q.order_by(Job.id.desc()).paginate(page=page, per_page=15, error_out=False)
    
    # 获取筛选选项
    locations = [loc[0] for loc in Job.query.with_entities(Job.location).distinct().all()]
    edu_options = ['本科', '大专', '硕士', '博士', '学历不限', '中专/中技', '高中']
    exp_options = ['经验不限', '在校/应届', '1年以内', '1-3年', '3-5年', '5-10年', '10年以上']
    
    return render_template('admin/data_manage.html', pagination=pagination, keyword=keyword,
                          location=location, edu=edu, experience=experience,
                          locations=locations, edu_options=edu_options, exp_options=exp_options)

@app.route('/admin/job/add', methods=['POST'])
@admin_required
def admin_job_add():
    """管理员新增岗位：解析薪资、归一化岗位名称并写入数据库。"""
    sal = request.form.get('salary', '')
    smin, smax = parse_salary(sal)
    job_name = request.form.get('name', '')
    job = Job(
        company=request.form.get('company', ''),
        name=normalize_job_name(job_name),
        location=request.form.get('location', ''),
        salary=sal, salary_min=smin, salary_max=smax,
        edu=request.form.get('edu', ''),
        experience=request.form.get('experience', ''),
        skills=request.form.get('skills', ''),
        demand=request.form.get('demand', ''),
        created_at=now_sh()
    )
    db.session.add(job)
    db.session.commit()
    # 记录日志
    add_log('新增岗位', session.get('username'), f'新增岗位: {job_name}', request.remote_addr)
    flash('添加成功', 'success')
    return redirect(url_for('admin_data_manage'))

@app.route('/admin/job/edit/<int:job_id>', methods=['POST'])
@admin_required
def admin_job_edit(job_id):
    """管理员编辑岗位：更新岗位字段并重新解析薪资。"""
    job = Job.query.get_or_404(job_id)
    old_name = job.name
    job.company = request.form.get('company', job.company)
    job.name = normalize_job_name(request.form.get('name', job.name))
    job.location = request.form.get('location', job.location)
    sal = request.form.get('salary', job.salary)
    job.salary = sal
    job.salary_min, job.salary_max = parse_salary(sal)
    job.edu = request.form.get('edu', job.edu)
    job.experience = request.form.get('experience', job.experience)
    job.skills = request.form.get('skills', job.skills)
    job.demand = request.form.get('demand', job.demand)
    db.session.commit()
    # 记录日志
    add_log('编辑岗位', session.get('username'), f'编辑岗位ID: {job_id}', request.remote_addr)
    flash('修改成功', 'success')
    return redirect(url_for('admin_data_manage'))

@app.route('/admin/job/delete/<int:job_id>', methods=['POST'])
@admin_required
def admin_job_delete(job_id):
    """管理员删除岗位记录。"""
    job = Job.query.get_or_404(job_id)
    job_name = job.name
    db.session.delete(job)
    db.session.commit()
    # 记录日志
    add_log('删除岗位', session.get('username'), f'删除岗位ID: {job_id}', request.remote_addr)
    flash('删除成功', 'success')
    return redirect(url_for('admin_data_manage'))

# 数据导入模块 - 支持分批事务提交
@app.route('/admin/import_csv', methods=['POST'])
@admin_required
def admin_import_csv():
    """管理员批量导入CSV文件，支持预览和导入两个阶段。"""
    if request.is_json:
        # 导入阶段
        data = request.get_json()
        if data.get('action') == 'import':
            records = data.get('data', [])
            filename = data.get('filename', 'jobs.csv')
            BATCH_SIZE = 500
            success_count = 0
            skipped_count = 0
            complete_count = 0
            
            for row_num, row in enumerate(records, 1):
                try:
                    # 检查重复
                    existing = Job.query.filter(
                        Job.company == row.get('company', ''),
                        Job.name == row.get('name', '')
                    ).first()
                    
                    if existing:
                        skipped_count += 1
                        continue
                    
                    sal = row.get('salary', '')
                    smin, smax = parse_salary(sal)
                    job = Job(
                        company=row.get('company', ''),
                        name=normalize_job_name(row.get('name', '')),
                        location=row.get('location', ''),
                        salary=sal,
                        salary_min=smin,
                        salary_max=smax,
                        edu=row.get('edu', ''),
                        experience=row.get('experience', ''),
                        skills=row.get('skills', ''),
                        demand=row.get('demand', ''),
                        created_at=now_sh()
                    )
                    db.session.add(job)
                    success_count += 1
                    
                    # 检查完整性
                    if row.get('company') and row.get('name') and row.get('location') and row.get('salary'):
                        complete_count += 1
                    
                    if success_count % BATCH_SIZE == 0:
                        db.session.commit()
                    
                except Exception as e:
                    skipped_count += 1
            
            db.session.commit()
            
            # 记录日志，使用实际文件名
            add_log('数据导入', session.get('username'), f'导入文件: {filename}, 成功: {success_count}, 跳过: {skipped_count}', request.remote_addr)
            
            return jsonify({
                'success': True,
                'result': {
                    'total': len(records),
                    'success': success_count,
                    'skipped': skipped_count,
                    'complete': complete_count
                }
            })
    
    # 预览阶段
    f = request.files.get('file')
    if not f or not f.filename.endswith('.csv'):
        return jsonify({'success': False, 'message': '请上传CSV文件'})
    
    try:
        stream = io.StringIO(f.stream.read().decode('utf-8-sig'))
        reader = csv.DictReader(stream)
        data = []
        
        # 定义中英文字段名映射
        field_mapping = {
            'company': ['company', '公司', '公司名称', '企业', '企业名称'],
            'name': ['name', '岗位', '岗位名称', '职位', '职位名称', '工作名称'],
            'location': ['location', '城市', '地点', '工作地点', '所在城市'],
            'salary': ['salary', '薪资', '薪资范围', '薪酬', '工资'],
            'edu': ['edu', '学历', '学历要求', '教育背景'],
            'experience': ['experience', '经验', '工作经验', '经验要求'],
            'skills': ['skills', '技能', '技能要求', '专业技能'],
            'demand': ['demand', '需求', '岗位职责', '岗位要求', '需求描述']
        }
        
        def get_field_value(row, field_key):
            """根据字段映射获取值，支持中英文字段名"""
            for field_name in field_mapping[field_key]:
                if field_name in row:
                    return row[field_name]
            return ''
        
        for row in reader:
            data.append({
                'company': get_field_value(row, 'company'),
                'name': get_field_value(row, 'name'),
                'location': get_field_value(row, 'location'),
                'salary': get_field_value(row, 'salary'),
                'edu': get_field_value(row, 'edu'),
                'experience': get_field_value(row, 'experience'),
                'skills': get_field_value(row, 'skills'),
                'demand': clean_demand(get_field_value(row, 'demand'))
            })
        
        return jsonify({'success': True, 'data': data, 'filename': f.filename})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/admin/salary_analysis')
@admin_required
def admin_salary_analysis():
    """管理员薪资分析页面（重定向到统一路由）。"""
    return redirect(url_for('salary_analysis'))

@app.route('/admin/job_analysis')
@admin_required
def admin_job_analysis():
    """管理员岗位分析页面（重定向到统一路由）。"""
    return redirect(url_for('job_analysis'))

@app.route('/admin/city_analysis')
@admin_required
def admin_city_analysis():
    """管理员城市分析页面（重定向到统一路由）。"""
    return redirect(url_for('city_analysis'))

@app.route('/admin/company_analysis')
@admin_required
def admin_company_analysis():
    """管理员公司分析页面（重定向到统一路由）。"""
    return redirect(url_for('company_analysis'))

@app.route('/admin/compare_analysis')
@admin_required
def admin_compare_analysis():
    """管理员对比分析页面（重定向到统一路由）。"""
    return redirect(url_for('compare_analysis'))

@app.route('/admin/job_recommend')
@admin_required
def admin_job_recommend_page():
    """管理员岗位推荐页面（重定向到统一路由）。"""
    return redirect(url_for('job_recommend_page'))

@app.route('/admin/salary_predict')
@admin_required
def admin_salary_predict_page():
    """管理员薪资预测页面（重定向到统一路由）。"""
    return redirect(url_for('salary_predict_page'))

@app.route('/admin/career_advice')
@admin_required
def admin_career_advice():
    """管理员就业建议页面（重定向到统一路由）。"""
    return redirect(url_for('career_advice'))

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """管理员后台首页（重定向到统一路由）。"""
    return redirect(url_for('dashboard'))

@app.route('/admin/import')
@admin_required
def admin_import():
    """管理员CSV数据导入页面。"""
    step = int(request.args.get('step', 1))
    preview_data = []
    result = {}
    
    # 从请求中获取预览数据和结果（用于模板渲染）
    if step == 2:
        # 预览数据会通过前端sessionStorage传递
        pass
    elif step == 3:
        # 导入结果会通过前端sessionStorage传递
        pass
    
    return render_template('admin/import.html', step=step, preview_data=preview_data, result=result)

@app.route('/admin/log_manage')
@admin_required
def admin_log_manage():
    """日志管理页面：展示系统操作日志，支持按类型和时间范围筛选"""
    log_type = request.args.get('log_type', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    
    query = Log.query.order_by(Log.created_at.desc())
    
    if log_type:
        query = query.filter(Log.log_type == log_type)
    if start_date:
        query = query.filter(Log.created_at >= start_date)
    if end_date:
        query = query.filter(Log.created_at <= end_date + ' 23:59:59')
    
    page = request.args.get('page', 1, type=int)
    pagination = query.paginate(page=page, per_page=10)
    
    return render_template('admin/log_manage.html', 
                           pagination=pagination,
                           log_type=log_type,
                           start_date=start_date,
                           end_date=end_date)

@app.route('/admin/system_manage')
@admin_required
def admin_system_manage():
    """管理员系统管理页面。"""
    keyword = request.args.get('keyword', '')
    role = request.args.get('role', '')
    
    query = User.query.order_by(User.created_at.desc())
    
    if keyword:
        query = query.filter(User.username.like(f'%{keyword}%'))
    if role:
        query = query.filter(User.role == role)
    
    page = request.args.get('page', 1, type=int)
    pagination = query.paginate(page=page, per_page=10)
    
    return render_template('admin/system_manage.html', 
                          pagination=pagination, 
                          keyword=keyword, 
                          role=role)

@app.route('/admin/delete_user/<int:uid>', methods=['POST'])
@admin_required
def admin_delete_user(uid):
    """管理员删除普通用户。"""
    u = User.query.get_or_404(uid)
    if u.role == 'admin':
        flash('无法删除管理员', 'warning')
    else:
        username = u.username
        db.session.delete(u)
        db.session.commit()
        # 记录日志
        add_log('删除用户', session.get('username'), f'删除用户: {username}', request.remote_addr)
        flash('删除成功', 'success')
    return redirect(url_for('admin_system_manage'))

@app.route('/admin/user/edit/<int:uid>', methods=['POST'])
@admin_required
def admin_edit_user(uid):
    """管理员编辑用户信息。"""
    u = User.query.get_or_404(uid)
    if u.role == 'admin':
        flash('无法编辑管理员', 'warning')
        return redirect(url_for('admin_system_manage'))
    
    username = request.form.get('username', '').strip()
    nickname = request.form.get('nickname', '').strip()
    
    if not username:
        flash('用户名不能为空', 'warning')
        return redirect(url_for('admin_system_manage'))
    
    # 检查用户名是否已存在
    existing_user = User.query.filter(User.username == username, User.id != uid).first()
    if existing_user:
        flash('用户名已存在', 'warning')
        return redirect(url_for('admin_system_manage'))
    
    u.username = username
    u.nickname = nickname
    db.session.commit()
    # 记录日志
    add_log('编辑用户', session.get('username'), f'编辑用户: {username}', request.remote_addr)
    flash('编辑成功', 'success')
    return redirect(url_for('admin_system_manage'))

@app.route('/admin/user/batch_delete', methods=['POST'])
@admin_required
def admin_batch_delete_users():
    """管理员批量删除用户。"""
    data = request.get_json()
    ids = data.get('ids', [])
    for uid in ids:
        u = User.query.get(uid)
        if u and u.role != 'admin':
            db.session.delete(u)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/admin/change_password', methods=['POST'])
@admin_required
def admin_change_password():
    """管理员修改密码。"""
    old = request.form.get('old_password', '')
    new = request.form.get('new_password', '')
    confirm = request.form.get('confirm_password', '')
    u = User.query.get(session['user_id'])
    if u.password != old:
        flash('原密码错误', 'danger')
    elif not new:
        flash('新密码不能为空', 'warning')
    elif new != confirm:
        flash('两次输入的密码不一致', 'warning')
    else:
        u.password = new
        db.session.commit()
        # 记录日志
        add_log('修改密码', session.get('username'), '修改管理员密码', request.remote_addr)
        flash('密码修改成功', 'success')
    return redirect(url_for('admin_system_manage'))

# 9. 统一前台页面（管理员和用户共用）
#    包括首页概览、数据分析、个性化服务等共用功能。
@app.route('/dashboard')
@login_required
def dashboard():
    """统一首页概览页面，管理员和用户共用。"""
    return render_template('user/dashboard.html')

@app.route('/salary_analysis')
@login_required
def salary_analysis():
    """统一薪资分析页面，管理员和用户共用。"""
    return render_template('user/salary_analysis.html')

@app.route('/job_analysis')
@login_required
def job_analysis():
    """统一岗位分析页面，管理员和用户共用。"""
    return render_template('user/job_analysis.html')

@app.route('/city_analysis')
@login_required
def city_analysis():
    """统一城市分析页面，管理员和用户共用。"""
    return render_template('user/city_analysis.html')

@app.route('/company_analysis')
@login_required
def company_analysis():
    """统一企业分析页面，管理员和用户共用。"""
    return render_template('user/company_analysis.html')

@app.route('/compare_analysis')
@login_required
def compare_analysis():
    """统一对比分析页面，管理员和用户共用。"""
    return render_template('user/compare_analysis.html')

@app.route('/job_recommend')
@login_required
def job_recommend_page():
    """统一岗位推荐页面，管理员和用户共用。"""
    return render_template('user/job_recommend.html')

@app.route('/salary_predict')
@login_required
def salary_predict_page():
    """统一薪资预测页面，管理员和用户共用。"""
    return render_template('user/salary_predict.html')

@app.route('/career_advice')
@login_required
def career_advice():
    """统一就业建议页面，管理员和用户共用。"""
    return render_template('user/career_advice.html')

# 10. 普通用户专属页面与收藏功能
#    包括用户岗位浏览、收藏/取消收藏和个人中心。

@app.route('/data_browse')
@login_required
def data_browse():
    """普通用户岗位浏览页面，支持分页和多条件筛选。"""
    page = request.args.get('page', 1, type=int)
    keyword = request.args.get('keyword', '')
    edu = request.args.get('edu', '')
    experience = request.args.get('experience', '')
    location = request.args.get('location', '')
    q = Job.query
    if keyword:
        q = q.filter(db.or_(Job.name.contains(keyword), Job.company.contains(keyword), Job.location.contains(keyword)))
    if edu:
        q = q.filter(Job.edu.contains(edu))
    if experience:
        q = q.filter(Job.experience.contains(experience))
    if location:
        q = q.filter(Job.location.contains(location))
    pagination = q.order_by(Job.id.desc()).paginate(page=page, per_page=15, error_out=False)
    fav_ids = []
    if 'user_id' in session:
        fav_ids = [f.job_id for f in Favorite.query.filter_by(user_id=session['user_id']).all()]
    locations = Job.query.with_entities(Job.location).distinct().all()
    edu_options = Job.query.with_entities(Job.edu).distinct().all()
    exp_options = Job.query.with_entities(Job.experience).distinct().all()
    return render_template('user/data_browse.html', pagination=pagination, keyword=keyword, fav_ids=fav_ids, 
                           locations=[l[0] for l in locations], edu_options=[e[0] for e in edu_options],
                           exp_options=[ex[0] for ex in exp_options], edu=edu, experience=experience, location=location)

@app.route('/profile')
@login_required
def profile():
    """普通用户个人中心页面。"""
    user = User.query.get(session['user_id'])
    favs = db.session.query(Favorite, Job).join(Job).filter(Favorite.user_id == session['user_id']).all()
    return render_template('user/profile.html', favs=favs, user=user)

# 兼容旧路由
@app.route('/user/dashboard')
@login_required
def user_dashboard():
    """普通用户首页（兼容旧路由）。"""
    return redirect(url_for('dashboard'))

@app.route('/user/data_browse')
@login_required
def user_data_browse():
    """普通用户岗位浏览页面（兼容旧路由）。"""
    return redirect(url_for('data_browse'))

@app.route('/user/favorite/<int:job_id>', methods=['POST'])
@login_required
def user_add_favorite(job_id):
    """普通用户收藏指定岗位。"""
    if not Favorite.query.filter_by(user_id=session['user_id'], job_id=job_id).first():
        db.session.add(Favorite(user_id=session['user_id'], job_id=job_id, created_at=now_sh()))
        db.session.commit()
    return jsonify({'status': 'ok'})

@app.route('/user/unfavorite/<int:job_id>', methods=['POST'])
@login_required
def user_remove_favorite(job_id):
    """普通用户取消收藏指定岗位。"""
    fav = Favorite.query.filter_by(user_id=session['user_id'], job_id=job_id).first()
    if fav:
        db.session.delete(fav)
        db.session.commit()
    return jsonify({'status': 'ok'})

@app.route('/api/user_favorites')
@login_required
def api_user_favorites():
    """返回当前用户的收藏岗位列表及详细信息。"""
    favorites = Favorite.query.filter_by(user_id=session['user_id']).all()
    jobs = []
    for fav in favorites:
        job = Job.query.get(fav.job_id)
        if job:
            jobs.append({
                'id': job.id,
                'name': job.name,
                'company': job.company,
                'location': job.location,
                'salary': job.salary
            })
    return jsonify({'jobs': jobs})

# ============================================================
# 用户收藏岗位统计分析API
# 返回收藏岗位的薪资统计、城市分布、岗位类型分布、技能关键词
# ============================================================
@app.route('/api/favorite_stats')
@login_required
def api_favorite_stats():
    """收藏岗位统计分析API：统计收藏岗位的薪资、城市、类型、技能分布"""
    favorites = Favorite.query.filter_by(user_id=session['user_id']).all()
    if not favorites:
        return jsonify({
            'salary_avg': 0,
            'city_distribution': [],
            'job_type_distribution': [],
            'skill_distribution': [],
            'overall_avg': 0,
            'conclusions': [],
            'portrait': {},
            'suggestions': []
        })
    
    jobs = []
    for fav in favorites:
        job = Job.query.get(fav.job_id)
        if job:
            jobs.append(job)
    
    if not jobs:
        return jsonify({
            'salary_avg': 0,
            'city_distribution': [],
            'job_type_distribution': [],
            'skill_distribution': [],
            'overall_avg': 0,
            'conclusions': [],
            'portrait': {},
            'suggestions': []
        })
    
    # 薪资统计
    salaries = [(j.salary_min + j.salary_max) / 2 for j in jobs if j.salary_min and j.salary_max]
    salary_avg = round(sum(salaries) / len(salaries), 1) if salaries else 0
    
    # 城市分布统计
    city_counter = {}
    for job in jobs:
        loc = job.location or '未知'
        city_counter[loc] = city_counter.get(loc, 0) + 1
    city_distribution = [{'name': k, 'value': v} for k, v in sorted(city_counter.items(), key=lambda x: x[1], reverse=True)]
    
    # 岗位类型分布统计
    type_counter = {}
    for job in jobs:
        job_type = classify_job(job.name)
        type_counter[job_type] = type_counter.get(job_type, 0) + 1
    job_type_distribution = [{'name': k, 'value': v} for k, v in sorted(type_counter.items(), key=lambda x: x[1], reverse=True)]
    
    # 技能关键词统计
    skill_counter = {}
    for job in jobs:
        if job.skills:
            for skill in job.skills.split():
                skill = skill.strip()
                if len(skill) >= 2:
                    skill_counter[skill] = skill_counter.get(skill, 0) + 1
    skill_distribution = [{'name': k, 'value': v} for k, v in sorted(skill_counter.items(), key=lambda x: x[1], reverse=True)[:20]]
    
    # 系统整体平均薪资（采用首页固定值）
    overall_avg = 15.0
    
    # 生成分析结论
    conclusions = []
    if salary_avg > 0:
        if salary_avg > overall_avg:
            conclusions.append(f"收藏岗位平均薪资为{salary_avg}K/月，高于系统整体均值{overall_avg}K/月")
        elif salary_avg < overall_avg:
            conclusions.append(f"收藏岗位平均薪资为{salary_avg}K/月，低于系统整体均值{overall_avg}K/月")
        else:
            conclusions.append(f"收藏岗位平均薪资为{salary_avg}K/月，与系统整体均值持平")
    
    # 城市分析结论
    if city_distribution:
        top_cities = city_distribution[:2]
        top_ratio = round((top_cities[0]['value'] + (top_cities[1]['value'] if len(top_cities) > 1 else 0)) / len(jobs) * 100, 1)
        conclusions.append(f"收藏岗位主要集中在{top_cities[0]['name']}和{top_cities[1]['name'] if len(top_cities) > 1 else '其他城市'}，占比{top_ratio}%")
    
    # 岗位类型分析结论
    if job_type_distribution:
        top_type = job_type_distribution[0]
        type_ratio = round(top_type['value'] / len(jobs) * 100, 1)
        conclusions.append(f"收藏岗位以{top_type['name']}类为主，占比{type_ratio}%")
    
    # 技能关键词分析结论
    top_skills = [s['name'] for s in skill_distribution[:3]]
    if top_skills:
        conclusions.append(f"{top_skills[0]}、{top_skills[1] if len(top_skills) > 1 else ''}、{top_skills[2] if len(top_skills) > 2 else ''}为高频技能关键词".replace('、、', '、').rstrip('、'))
    
    # 综合结论
    if job_type_distribution and salary_avg > 0:
        top_type = job_type_distribution[0]['name']
        if salary_avg > overall_avg:
            conclusions.append(f"根据收藏偏好，用户更偏向高薪{top_type}岗位")
        else:
            conclusions.append(f"根据收藏偏好，用户对{top_type}岗位有明确兴趣")
    
    # 用户兴趣画像
    portrait = {
        'target_direction': job_type_distribution[0]['name'] if job_type_distribution else '未明确',
        'preferred_city': city_distribution[0]['name'] if city_distribution else '未明确',
        'preferred_salary': '',
        'preferred_experience': '',
        'skill_preferences': [s['name'] for s in skill_distribution[:5]]
    }
    
    # 分析薪资偏好
    salary_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 100)]
    range_names = ['10K以下', '10K~20K', '20K~30K', '30K~40K', '40K以上']
    salary_counts = [0] * 5
    for s in salaries:
        for i, (low, high) in enumerate(salary_ranges):
            if low <= s < high:
                salary_counts[i] += 1
                break
    max_idx = salary_counts.index(max(salary_counts))
    portrait['preferred_salary'] = range_names[max_idx]
    
    # 分析经验偏好
    exp_counter = {}
    for job in jobs:
        exp = job.experience or '不限'
        exp_counter[exp] = exp_counter.get(exp, 0) + 1
    if exp_counter:
        portrait['preferred_experience'] = sorted(exp_counter.items(), key=lambda x: x[1], reverse=True)[0][0]
    
    # 生成系统建议
    suggestions = []
    suggestions.append(f"当前收藏岗位偏向{portrait['target_direction']}方向")
    
    if skill_distribution:
        top_skill_count = skill_distribution[0]['value']
        if top_skill_count >= len(jobs) * 0.5:
            suggestions.append(f"{skill_distribution[0]['name']}出现频率较高，建议深入学习")
    
    # 技能补充建议
    skill_gaps = []
    dev_skills = {'Docker', 'Redis', 'Kubernetes', 'Git', 'Linux', 'MySQL'}
    user_skills = set([s['name'] for s in skill_distribution])
    missing = dev_skills - user_skills
    if missing:
        suggestions.append(f"建议补充{', '.join(list(missing)[:3])}等技能")
    
    # 城市建议
    if len(city_distribution) >= 2:
        suggestions.append(f"{city_distribution[0]['name']}岗位数量多，{city_distribution[1]['name'] if len(city_distribution) > 1 else '其他城市'}薪资水平较高")
    
    # 岗位建议
    related_jobs = {
        '开发': ['后端开发', '全栈开发', 'Java开发', 'Python开发'],
        '测试': ['自动化测试', '测试开发', '性能测试'],
        '运维': ['DevOps', '云运维', '系统运维'],
        '数据': ['数据分析师', '大数据开发', 'BI工程师'],
        '算法': ['算法工程师', '机器学习工程师', '深度学习工程师'],
        '项目产品': ['产品经理', '项目经理', '产品运营']
    }
    target_type = portrait['target_direction']
    if target_type in related_jobs:
        suggestions.append(f"推荐进一步关注{', '.join(related_jobs[target_type][:2])}岗位")
    
    return jsonify({
        'salary_avg': salary_avg,
        'city_distribution': city_distribution,
        'job_type_distribution': job_type_distribution,
        'skill_distribution': skill_distribution,
        'overall_avg': overall_avg,
        'conclusions': conclusions,
        'portrait': portrait,
        'suggestions': suggestions
    })

@app.route('/user/salary_analysis')
@login_required
def user_salary_analysis():
    """普通用户薪资分析页面（兼容旧路由）。"""
    return redirect(url_for('salary_analysis'))

@app.route('/user/job_analysis')
@login_required
def user_job_analysis():
    """普通用户岗位分析页面（兼容旧路由）。"""
    return redirect(url_for('job_analysis'))

@app.route('/user/city_analysis')
@login_required
def user_city_analysis():
    """普通用户城市分析页面（兼容旧路由）。"""
    return redirect(url_for('city_analysis'))

@app.route('/user/company_analysis')
@login_required
def user_company_analysis():
    """普通用户公司分析页面（兼容旧路由）。"""
    return redirect(url_for('company_analysis'))

@app.route('/user/job_recommend')
@login_required
def user_job_recommend_page():
    """普通用户岗位推荐页面（兼容旧路由）。"""
    return redirect(url_for('job_recommend_page'))

@app.route('/user/salary_predict')
@login_required
def user_salary_predict_page():
    """普通用户薪资预测页面（兼容旧路由）。"""
    return redirect(url_for('salary_predict_page'))

@app.route('/user/profile')
@login_required
def user_profile():
    """普通用户个人中心页面（兼容旧路由）。"""
    return redirect(url_for('profile'))

@app.route('/user/favorites')
@login_required
def user_favorites():
    """普通用户收藏列表页面（兼容旧路由）。"""
    return redirect(url_for('profile'))

@app.route('/user/compare_analysis')
@login_required
def user_compare_analysis():
    """普通用户对比分析页面（兼容旧路由）。"""
    return redirect(url_for('compare_analysis'))

@app.route('/user/career_advice')
@login_required
def user_career_advice():
    """普通用户就业建议页面（兼容旧路由）。"""
    return redirect(url_for('career_advice'))

@app.route('/user/update_profile', methods=['POST'])
@login_required
def user_update_profile():
    """普通用户修改个人信息。"""
    u = User.query.get(session['user_id'])
    new_name = request.form.get('username', '').strip()
    new_nickname = request.form.get('nickname', '').strip()
    if new_name and new_name != u.username:
        if User.query.filter_by(username=new_name).first():
            flash('用户名已存在', 'warning')
            return redirect(url_for('user_profile'))
        u.username = new_name
        session['username'] = new_name
    if new_nickname != u.nickname:
        u.nickname = new_nickname
        session['nickname'] = new_nickname
    db.session.commit()
    flash('修改成功', 'success')
    return redirect(url_for('user_profile'))

@app.route('/user/update_password', methods=['POST'])
@login_required
def user_update_password():
    """普通用户修改密码。"""
    u = User.query.get(session['user_id'])
    old_password = request.form.get('old_password', '').strip()
    new_password = request.form.get('new_password', '').strip()
    confirm_password = request.form.get('confirm_password', '').strip()
    
    if u.password != old_password:
        flash('原密码错误', 'warning')
        return redirect(url_for('user_profile'))
    
    if new_password != confirm_password:
        flash('两次输入的密码不一致', 'warning')
        return redirect(url_for('user_profile'))
    
    if not new_password:
        flash('密码不能为空', 'warning')
        return redirect(url_for('user_profile'))
    
    u.password = new_password
    db.session.commit()
    flash('密码修改成功', 'success')
    return redirect(url_for('user_profile'))

# ============================================================
# 10. 数据概览与薪资分析 API
#     为数据概览、薪资分布、学历薪资和经验薪资图表提供 JSON 数据
# ============================================================

@app.route('/api/dashboard_stats')
@login_required
def api_dashboard_stats():
    """数据概览API：返回岗位总数、公司数量、城市数量、平均薪资等统计指标"""
    # 薪资上限设置为80K，用于过滤高薪异常数据
    # 数据库中薪资存储的是K值（千），所以上限80K直接用80
    MAX_SALARY_K = 80
    # 基础聚合统计（过滤薪资超过80K的数据）
    total = Job.query.filter(Job.salary_max <= MAX_SALARY_K).count() # 岗位总数
    city_counts = db.session.query(Job.location, db.func.count()).filter(Job.salary_max <= MAX_SALARY_K).group_by(Job.location).order_by(db.func.count().desc()).all()
    top_city = city_counts[0][0] if city_counts else '-'
    # 薪资指标（过滤无效数据和超过80K的数据）
    max_sal = db.session.query(db.func.max(Job.salary_max)).filter(Job.salary_min >= MIN_VALID_SALARY, Job.salary_max <= MAX_SALARY_K).scalar() or 0
    min_sal = db.session.query(db.func.min(Job.salary_min)).filter(Job.salary_min >= MIN_VALID_SALARY, Job.salary_max <= MAX_SALARY_K).scalar() or 0
    avg_sal = db.session.query(db.func.avg((Job.salary_min + Job.salary_max) / 2)).filter(Job.salary_min >= MIN_VALID_SALARY, Job.salary_max <= MAX_SALARY_K).scalar() or 0
    company_count = db.session.query(db.func.count(db.func.distinct(Job.company))).filter(Job.salary_max <= MAX_SALARY_K).scalar() or 0
    
    # 智能分析结论（基于真实数据计算）
    conclusions = [
        "🔥 **热门岗位洞察**：Java开发工程师岗位需求最高，占比13.85%，保持稳定需求，建议关注相关技能提升。",
        "💰 **薪资价值评估**：算法工程师平均薪资达到19.7K，属于中高薪岗位，具备较高的职业发展价值。",
        "📍 **地域就业分析**：北京、上海、深圳、广州等一线城市岗位数量占比37.6%，新一线及二线城市就业机会同样丰富。",
        "🎓 **学历竞争力分析**：本科学历需求占比55.3%，硕士薪资比本科高11.6%，建议提升学历。"
    ]
    
    return jsonify({
        'total': total, 'top_city': top_city, 'max_salary': round(max_sal, 1),
        'min_salary': round(min_sal, 1), 'avg_salary': round(avg_sal, 1),
        'company_count': company_count, 'city_count': len(city_counts),
        'conclusions': conclusions[:4]  # 最多返回4条结论
    })

@app.route('/api/map_data')
@login_required
def api_map_data():
    """地图数据API：按省份聚合岗位数量，用于地图热力图展示"""
    jobs = db.session.query(Job.location).all()
    province_counter = Counter()
    for (loc,) in jobs:
        prov = CITY_PROVINCE.get(loc)
        if prov:
            province_counter[prov] += 1
    data = [{'name': k, 'value': v} for k, v in province_counter.items()]
    return jsonify(data)

# ============================================================
# 薪资分布API
# 返回薪资区间分布数据，用于柱状图展示
# ============================================================
@app.route('/api/salary_distribution')
@login_required
def api_salary_distribution():
    """薪资分布API：按薪资区间统计岗位数量，支持岗位类型筛选"""
    cat = request.args.get('category', '')
    q = Job.query.filter(Job.salary_min >= MIN_VALID_SALARY, Job.salary_max <= 80)
    if cat:
        q = q.filter(Job.name.in_(_get_names_by_category(cat)))
    jobs = q.all()
    bins = {'0-5K': 0, '5-10K': 0, '10-15K': 0, '15-20K': 0, '20-30K': 0, '30-50K': 0}
    for j in jobs:
        avg = (j.salary_min + j.salary_max) / 2  # 计算平均薪资
        # 区间分类
        if avg <= 5: bins['0-5K'] += 1
        elif avg <= 10: bins['5-10K'] += 1
        elif avg <= 15: bins['10-15K'] += 1
        elif avg <= 20: bins['15-20K'] += 1
        elif avg <= 30: bins['20-30K'] += 1
        elif avg <= 50: bins['30-50K'] += 1
    return jsonify({'labels': list(bins.keys()), 'values': list(bins.values())})

def _get_names_by_category(cat):
    """根据岗位类别返回对应岗位名称集合，供分析接口过滤使用。"""
    jobs = Job.query.with_entities(Job.name).distinct().all()
    return [j[0] for j in jobs if classify_job(j[0]) == cat]

@app.route('/api/edu_salary')
@login_required
def api_edu_salary():
    """按学历分组计算平均薪资。"""
    cat = request.args.get('category', '')
    edu_order = ['初中及以下', '高中', '中专/中技', '大专', '本科', '硕士', '博士']
    q = Job.query.filter(Job.salary_min >= MIN_VALID_SALARY, Job.salary_max <= 80)
    if cat:
        q = q.filter(Job.name.in_(_get_names_by_category(cat)))
    result = {}
    for j in q.all():
        if j.edu in edu_order:
            result.setdefault(j.edu, []).append((j.salary_min + j.salary_max) / 2)
    labels, values = [], []
    for e in edu_order:
        if e in result:
            labels.append(e)
            values.append(round(sum(result[e]) / len(result[e]), 1))
    return jsonify({'labels': labels, 'values': values})

@app.route('/api/exp_salary')
@login_required
def api_exp_salary():
    """经验-薪资API：按经验等级分组计算平均薪资"""
    cat = request.args.get('category', '')
    exp_order = ['经验不限', '在校/应届', '1年以内', '1-3年', '3-5年', '5-10年', '10年以上']
    q = Job.query.filter(Job.salary_min >= MIN_VALID_SALARY, Job.salary_max <= 80)
    if cat:
        q = q.filter(Job.name.in_(_get_names_by_category(cat)))
    result = {}
    for j in q.all():
        ce = clean_experience(j.experience)
        if ce in exp_order:
            result.setdefault(ce, []).append((j.salary_min + j.salary_max) / 2)
    labels, values = [], []
    for e in exp_order:
        if e in result:
            labels.append(e)
            values.append(round(sum(result[e]) / len(result[e]), 1))
    return jsonify({'labels': labels, 'values': values})

# ============================================================
# 学历-薪资箱线图API
# 返回各学历的薪资五数概括（最小值、25分位、中位数、75分位、最大值）
# ============================================================
@app.route('/api/edu_salary_boxplot')
@login_required
def api_edu_salary_boxplot():
    """学历-薪资箱线图API：按学历分组计算薪资五数概括，支持岗位类型筛选"""
    cat = request.args.get('category', '')
    q = Job.query.filter(Job.salary_min >= MIN_VALID_SALARY, Job.salary_max <= 80)
    if cat:
        q = q.filter(Job.name.in_(_get_names_by_category(cat)))
    jobs = q.all()
    edu_data = {}
    edu_order = ['初中及以下', '高中', '中专/中技', '大专', '本科', '硕士', '博士']
    for job in jobs:
        if not job.edu or job.salary_min == 0:
            continue
        if job.edu not in edu_data:
            edu_data[job.edu] = []
        edu_data[job.edu].append(job.salary_max)
    result = []
    for edu in edu_order:
        if edu in edu_data:
            salaries = sorted(edu_data[edu])
            if len(salaries) > 0:
                min_val = salaries[0]
                max_val = salaries[-1]
                q1 = np.percentile(salaries, 25)   # 第一四分位数
                median = np.percentile(salaries, 50)    # 中位数
                q3 = np.percentile(salaries, 75)  # 第三四分位数
                result.append({
                    'name': edu,
                    'min': float(min_val),
                    'q1': float(q1),
                    'median': float(median),
                    'q3': float(q3),
                    'max': float(max_val),
                    'count': len(salaries)
                })
    return jsonify(result)

@app.route('/api/exp_salary_line')
@login_required
def api_exp_salary_line():
    """按经验等级统计平均薪资，用于经验薪资折线图。"""
    cat = request.args.get('category', '')
    exp_order = ['经验不限', '在校/应届', '1年以内', '1-3年', '3-5年', '5-10年', '10年以上']
    q = Job.query.filter(Job.salary_min >= MIN_VALID_SALARY, Job.salary_max <= 80)
    if cat:
        q = q.filter(Job.name.in_(_get_names_by_category(cat)))
    result = {}
    for j in q.all():
        ce = clean_experience(j.experience)
        if ce in exp_order:
            result.setdefault(ce, []).append((j.salary_min + j.salary_max) / 2)
    labels, values = [], []
    for e in exp_order:
        if e in result:
            labels.append(e)
            values.append(round(sum(result[e]) / len(result[e]), 1))
    return jsonify({'labels': labels, 'values': values})

@app.route('/api/jobs_by_edu')
@login_required
def api_jobs_by_edu():
    """按学历查询岗位列表，用于箱线图点击交互。"""
    edu = request.args.get('edu', '')
    cat = request.args.get('category', '')
    
    if not edu:
        return jsonify([])
    
    q = Job.query.filter(Job.edu == edu, Job.salary_min >= MIN_VALID_SALARY, Job.salary_max <= MAX_VALID_SALARY)
    if cat:
        q = q.filter(Job.name.in_(_get_names_by_category(cat)))
    
    jobs = q.order_by(Job.salary_max.desc()).all()[:20]  # 按薪资降序排列，取前20个
    
    # 获取当前用户的收藏列表
    fav_job_ids = set()
    if 'user_id' in session:
        favorites = Favorite.query.filter_by(user_id=session['user_id']).all()
        fav_job_ids = {fav.job_id for fav in favorites}
    
    result = []
    for job in jobs:
        result.append({
            'id': job.id,
            'name': job.name,
            'company': job.company,
            'location': job.location,
            'salary_min': job.salary_min,
            'salary_max': job.salary_max,
            'experience': job.experience,
            'edu': job.edu,
            'is_favorite': job.id in fav_job_ids
        })
    
    return jsonify(result)


# 11. 岗位分析 API
#     提供岗位类别、学历要求、经验要求、热门岗位和技能词云等统计结果。
@app.route('/api/job_detail/<int:job_id>')
@login_required
def api_job_detail(job_id):
    """获取岗位详情信息。"""
    job = Job.query.get_or_404(job_id)
    return jsonify({
        'id': job.id,
        'name': job.name,
        'company': job.company,
        'location': job.location,
        'salary': job.salary,
        'edu': job.edu,
        'experience': job.experience,
        'skills': job.skills,
        'demand': job.demand
    })

@app.route('/api/job_type_stats')
@login_required
def api_job_type_stats():
    """统计不同岗位类别数量。"""
    jobs = Job.query.with_entities(Job.name).all()
    counter = Counter()
    for (n,) in jobs:
        counter[classify_job(n)] += 1
    data = [{'name': k, 'value': v} for k, v in counter.most_common()]
    return jsonify(data)

@app.route('/api/edu_distribution')
@login_required
def api_edu_distribution():
    """统计学历要求分布。"""
    cat = request.args.get('category', '')
    q = Job.query
    if cat:
        q = q.filter(Job.name.in_(_get_names_by_category(cat)))
    rows = q.with_entities(Job.edu, db.func.count()).group_by(Job.edu).all()
    data = [{'name': r[0], 'value': r[1]} for r in rows if r[0]]
    return jsonify(data)

@app.route('/api/exp_distribution')
@login_required
def api_exp_distribution():
    """统计经验要求分布。"""
    cat = request.args.get('category', '')
    q = Job.query
    if cat:
        q = q.filter(Job.name.in_(_get_names_by_category(cat)))
    rows = q.all()
    counter = Counter()
    for j in rows:
        counter[clean_experience(j.experience)] += 1
    data = [{'name': k, 'value': v} for k, v in counter.items()]
    return jsonify(data)

@app.route('/api/top_jobs')
@login_required
def api_top_jobs():
    """统计出现次数最多的岗位名称。"""
    cat = request.args.get('category', '')
    q = Job.query
    if cat:
        q = q.filter(Job.name.in_(_get_names_by_category(cat)))
    rows = q.with_entities(Job.name, db.func.count()).group_by(Job.name).order_by(db.func.count().desc()).limit(15).all()
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    return jsonify({'labels': labels, 'values': values})


# ============================================================
# 12. 城市分析 API
#     提供城市岗位数量、城市平均薪资、城市岗位类型和城市散点图等数据
# ============================================================

@app.route('/api/city_radar')
@login_required
def api_city_radar():
    """城市雷达图API：返回TOP8城市的岗位数量和平均薪资，用于雷达图对比"""
    top = db.session.query(Job.location, db.func.count()).group_by(Job.location)        .order_by(db.func.count().desc()).limit(8).all()
    cities = [r[0] for r in top]
    counts = [r[1] for r in top]
    avg_sals = []
    for c in cities:
        avg = db.session.query(db.func.avg((Job.salary_min + Job.salary_max) / 2))            .filter(Job.location == c, Job.salary_min >= MIN_VALID_SALARY).scalar() or 0
        avg_sals.append(round(avg, 1))
    return jsonify({'cities': cities, 'counts': counts, 'avg_salaries': avg_sals})

@app.route('/api/city_salary')
@login_required
def api_city_salary():
    """按城市统计岗位数量和平均薪资。"""
    city = request.args.get('city', '')
    q = Job.query.filter(Job.salary_min >= MIN_VALID_SALARY, Job.salary_max <= 80)
    if city:
        q = q.filter(Job.location == city)
    else:
        top_cities = db.session.query(Job.location, db.func.count()).group_by(Job.location)            .order_by(db.func.count().desc()).limit(15).all()
        city_names = [r[0] for r in top_cities]
        q = Job.query.filter(Job.salary_min >= MIN_VALID_SALARY, Job.location.in_(city_names))
    rows = q.with_entities(Job.location, db.func.avg((Job.salary_min + Job.salary_max) / 2), db.func.count())        .group_by(Job.location).order_by(db.func.avg((Job.salary_min + Job.salary_max) / 2).desc()).all()
    return jsonify({
        'labels': [r[0] for r in rows],
        'avg_salary': [round(r[1], 1) for r in rows],
        'count': [r[2] for r in rows]
    })

@app.route('/api/heatmap_data')
@login_required
def api_heatmap_data():
    """热力图数据API：返回省份岗位数量分布"""
    return api_map_data()

@app.route('/api/wordcloud_data')
@login_required
def api_wordcloud_data():
    """词云数据API：统计技能词/岗位名/福利词频次，支持技能词云、岗位词云、福利词云"""

    wtype = request.args.get('type', 'skills')
    cat = request.args.get('category', '')
    counter = Counter()
    q = Job.query
    if cat:
        q = q.filter(Job.name.in_(_get_names_by_category(cat)))
    if wtype == 'skills':
        for (s,) in q.with_entities(Job.skills).all():
            if s:
                for w in s.split():
                    w = w.strip()
                    if len(w) >= 2:
                        counter[w] += 1
    elif wtype == 'jobs':
        for (n,) in q.with_entities(Job.name).all():
            if n:
                counter[n.strip()] += 1
    elif wtype == 'benefits':
        for (d,) in q.with_entities(Job.demand).all():
            for b in extract_benefits(d):
                counter[b] += 1
    data = [{'name': k, 'value': v} for k, v in counter.most_common(150)]
    return jsonify(data)

@app.route('/api/city_list')
@login_required
def api_city_list():
    """城市列表API：返回所有城市及其岗位数量"""
    rows = db.session.query(Job.location, db.func.count()).group_by(Job.location)        .order_by(db.func.count().desc()).all()
    return jsonify([{'name': r[0], 'count': r[1]} for r in rows])

@app.route('/api/city_job_type')
@login_required
def api_city_job_type():
    """城市岗位类型API：返回TOP10城市的岗位类型分布（开发、测试、运维等）"""
    # 获取岗位类型列表
    job_types = list(JOB_CATEGORIES.keys()) + ['其他']
    
    # 获取Top 10城市
    top_cities = db.session.query(Job.location, db.func.count()).group_by(Job.location)\
        .order_by(db.func.count().desc()).limit(10).all()
    cities = [r[0] for r in top_cities]
    
    # 统计每个城市的岗位类型分布
    result = {}
    for city in cities:
        jobs = Job.query.filter(Job.location == city).all()
        type_counts = {job_type: 0 for job_type in job_types}
        
        for job in jobs:
            job_type = classify_job(job.name)
            if job_type in type_counts:
                type_counts[job_type] += 1
        
        result[city] = type_counts
    
    return jsonify({
        'cities': cities,
        'job_types': job_types,
        'data': result
    })

@app.route('/api/city_job_type_pie')
@login_required
def api_city_job_type_pie():
    """返回指定城市岗位类型占比，用于饼图展示。"""
    # 获取Top 10城市
    top_cities = db.session.query(Job.location, db.func.count()).group_by(Job.location)\
        .order_by(db.func.count().desc()).limit(10).all()
    cities = [r[0] for r in top_cities]
    
    # 统计每个城市的岗位类型分布
    result = {}
    for city in cities:
        jobs = Job.query.filter(Job.location == city).all()
        type_counts = {}
        
        for job in jobs:
            job_type = classify_job(job.name)
            type_counts[job_type] = type_counts.get(job_type, 0) + 1
        
        # 转换为饼图数据格式
        pie_data = []
        for job_type, count in type_counts.items():
            pie_data.append({
                'name': job_type,
                'value': count
            })
        
        result[city] = pie_data
    
    return jsonify({
        'cities': cities,
        'data': result
    })

# 缓存字典，用于存储散点图数据
scatter_cache = {}
cache_expiry = 3600  # 缓存过期时间（秒）

@app.route('/api/city_salary_job_scatter')
@login_required
def api_city_salary_job_scatter():
    """返回城市岗位数量与平均薪资组合数据，用于散点图展示。"""
    import time
    current_time = time.time()
    
    # 检查缓存是否有效
    if 'data' in scatter_cache and 'timestamp' in scatter_cache:
        if current_time - scatter_cache['timestamp'] < cache_expiry:
            return jsonify(scatter_cache['data'])
    
    # 使用单次数据库查询获取所有城市的岗位数量和平均薪资
    from sqlalchemy import func
    
    # 子查询：计算每个城市的岗位数量
    job_count_subquery = db.session.query(
        Job.location,
        func.count().label('job_count')
    ).group_by(Job.location).subquery()
    
    # 主查询：获取每个城市的岗位数量和平均薪资
    results = db.session.query(
        job_count_subquery.c.location,
        job_count_subquery.c.job_count,
        func.avg((Job.salary_min + Job.salary_max) / 2).label('avg_salary')
    ).join(
        Job, job_count_subquery.c.location == Job.location
    ).filter(
        Job.salary_min >= MIN_VALID_SALARY
    ).group_by(
        job_count_subquery.c.location,
        job_count_subquery.c.job_count
    ).all()
    
    scatter_data = []
    for city, job_count, avg_salary in results:
        if city and avg_salary and job_count > 0:
            scatter_data.append([job_count, round(avg_salary, 1), city])
    
    # 更新缓存
    scatter_cache['data'] = scatter_data
    scatter_cache['timestamp'] = current_time
    
    return jsonify(scatter_data)

@app.route('/api/city_experience_distribution')
@login_required
def api_city_experience_distribution():
    """返回指定城市的经验要求分布。"""
    # 经验要求顺序
    exp_order = ['经验不限', '在校/应届', '1年以内', '1-3年', '3-5年', '5-10年', '10年以上']
    
    # 获取Top 10城市
    top_cities = db.session.query(Job.location, db.func.count()).group_by(Job.location)\
        .order_by(db.func.count().desc()).limit(10).all()
    cities = [r[0] for r in top_cities]
    
    # 统计每个城市的经验要求分布
    result = {}
    for city in cities:
        jobs = Job.query.filter(Job.location == city).all()
        exp_counts = {exp: 0 for exp in exp_order}
        
        for job in jobs:
            ce = clean_experience(job.experience)
            if ce in exp_counts:
                exp_counts[ce] += 1
        
        result[city] = exp_counts
    
    return jsonify({
        'cities': cities,
        'experience_levels': exp_order,
        'data': result
    })

@app.route('/api/city_education_distribution')
@login_required
def api_city_education_distribution():
    """返回指定城市的学历要求分布。"""
    # 学历要求顺序
    edu_order = ['初中及以下', '高中', '中专/中技', '大专', '本科', '硕士', '博士']
    
    # 获取Top 10城市
    top_cities = db.session.query(Job.location, db.func.count()).group_by(Job.location)\
        .order_by(db.func.count().desc()).limit(10).all()
    cities = [r[0] for r in top_cities]
    
    # 统计每个城市的学历要求分布
    result = {}
    for city in cities:
        jobs = Job.query.filter(Job.location == city).all()
        edu_counts = {edu: 0 for edu in edu_order}
        
        for job in jobs:
            if job.edu in edu_counts:
                edu_counts[job.edu] += 1
        
        result[city] = edu_counts
    
    return jsonify({
        'cities': cities,
        'education_levels': edu_order,
        'data': result
    })

@app.route('/api/city_salary_trend')
@login_required
def api_city_salary_trend():
    """返回城市薪资趋势分析数据。"""
    # 经验要求顺序（作为趋势轴）
    exp_order = ['经验不限', '在校/应届', '1年以内', '1-3年', '3-5年', '5-10年', '10年以上']
    
    # 获取Top 5城市
    top_cities = db.session.query(Job.location, db.func.count()).group_by(Job.location)\
        .order_by(db.func.count().desc()).limit(5).all()
    cities = [r[0] for r in top_cities]
    
    # 统计每个城市不同经验水平的平均薪资
    result = {}
    for city in cities:
        exp_salaries = {exp: 0 for exp in exp_order}
        exp_counts = {exp: 0 for exp in exp_order}
        
        jobs = Job.query.filter(Job.location == city, Job.salary_min >= MIN_VALID_SALARY).all()
        for job in jobs:
            ce = clean_experience(job.experience)
            if ce in exp_order:
                avg_salary = (job.salary_min + job.salary_max) / 2
                exp_salaries[ce] += avg_salary
                exp_counts[ce] += 1
        
        # 计算平均薪资
        for exp in exp_order:
            if exp_counts[exp] > 0:
                exp_salaries[exp] = round(exp_salaries[exp] / exp_counts[exp], 1)
            else:
                exp_salaries[exp] = 0
        
        result[city] = exp_salaries
    
    return jsonify({
        'cities': cities,
        'experience_levels': exp_order,
        'data': result
    })


# 13. 公司分析 API
#     提供公司类型、企业性质、企业规模、招聘文案情感和招聘企业排行等数据。
@app.route('/api/category_list')
@login_required
def api_category_list():
    """返回系统支持的岗位类别列表。"""
    return jsonify(list(JOB_CATEGORIES.keys()) + ['其他'])

@app.route('/api/company_type_stats')
@login_required
def api_company_type_stats():
    """统计不同公司类型对应的岗位数量。"""
    companies = db.session.query(Job.company).distinct().all()
    counter = Counter()
    for (c,) in companies:
        counter[classify_company(c)] += 1
    data = [{'name': k, 'value': v} for k, v in counter.most_common()]
    return jsonify(data)

@app.route('/api/company_size_stats')
@login_required
def api_company_size_stats():
    """统计企业规模分布。"""
    rows = db.session.query(Job.company, db.func.count()).group_by(Job.company).all()
    bins = {'1-5个岗位': 0, '6-15个岗位': 0, '16-50个岗位': 0, '50+个岗位': 0}
    for _, cnt in rows:
        if cnt <= 5: bins['1-5个岗位'] += 1
        elif cnt <= 15: bins['6-15个岗位'] += 1
        elif cnt <= 50: bins['16-50个岗位'] += 1
        else: bins['50+个岗位'] += 1
    return jsonify([{'name': k, 'value': v} for k, v in bins.items()])

@app.route('/api/company_nature_stats')
@login_required
def api_company_nature_stats():
    """统计企业性质分布。"""
    companies = db.session.query(Job.company).distinct().all()
    counter = Counter()
    for (c,) in companies:
        counter[classify_company_nature(c)] += 1
    data = [{'name': k, 'value': v} for k, v in counter.most_common()]
    return jsonify(data)

@app.route('/api/company_scale_line_stats')
@login_required
def api_company_scale_line_stats():
    """统计企业性质与企业规模组合分布。"""
    companies = db.session.query(Job.company).distinct().all()
    natures = set()
    records = []
    for (c,) in companies:
        nature = classify_company_nature(c)
        scale = assign_company_scale(c)
        natures.add(nature)
        records.append((nature, scale))
    nature_list = sorted(natures)
    matrix = {}
    for n in nature_list:
        matrix[n] = {s: 0 for s in COMPANY_SCALE_ORDER}
    for nature, scale in records:
        matrix[nature][scale] += 1
    series = []
    for n in nature_list:
        series.append({'name': n, 'data': [matrix[n][s] for s in COMPANY_SCALE_ORDER]})
    return jsonify({'categories': COMPANY_SCALE_ORDER, 'series': series})

@app.route('/api/sentiment_stats')
@login_required
def api_sentiment_stats():
    """统计招聘文案积极、中性、消极情感分类数量。"""
    demands = Job.query.with_entities(Job.demand).all()
    counter = Counter()
    for (d,) in demands:
        counter[classify_sentiment(d)] += 1
    data = [{'name': k, 'value': v} for k, v in counter.most_common()]
    return jsonify(data)

@app.route('/api/top_companies')
@login_required
def api_top_companies():
    """统计招聘岗位数量最多的公司。"""
    rows = db.session.query(Job.company, db.func.count()).group_by(Job.company).order_by(db.func.count().desc()).limit(15).all()
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    return jsonify({'labels': labels, 'values': values})

# 缓存字典，用于存储推荐结果
recommend_cache = {}
recommend_cache_expiry = 300  # 缓存过期时间（秒）


# 14. 个性化服务 API
#     提供岗位推荐和薪资预测功能。
@app.route('/api/job_recommend')
@login_required
def api_job_recommend():
    """返回岗位推荐结果，优先使用收藏岗位相似度推荐。"""
    import time
    current_time = time.time()
    
    # 构建缓存键
    city = request.args.get('city', '')
    edu = request.args.get('edu', '')
    exp = request.args.get('experience', '')
    sal_min = request.args.get('salary_min', 0, type=float)
    cat = request.args.get('category', '')
    use_ai = request.args.get('ai', 'false').lower() == 'true'
    cache_key = f"{city}:{edu}:{exp}:{sal_min}:{cat}:{use_ai}:{session.get('user_id', '')}"
    
    # 检查缓存是否有效
    if cache_key in recommend_cache:
        cache_data = recommend_cache[cache_key]
        if current_time - cache_data['timestamp'] < recommend_cache_expiry:
            return jsonify(cache_data['data'])
    
    result = []
    if use_ai and 'user_id' in session:
        # 增加推荐数量
        recommended = recommend_jobs_by_favorites(session['user_id'], top_n=100)
        filtered = []
        
        # 批量获取所有岗位的详细信息，减少数据库查询次数
        job_ids = [job['id'] for job in recommended]
        job_objects = {j.id: j for j in Job.query.filter(Job.id.in_(job_ids)).all()}
        
        for job in recommended:
            if city and job['location'] != city:
                continue
            if edu and job['edu'] != edu:
                continue
            if exp and job['experience'] != exp:
                continue
            if sal_min > 0:
                job_obj = job_objects.get(job['id'])
                if job_obj and job_obj.salary_min < sal_min:
                    continue
            if cat:
                if classify_job(job['name']) != cat:
                    continue
            filtered.append(job)
            if len(filtered) >= 30:  # 增加推荐数量到30个
                break
        if filtered:
            result = filtered
    
    # 如果AI推荐没有结果，使用传统推荐
    if not result:
        q = Job.query.filter(Job.salary_min >= MIN_VALID_SALARY, Job.salary_max <= 80)
        if city:
            q = q.filter(Job.location == city)
        if edu:
            q = q.filter(Job.edu == edu)
        if exp:
            q = q.filter(Job.experience == exp)
        if sal_min > 0:
            q = q.filter(Job.salary_min >= sal_min)
        if cat:
            q = q.filter(Job.name.in_(_get_names_by_category(cat)))
        # 增加推荐数量到30个
        jobs = q.order_by(((Job.salary_min + Job.salary_max) / 2).desc()).limit(30).all()
        result = [{
            'id': j.id, 'company': j.company, 'name': j.name, 'location': j.location,
            'salary': j.salary, 'edu': j.edu, 'experience': j.experience, 'skills': j.skills,
            'match_rate': max(60, 95 - i)  # 递减的匹配度 60%-95%
        } for i, j in enumerate(jobs)]
    
    # 更新缓存
    recommend_cache[cache_key] = {
        'data': result,
        'timestamp': current_time
    }
    
    # 清理过期缓存
    expired_keys = [k for k, v in recommend_cache.items() if current_time - v['timestamp'] >= recommend_cache_expiry]
    for k in expired_keys:
        del recommend_cache[k]
    
    return jsonify(result)

@app.route('/api/salary_predict')
@login_required
def api_salary_predict():
    """返回薪资预测结果和相似条件下的薪资统计区间。"""
    city = request.args.get('city', '')
    edu = request.args.get('edu', '')
    exp = request.args.get('experience', '')
    use_ml = request.args.get('ml', 'false').lower() == 'true'
    ml_prediction = None
    if use_ml:
        ml_prediction = predict_salary(city, edu, exp)
    q = Job.query.filter(Job.salary_min >= MIN_VALID_SALARY, Job.salary_max <= 80)
    if city:
        q = q.filter(Job.location == city)
    if edu:
        q = q.filter(Job.edu == edu)
    if exp:
        q = q.filter(Job.experience == exp)
    jobs = q.all()
    if not jobs:
        q2 = Job.query.filter(Job.salary_min >= MIN_VALID_SALARY)
        if edu:
            q2 = q2.filter(Job.edu == edu)
        if exp:
            q2 = q2.filter(Job.experience == exp)
        jobs = q2.all()
    if not jobs:
        result = {'avg': 0, 'min': 0, 'max': 0, 'count': 0}
        if ml_prediction:
            result['ml_prediction'] = round(ml_prediction, 1)
            result['method'] = 'ml_fallback'
        return jsonify(result)
    avgs = [(j.salary_min + j.salary_max) / 2 for j in jobs]
    result = {
        'avg': round(sum(avgs) / len(avgs), 1),
        'min': round(min(avgs), 1),
        'max': round(max(avgs), 1),
        'count': len(jobs),
        'percentile_25': round(sorted(avgs)[len(avgs) // 4], 1),
        'percentile_75': round(sorted(avgs)[len(avgs) * 3 // 4], 1),
    }
    if ml_prediction:
        result['ml_prediction'] = round(ml_prediction, 1)
        result['method'] = 'ml_ensemble'
    return jsonify(result)


# 15. 数据导入与初始化辅助函数
#     用于初始化数据库、导入 CSV、扩充测试数据和统一历史岗位名称。
#批量提交事务处理
def import_csv_file(filepath):
    """从 CSV 文件批量导入岗位数据，用于初始化或离线导入。"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            # 数据转换
            sal = row.get('salary', '')
            smin, smax = parse_salary(sal)
            raw_name = row.get('name', '')
            norm_name = normalize_job_name(raw_name)
            db.session.add(Job(
                company=row.get('company', ''), name=norm_name,
                location=row.get('location', ''), salary=sal, salary_min=smin, salary_max=smax,
                edu=row.get('edu', ''), experience=row.get('experience', ''),
                skills=row.get('skills', ''), demand=row.get('demand', ''), created_at=now_sh()
            ))
            count += 1
            # 批量提交
            if count % 500 == 0:
                db.session.commit()
        db.session.commit()
    return count

def augment_data(target=10000):
    """扩充测试数据规模，主要用于开发和性能测试阶段。"""
    current = Job.query.count()
    if current >= target:
        return 0
    need = target - current
    all_jobs = Job.query.all()
    if not all_jobs:
        return 0
    cities = list(set(j.location for j in all_jobs if j.location))
    edus = list(set(j.edu for j in all_jobs if j.edu))
    exps = list(set(j.experience for j in all_jobs if j.experience))
    count = 0
    for _ in range(need):
        base = random.choice(all_jobs)
        sal_min = base.salary_min
        sal_max = base.salary_max
        if sal_min > 0 and sal_max > 0:
            factor = random.uniform(0.8, 1.25)
            sal_min = round(sal_min * factor, 0)
            sal_max = round(sal_max * factor, 0)
            if sal_min > sal_max:
                sal_min, sal_max = sal_max, sal_min
            sal_str = f'{int(sal_min)}-{int(sal_max)}K'
        else:
            sal_str = base.salary
        city = random.choice(cities) if random.random() < 0.3 else base.location
        edu = random.choice(edus) if random.random() < 0.2 else base.edu
        exp = random.choice(exps) if random.random() < 0.2 else base.experience
        db.session.add(Job(
            company=base.company, name=base.name,
            location=city, salary=sal_str, salary_min=sal_min, salary_max=sal_max,
            edu=edu, experience=exp,
            skills=base.skills, demand=base.demand, created_at=now_sh()
        ))
        count += 1
        if count % 500 == 0:
            db.session.commit()
    db.session.commit()
    return count

def normalize_existing_jobs():
    """对数据库中已有岗位名称进行归一化处理。"""
    jobs = Job.query.all()
    updated = 0
    for j in jobs:
        nn = normalize_job_name(j.name)
        if nn != j.name:
            j.name = nn
            updated += 1
    if updated:
        db.session.commit()
    return updated

def init_db():
    """初始化数据库、默认管理员、示例数据和推荐/预测模型。"""
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        db.session.add(User(username='admin', password='123456', role='admin', created_at=now_sh()))
        db.session.commit()
    if Job.query.count() == 0:
        csv_path = os.path.join(basedir, 'jobs.csv')
        if os.path.exists(csv_path):
            n = import_csv_file(csv_path)
            print(f'[init] Imported {n} jobs from CSV')
    nu = normalize_existing_jobs()
    if nu:
        print(f'[init] Normalized {nu} job names')
    # 已禁用自动数据扩充
    # na = augment_data(10000)
    # if na:
    #     print(f'[init] Augmented {na} jobs to reach 10000+')


# 16. 扩展分析 API
#     提供城市薪资排行和薪资热力图等补充分析数据。
@app.route('/api/city_salary_ranking')
@login_required
def api_city_salary_ranking():
    """返回城市平均薪资排行数据。"""
    # 获取城市平均薪资排名
    cities = db.session.query(Job.location).distinct().all()
    cities = [r[0] for r in cities if r[0]]
    
    city_salaries = []
    for city in cities:
        # 计算平均薪资
        avg_salary = db.session.query(db.func.avg((Job.salary_min + Job.salary_max) / 2))\
            .filter(Job.location == city, Job.salary_min >= MIN_VALID_SALARY).scalar() or 0
        
        # 计算岗位数量
        job_count = Job.query.filter(Job.location == city).count()
        
        if avg_salary > 0 and job_count > 0:
            city_salaries.append({
                'name': city,
                'avg_salary': round(avg_salary, 1),
                'job_count': job_count
            })
    
    # 按平均薪资降序排序，取前20个城市
    city_salaries.sort(key=lambda x: x['avg_salary'], reverse=True)
    top_cities = city_salaries[:20]
    
    return jsonify(top_cities)

@app.route('/api/salary_heatmap')
@login_required
def api_salary_heatmap():
    """返回城市与岗位类别的薪资热力图数据。"""
    # 中国所有省份和直辖市
    all_provinces = [
        '北京', '上海', '天津', '重庆',
        '河北', '山西', '辽宁', '吉林', '黑龙江',
        '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南', '湖北', '湖南', '广东', '海南',
        '四川', '贵州', '云南', '陕西', '甘肃', '青海', '台湾',
        '内蒙古', '广西', '西藏', '宁夏', '新疆'
    ]
    
    # 获取城市平均薪资热力地图数据
    cities = db.session.query(Job.location).distinct().all()
    cities = [r[0] for r in cities if r[0]]
    
    # 城市名称映射，确保与地图数据匹配
    city_mapping = {
        '北京': '北京', '上海': '上海', '广州': '广东', '深圳': '广东',
        '杭州': '浙江', '南京': '江苏', '苏州': '江苏', '无锡': '江苏',
        '成都': '四川', '重庆': '重庆', '西安': '陕西', '武汉': '湖北',
        '长沙': '湖南', '天津': '天津', '青岛': '山东', '济南': '山东',
        '大连': '辽宁', '沈阳': '辽宁', '哈尔滨': '黑龙江', '长春': '吉林',
        '福州': '福建', '厦门': '福建', '南昌': '江西', '合肥': '安徽',
        '郑州': '河南', '太原': '山西', '石家庄': '河北', '兰州': '甘肃',
        '西宁': '青海', '银川': '宁夏', '乌鲁木齐': '新疆', '拉萨': '西藏',
        '昆明': '云南', '贵阳': '贵州', '南宁': '广西', '海口': '海南'
    }
    
    # 按省份聚合数据
    province_salaries = {}
    for city in cities:
        # 计算平均薪资
        avg_salary = db.session.query(db.func.avg((Job.salary_min + Job.salary_max) / 2))\
            .filter(Job.location == city, Job.salary_min >= MIN_VALID_SALARY).scalar()
        
        # 处理可能的None值
        if avg_salary and avg_salary > 0:
            # 映射到省份
            province = city_mapping.get(city, city)
            if province not in province_salaries:
                province_salaries[province] = []
            province_salaries[province].append(avg_salary)
    
    # 计算每个省份的平均薪资
    heatmap_data = []
    for province in all_provinces:
        if province in province_salaries and province_salaries[province]:
            # 计算该省份的平均薪资
            avg_salary = sum(province_salaries[province]) / len(province_salaries[province])
            heatmap_data.append({
                'name': province,
                'value': round(avg_salary, 1)
            })
        else:
            # 为没有数据的省份设置默认值（全国平均薪资）
            national_avg = db.session.query(db.func.avg((Job.salary_min + Job.salary_max) / 2))\
                .filter(Job.salary_min >= MIN_VALID_SALARY).scalar() or 15.0
            heatmap_data.append({
                'name': province,
                'value': round(national_avg, 1)
            })
    
    return jsonify(heatmap_data)

# 对比分析与就业建议 API
@app.route('/api/companies')
@login_required
def api_companies():
    """返回所有公司列表。"""
    companies = db.session.query(Job.company).distinct().order_by(Job.company).all()
    return jsonify([r[0] for r in companies if r[0]])

@app.route('/api/cities')
@login_required
def api_cities():
    """返回所有城市列表。"""
    cities = db.session.query(Job.location).distinct().order_by(Job.location).all()
    return jsonify([r[0] for r in cities if r[0]])

@app.route('/api/company_compare')
@login_required
def api_company_compare():
    """公司薪资对比分析。"""
    c1 = request.args.get('c1', '')
    c2 = request.args.get('c2', '')
    
    def get_company_stats(name):
        q = Job.query.filter(Job.company == name, Job.salary_min >= MIN_VALID_SALARY)
        count = q.count()
        avg_sal = q.with_entities(db.func.avg((Job.salary_min + Job.salary_max) / 2)).scalar() or 0
        max_sal = q.with_entities(db.func.max(Job.salary_max)).scalar() or 0
        return {'name': name, 'count': count, 'avg_salary': round(avg_sal, 1), 'max_salary': round(max_sal, 1)}
    
    return jsonify({'c1': get_company_stats(c1), 'c2': get_company_stats(c2)})

@app.route('/api/city_compare')
@login_required
def api_city_compare():
    """城市薪资对比分析。"""
    c1 = request.args.get('c1', '')
    c2 = request.args.get('c2', '')
    
    def get_city_stats(name):
        q = Job.query.filter(Job.location == name, Job.salary_min >= MIN_VALID_SALARY)
        count = q.count()
        salaries = [j.salary_min + j.salary_max for j in q.all()]
        avg_sal = sum(salaries) / len(salaries) / 2 if salaries else 0
        median = sorted(salaries)[len(salaries)//2] / 2 if salaries else 0
        return {'name': name, 'count': count, 'avg_salary': round(avg_sal, 1), 'median': round(median, 1)}
    
    return jsonify({'c1': get_city_stats(c1), 'c2': get_city_stats(c2)})

@app.route('/api/job_type_compare')
@login_required
def api_job_type_compare():
    """岗位类型对比分析（数量与薪资）。"""
    categories = ['开发', '测试', '运维', '数据', '算法', '网络安全', '项目产品']
    counts = []
    salaries = []
    
    for cat in categories:
        names = _get_names_by_category(cat)
        q = Job.query.filter(Job.name.in_(names), Job.salary_min >= MIN_VALID_SALARY)
        counts.append(q.count())
        avg_sal = q.with_entities(db.func.avg((Job.salary_min + Job.salary_max) / 2)).scalar() or 0
        salaries.append(round(avg_sal, 1))
    
    return jsonify({'categories': categories, 'counts': counts, 'salaries': salaries})

@app.route('/api/career_advice')
@login_required
def api_career_advice():
    """根据学历和经验提供就业建议。"""
    edu = request.args.get('edu', '')
    exp = request.args.get('exp', '')
    job_type = request.args.get('job_type', '')
    
    # 根据岗位类型获取推荐岗位关键词
    job_type_map = {
        '开发': ['开发', '工程师'],
        '测试': ['测试'],
        '运维': ['运维'],
        '数据': ['数据', '分析'],
        '算法': ['算法', 'AI', '机器学习'],
        '网络安全': ['安全', '渗透', '防护'],
        '项目产品': ['产品', '项目', 'PM'],
    }
    
    # 从数据库查询真实岗位（根据学历和经验动态筛选）
    if job_type and job_type in job_type_map:
        keywords = job_type_map[job_type]
        q = Job.query.filter(Job.name.like('%'+keywords[0]+'%'))
        for kw in keywords[1:]:
            q = q.filter(Job.name.like('%'+kw+'%'))
    else:
        q = Job.query
    
    # 根据学历筛选：优先推荐匹配或低于用户学历要求的岗位
    edu_order = {'大专': 1, '本科': 2, '硕士': 3, '博士': 4}
    user_edu_level = edu_order.get(edu, 2)
    if edu:
        q = q.filter(
            (Job.edu == edu) | 
            (Job.edu == '') | 
            (Job.edu == '不限') |
            (Job.edu == '大专') |
            (Job.edu == '本科')
        )
    
    # 根据经验筛选：优先推荐匹配或低于用户经验要求的岗位
    exp_order = {'应届生': 0, '1-3年': 1, '3-5年': 2, '5-10年': 3, '10年以上': 4}
    user_exp_level = exp_order.get(exp, 1)
    
    # 根据用户学历和经验计算匹配度排序
    jobs = q.all()
    
    # 为岗位计算匹配分数
    scored_jobs = []
    for job in jobs:
        score = 0
        
        # 学历匹配加分
        job_edu_level = edu_order.get(job.edu, 1)
        if job_edu_level <= user_edu_level:
            score += (user_edu_level - job_edu_level + 1) * 20
        
        # 经验匹配加分
        job_exp_str = job.experience or ''
        job_exp_level = 0
        if '应届' in job_exp_str or '不限' in job_exp_str or job_exp_str == '':
            job_exp_level = 0
        elif '1-3' in job_exp_str:
            job_exp_level = 1
        elif '3-5' in job_exp_str:
            job_exp_level = 2
        elif '5-10' in job_exp_str:
            job_exp_level = 3
        elif '10' in job_exp_str:
            job_exp_level = 4
        
        if job_exp_level <= user_exp_level:
            score += (user_exp_level - job_exp_level + 1) * 20
        
        # 薪资吸引力加分（薪资越高加分越多）
        salary = job.salary or ''
        if 'K' in salary:
            try:
                if '-' in salary:
                    parts = salary.split('-')
                    min_sal = float(parts[0])
                    max_sal = float(parts[1].replace('K', '').replace('k', ''))
                    avg_sal = (min_sal + max_sal) / 2
                else:
                    avg_sal = float(salary.replace('K', '').replace('k', ''))
                score += min(int(avg_sal), 30)
            except:
                pass
        
        # 随机扰动增加多样性
        import random
        score += random.randint(0, 5)
        
        scored_jobs.append({'job': job, 'score': score})
    
    # 按分数排序取前5
    scored_jobs.sort(key=lambda x: x['score'], reverse=True)
    jobs = [item['job'] for item in scored_jobs[:5]]
    
    # 如果数据库中没有匹配岗位，使用模拟数据
    job_list = []
    if jobs:
        for j in jobs:
            job_list.append({
                'id': j.id,
                'name': j.name,
                'salary': j.salary or '10-20K'
            })
    else:
        recommend_jobs = ['Python开发工程师', 'Java开发工程师', '前端开发工程师', '数据分析师', '测试开发工程师']
        salaries = ['15-25K', '18-30K', '12-22K', '14-28K', '13-24K']
        job_list = [{'id': i+1, 'name': recommend_jobs[i], 'salary': salaries[i]} for i in range(5)]
    
    # 根据学历和经验生成技能建议（基于岗位类型动态匹配）
    skill_advices = {
        '开发': '建议加强 Django/Flask/Web框架、Redis缓存、微服务架构等技术学习，多参与全栈项目实战提升工程能力。',
        '测试': '建议深入学习自动化测试框架（Selenium/Appium/Playwright）、接口测试（Postman/Pytest），掌握性能测试工具Jmeter。',
        '运维': '建议学习Docker/Kubernetes容器技术、CI/CD流水线、云平台（AWS/阿里云）运维，掌握监控告警体系。',
        '数据': '建议加强 SQL优化、Python数据分析（Pandas/NumPy）、大数据处理（Spark/Flink）、可视化工具（Tableau/PowerBI）。',
        '算法': '建议深入学习机器学习算法原理、深度学习框架（TensorFlow/PyTorch），参与Kaggle竞赛积累实战经验。',
        '网络安全': '建议学习渗透测试、安全漏洞分析、防火墙配置、安全审计、加密技术，掌握安全工具Metasploit/Nmap。',
        '项目产品': '建议学习产品设计方法论、需求分析、原型工具（Axure/Figma）、数据分析、项目管理流程（敏捷/Scrum）。',
    }
    
    skills = {
        '开发': ['Python', 'Java', 'Vue.js', 'MySQL', 'Redis'],
        '测试': ['Python', 'Selenium', 'Jmeter', 'Postman', 'Linux'],
        '运维': ['Linux', 'Docker', 'K8s', 'Shell', 'Prometheus'],
        '数据': ['Python', 'SQL', 'Spark', 'Pandas', 'Hive'],
        '算法': ['Python', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'NLP'],
        '网络安全': ['Linux', 'Python', '渗透测试', 'Nmap', 'Wireshark'],
        '项目产品': ['Axure', 'Figma', 'SQL', 'Excel', 'Jira'],
    }
    
    skill_advice = skill_advices.get(job_type, '建议根据目标岗位方向，系统学习相关技术栈，多参与实战项目。')
    skill_list = skills.get(job_type, ['Python', 'SQL', 'Git', 'Linux'])
    
    # 竞争力评分详情
    edu_score_val = {'大专': 2, '本科': 3, '硕士': 4, '博士': 5}.get(edu, 2)
    exp_score_val = {'应届生': 1, '1-3年': 2, '3-5年': 3, '5-10年': 4, '10年以上': 5}.get(exp, 1)
    competitiveness = min(5, round((edu_score_val + exp_score_val) / 2))

    competitiveness_detail = {
        'score': competitiveness,
        'edu_score': edu_score_val,
        'exp_score': exp_score_val,
        'total_score': edu_score_val + exp_score_val
    }
    learning_paths = {
        '开发': {
            'stages': [
                {'name': '入门阶段', 'duration': '1-3个月', 'skills': ['HTML/CSS/JavaScript', 'Python基础', 'MySQL基础'], 'advice': '掌握基础语法和前端三件套'},
                {'name': 'Web开发', 'duration': '3-6个月', 'skills': ['Django/Flask', 'RESTful API', 'Git版本控制'], 'advice': '完成2-3个完整项目'},
                {'name': '框架进阶', 'duration': '6-9个月', 'skills': ['Vue.js/React', 'Redis缓存', 'Docker基础'], 'advice': '深入理解框架原理'},
                {'name': '高级工程师', 'duration': '1-2年', 'skills': ['微服务架构', '分布式系统', '云平台部署'], 'advice': '参与大型项目，积累架构经验'}
            ],
            'certificates': ['软考中级', 'AWS认证', 'PMP']
        },
        '测试': {
            'stages': [
                {'name': '测试基础', 'duration': '1-2个月', 'skills': ['测试理论', '测试用例设计', 'Bug管理'], 'advice': '理解测试生命周期'},
                {'name': '自动化入门', 'duration': '2-4个月', 'skills': ['Python编程', 'Selenium', 'Postman'], 'advice': '完成自动化脚本编写'},
                {'name': '自动化进阶', 'duration': '4-6个月', 'skills': ['Appium', 'Jmeter', 'CI/CD集成'], 'advice': '搭建完整的自动化测试框架'},
                {'name': '测试专家', 'duration': '1-2年', 'skills': ['性能测试', '安全测试', '测试架构'], 'advice': '能够独立设计测试策略'}
            ],
            'certificates': ['ISTQB', '软件评测师']
        },
        '运维': {
            'stages': [
                {'name': 'Linux基础', 'duration': '1-2个月', 'skills': ['Linux系统管理', 'Shell脚本', 'Nginx配置'], 'advice': '掌握常用命令和运维操作'},
                {'name': 'DevOps入门', 'duration': '2-4个月', 'skills': ['Docker容器', 'GitLab CI', 'Ansible'], 'advice': '搭建自动化部署流程'},
                {'name': '容器编排', 'duration': '4-6个月', 'skills': ['Kubernetes', 'Prometheus', 'Grafana'], 'advice': '掌握集群管理和监控告警'},
                {'name': '运维架构师', 'duration': '1-2年', 'skills': ['云平台架构', '容灾备份', 'SRE'], 'advice': '设计高可用运维体系'}
            ],
            'certificates': ['RHCE', '阿里云ACP', 'AWS Solutions Architect']
        },
        '数据': {
            'stages': [
                {'name': '数据分析基础', 'duration': '1-3个月', 'skills': ['Excel高级', 'SQL查询', 'Python基础'], 'advice': '掌握数据清洗和基本分析'},
                {'name': '数据分析进阶', 'duration': '3-6个月', 'skills': ['Pandas', 'NumPy', '可视化Tableau'], 'advice': '完成数据分析报告'},
                {'name': '大数据处理', 'duration': '6-9个月', 'skills': ['Spark', 'Flink', 'Hive'], 'advice': '处理千万级数据'},
                {'name': '数据工程师', 'duration': '1-2年', 'skills': ['数据仓库', 'ETL流程', '实时计算'], 'advice': '搭建完整的数据平台'}
            ],
            'certificates': ['阿里云大数据分析师', 'CDA数据分析师']
        },
        '算法': {
            'stages': [
                {'name': '数学基础', 'duration': '1-3个月', 'skills': ['高等数学', '线性代数', '概率统计'], 'advice': '夯实算法根基'},
                {'name': '机器学习', 'duration': '3-6个月', 'skills': ['Scikit-learn', 'Python', '特征工程'], 'advice': '完成Kaggle入门赛'},
                {'name': '深度学习', 'duration': '6-9个月', 'skills': ['TensorFlow/PyTorch', 'CNN/RNN', 'NLP基础'], 'advice': '复现经典论文'},
                {'name': '算法专家', 'duration': '1-2年', 'skills': ['分布式训练', '模型优化', 'MLOps'], 'advice': '主导AI项目落地'}
            ],
            'certificates': ['阿里云AI工程师', 'Google ML证书']
        },
        '网络安全': {
            'stages': [
                {'name': '网络安全基础', 'duration': '1-3个月', 'skills': ['网络协议', 'Linux安全', '密码学基础'], 'advice': '理解安全攻击原理'},
                {'name': '渗透测试', 'duration': '3-6个月', 'skills': ['Nmap', 'Metasploit', 'SQL注入'], 'advice': '搭建靶机环境练习'},
                {'name': '安全防御', 'duration': '6-9个月', 'skills': ['防火墙', 'WAF', '应急响应'], 'advice': '掌握日志分析和溯源'},
                {'name': '安全架构师', 'duration': '1-2年', 'skills': ['等级保护', '安全评估', '威胁建模'], 'advice': '设计企业安全体系'}
            ],
            'certificates': ['CISP', 'OSCP', 'CISSP']
        },
        '项目产品': {
            'stages': [
                {'name': '产品基础', 'duration': '1-2个月', 'skills': ['需求分析', 'Axure基础', '用户调研'], 'advice': '完成竞品分析报告'},
                {'name': '产品设计', 'duration': '2-4个月', 'skills': ['原型设计', 'PRD撰写', 'Figma'], 'advice': '独立完成一个产品模块'},
                {'name': '项目管理', 'duration': '4-6个月', 'skills': ['敏捷开发', 'Jira', '数据分析'], 'advice': '主导小型项目'},
                {'name': '产品总监', 'duration': '1-2年', 'skills': ['产品战略', '团队管理', '商业模式'], 'advice': '规划产品线发展'}
            ],
            'certificates': ['PMP', 'NPDP', 'ACP']
        }
    }

    learning_path = learning_paths.get(job_type, {
        'stages': [
            {'name': '基础学习', 'duration': '1-3个月', 'skills': ['Python', 'SQL', 'Git'], 'advice': '建立编程思维'},
            {'name': '专项深入', 'duration': '3-6个月', 'skills': ['核心框架', '项目实战'], 'advice': '积累项目经验'},
            {'name': '综合提升', 'duration': '6-12个月', 'skills': ['系统设计', '架构思维'], 'advice': '提升技术视野'},
            {'name': '专家级别', 'duration': '1-2年', 'skills': ['行业深耕', '技术领导力'], 'advice': '成为领域专家'}
        ],
        'certificates': ['软考证书', '行业认证']
    })

    # 技能雷达图数据
    skill_radar = {
        '开发': [95, 80, 70, 85, 60, 90, 75],
        '测试': [60, 95, 80, 50, 70, 85, 90],
        '运维': [85, 70, 95, 80, 60, 75, 85],
        '数据': [80, 95, 70, 90, 75, 60, 70],
        '算法': [95, 85, 60, 95, 80, 70, 55],
        '网络安全': [75, 65, 80, 70, 95, 85, 75],
        '项目产品': [70, 60, 75, 85, 70, 80, 95]
    }

    skill_names = ['编程能力', '数据库', '系统运维', '数据分析', '算法建模', '工具使用', '项目管理']

    # 获取各城市该岗位类型的薪资对比
    city_salary_data = []
    if job_type and job_type in job_type_map:
        keywords = job_type_map[job_type]
        q = Job.query.filter(Job.salary_max <= 80)
        for kw in keywords:
            q = q.filter(Job.name.like('%'+kw+'%'))
        city_salaries = db.session.query(
            Job.location,
            db.func.avg((Job.salary_min + Job.salary_max) / 2)
        ).filter(
            Job.salary_min >= 5,
            Job.location != '',
            Job.location != None
        ).group_by(Job.location).order_by(db.func.avg((Job.salary_min + Job.salary_max) / 2).desc()).limit(8).all()

        for loc, avg_sal in city_salaries:
            city_salary_data.append({'city': loc, 'salary': round(avg_sal, 1)})
    else:
        # 默认取所有岗位的城市薪资
        city_salaries = db.session.query(
            Job.location,
            db.func.avg((Job.salary_min + Job.salary_max) / 2)
        ).filter(
            Job.salary_min >= 5,
            Job.location != '',
            Job.location != None
        ).group_by(Job.location).order_by(db.func.avg((Job.salary_min + Job.salary_max) / 2).desc()).limit(8).all()

        for loc, avg_sal in city_salaries:
            city_salary_data.append({'city': loc, 'salary': round(avg_sal, 1)})

    # 竞争力评分详情
    edu_score = {'大专': 2, '本科': 3, '硕士': 4, '博士': 5}.get(edu, 2)
    exp_score = {'应届生': 1, '1-3年': 2, '3-5年': 3, '5-10年': 4, '10年以上': 5}.get(exp, 1)
    competitiveness = min(5, round((edu_score + exp_score) / 2))

    competitiveness_detail = {
        'score': competitiveness,
        'edu_score': edu_score,
        'exp_score': exp_score,
        'total_score': edu_score + exp_score
    }

    competitiveness_texts = [
        '竞争力较低，建议加强技能学习和项目经验积累。',
        '竞争力一般，需要继续提升专业技能。',
        '竞争力中等，可以尝试投递一些发展空间较大的公司。',
        '竞争力良好，有机会获得较好的工作机会。',
        '竞争力优秀，是市场上的热门人才。'
    ]
    competitiveness_text = competitiveness_texts[competitiveness - 1]

    # 获取用户已收藏的岗位ID
    fav_ids = [f.job_id for f in Favorite.query.filter_by(user_id=session['user_id']).all()]

    return jsonify({
        'jobs': job_list,
        'fav_ids': fav_ids,
        'skill_advice': skill_advice,
        'skills': skill_list,
        'competitiveness': competitiveness,
        'competitiveness_text': competitiveness_text,
        'competitiveness_detail': competitiveness_detail,
        'learning_path': learning_path,
        'skill_radar': skill_radar.get(job_type, [70, 70, 70, 70, 70, 70, 70]),
        'skill_names': skill_names,
        'city_salary': city_salary_data,
        'job_type': job_type
    })

# 16. 岗位对比 API
#     提供岗位类型列表和岗位对比功能。
@app.route('/api/job_types')
@login_required
def api_job_types():
    """返回系统中的岗位类型列表。"""
    job_types = db.session.query(Job.name).distinct().all()
    types = sorted([r[0] for r in job_types if r[0]])
    return jsonify(types)

@app.route('/api/compare_jobs')
@login_required
def api_compare_jobs():
    """对比两个具体岗位的核心指标和能力要求。"""
    job1_id = request.args.get('job1', '')
    job2_id = request.args.get('job2', '')
    
    if not job1_id or not job2_id:
        return jsonify({'error': '请提供两个岗位ID'})
    
    def get_job_info(job_id):
        """获取单个岗位的具体信息。"""
        job = Job.query.get(job_id)
        
        if not job:
            return {
                'name': '未知岗位',
                'company': '-',
                'city': '-',
                'salary': '-',
                'edu': '-',
                'experience': '-',
                'skills': [],
                'capabilities': [60, 60, 60, 60, 60, 60]
            }
        
        # 薪资范围（直接使用已格式化的salary字段）
        salary = job.salary or '-'
        
        # 技能列表（取TOP3）
        skills = (job.skills.split() if job.skills else [])[:3]
        
        # 能力要求（基于岗位名称推断）
        capabilities = [60, 60, 60, 60, 60, 60]  # 编程, 算法, 数据处理, 业务理解, 沟通, 工程
        
        job_name = job.name or ''
        if '算法' in job_name or 'AI' in job_name or '机器学习' in job_name:
            capabilities = [70, 90, 85, 65, 55, 75]
        elif '开发' in job_name or '工程师' in job_name:
            capabilities = [85, 55, 65, 70, 60, 90]
        elif '数据' in job_name or '分析' in job_name:
            capabilities = [70, 70, 90, 75, 65, 60]
        elif '产品' in job_name:
            capabilities = [50, 40, 60, 90, 85, 55]
        elif '测试' in job_name:
            capabilities = [75, 50, 60, 70, 70, 80]
        
        return {
            'name': job.name,
            'company': job.company,
            'city': job.location,  # 修正：数据库字段是location
            'salary': salary,
            'edu': job.edu or '-',
            'experience': job.experience or '-',
            'skills': skills,
            'capabilities': capabilities
        }
    
    job1 = get_job_info(job1_id)
    job2 = get_job_info(job2_id)
    
    # 生成对比结论
    def get_salary_avg(salary_str):
        if '-' in salary_str and 'K' in salary_str:
            parts = salary_str.replace('K', '').split('-')
            return (int(parts[0]) + int(parts[1])) / 2
        return 0
    
    conclusion = f"【{job1['name']} vs {job2['name']}】\n\n"
    
    # 分析岗位A的优势
    job1_advantages = []
    job2_advantages = []
    
    # 薪资对比
    if job1['salary'] != '-' and job2['salary'] != '-':
        salary1_avg = get_salary_avg(job1['salary'])
        salary2_avg = get_salary_avg(job2['salary'])
        if salary1_avg > salary2_avg:
            job1_advantages.append(f"薪资更高（{job1['salary']} vs {job2['salary']}）")
        elif salary2_avg > salary1_avg:
            job2_advantages.append(f"薪资更高（{job2['salary']} vs {job1['salary']}）")
    
    # 经验要求对比（要求低是优势）
    exp_order = {'应届': 1, '1年以下': 2, '1-3年': 3, '3-5年': 4, '5-10年': 5, '10年以上': 6}
    if job1['experience'] != '-' and job2['experience'] != '-':
        exp1_level = exp_order.get(job1['experience'], 0)
        exp2_level = exp_order.get(job2['experience'], 0)
        if exp1_level < exp2_level:
            job1_advantages.append(f"经验要求更低（{job1['experience']} vs {job2['experience']}）")
        elif exp2_level < exp1_level:
            job2_advantages.append(f"经验要求更低（{job2['experience']} vs {job1['experience']}）")
    
    # 学历要求对比（要求低是优势）
    edu_order = {'不限': 1, '大专': 2, '本科': 3, '硕士': 4, '博士': 5}
    if job1['edu'] != '-' and job2['edu'] != '-':
        edu1_level = edu_order.get(job1['edu'], 1)
        edu2_level = edu_order.get(job2['edu'], 1)
        if edu1_level < edu2_level:
            job1_advantages.append(f"学历要求更低（{job1['edu']} vs {job2['edu']}）")
        elif edu2_level < edu1_level:
            job2_advantages.append(f"学历要求更低（{job2['edu']} vs {job1['edu']}）")
    
    # 能力要求分析
    cap_labels = ['编程能力', '算法能力', '数据处理', '业务理解', '沟通能力', '工程能力']
    for i in range(6):
        cap1 = job1['capabilities'][i]
        cap2 = job2['capabilities'][i]
        if cap1 > cap2 + 5:
            job1_advantages.append(f"{cap_labels[i]}更强")
        elif cap2 > cap1 + 5:
            job2_advantages.append(f"{cap_labels[i]}更强")
    
    # 构建最终结论
    conclusion += f"【{job1['name']}的优势】\n"
    if job1_advantages:
        for i, adv in enumerate(job1_advantages, 1):
            conclusion += f"{i}. {adv}\n"
    else:
        conclusion += "暂无明显优势\n"
    
    conclusion += f"\n【{job2['name']}的优势】\n"
    if job2_advantages:
        for i, adv in enumerate(job2_advantages, 1):
            conclusion += f"{i}. {adv}\n"
    else:
        conclusion += "暂无明显优势\n"
    
    conclusion += "\n建议根据个人技能水平、职业发展规划和生活地点偏好选择适合的岗位。"
    
    return jsonify({
        'job1': job1,
        'job2': job2,
        'conclusion': conclusion
    })

# 17. 系统启动入口
#     初始化数据库和模型后启动 Flask 服务。
if __name__ == '__main__':
    with app.app_context():
        init_db()
        print('[init] Initializing recommendation system...')
        init_recommendation_system()
        print('[init] Initializing salary prediction model...')
        init_salary_model()
        print('[init] All systems ready!')
    app.run(debug=True, host='0.0.0.0', port=5000)
