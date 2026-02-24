import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import japanize_matplotlib
import random
import functools
from concurrent.futures import ThreadPoolExecutor


# 1. 数据准备（新的技能体系）
def prepare_data():
    skills = ['身心健康', '人格和志向', '价值观和信念', '业务能力', '经营实务的知识和发挥能力', '职务经历和业绩']

    position_requirements = {
        "总裁": {
            "必需技能": {s: 5 for s in skills if s != '业务能力'},
            "权重": {'身心健康': 0.2, '人格和志向': 0.25, '价值观和信念': 0.2, '业务能力': 0.1,
                     '经营实务的知识和发挥能力': 0.15, '职务经历和业绩': 0.1},
            "层级": 1,
            "说明": "全公司的愿景制定和经营决策是主要职责",
            "关键词": ["经营", "愿景", "判断力", "决断力"]
        },
        "经营企划本部长": {
            "必需技能": {s: 4 for s in skills},
            "权重": {'身心健康': 0.15, '人格和志向': 0.2, '价值观和信念': 0.15, '业务能力': 0.15,
                     '经营实务的知识和发挥能力': 0.25, '职务经历和业绩': 0.1},
            "层级": 2,
            "说明": "中长期经营计划的制定和执行管理",
            "关键词": ["计划", "情景", "分析", "战略"]
        },
        "事业部长": {
            "必需技能": {s: 4 for s in skills},
            "权重": {'身心健康': 0.15, '人格和志向': 0.15, '价值观和信念': 0.15, '业务能力': 0.25,
                     '经营实务的知识和发挥能力': 0.2, '职务经历和业绩': 0.1},
            "层级": 2,
            "说明": "事业单位的P&L责任和战略执行",
            "关键词": ["事业", "收益", "客户", "市场"]
        },
        "内部审计部长": {
            "必需技能":  {s: 3 for s in skills},
            "权重": {
                '价值观和信念': 0.3,
                '经营实务的知识和发挥能力': 0.25,
                '职务经历和业绩': 0.2,
                '身心健康': 0.1,
                '人格和志向': 0.1,
                '业务能力': 0.05
            },
            "层级": 3,
            "说明": "企业治理和合规性的彻底审计和管理",
            "关键词": ["审计", "合规性", "风险管理", "治理"]
        },
        "A事业部・东京分公司总经理": {
            "必需技能": {s: 3 for s in skills},
            "权重": {
                '业务能力': 0.3,
                '职务经历和业绩': 0.25,
                '身心健康': 0.15,
                '人格和志向': 0.15,
                '价值观和信念': 0.1,
                '经营实务的知识和发挥能力': 0.05
            },
            "层级": 3,
            "说明": "东京地区的事业战略执行和销售目标的达成管理",
            "关键词": ["地区战略", "销售管理", "团队统率", "市场开拓"]
        },
        # 其他职位定义
        "A事业部・大阪分公司总经理": {
            "必需技能": {s: 4 for s in skills},
            "权重": {
                '业务能力': 0.3, '职务经历和业绩': 0.25, '身心健康': 0.15,
                '人格和志向': 0.15, '价值观和信念': 0.1, '经营实务的知识和发挥能力': 0.05
            },
            "层级": 3,
            "说明": "大阪地区的事业展开和收益目标的达成管理",
            "关键词": ["地区管理", "收益管理", "据点运营"]
        },
        "A事业部・总部・销售统括部长": {
            "必需技能": {s: 4 for s in skills if s != '经营实务的知识和发挥能力'},
            "权重": {
                '业务能力': 0.35, '职务经历和业绩': 0.3, '身心健康': 0.15,
                '人格和志向': 0.1, '价值观和信念': 0.05, '经营实务的知识和发挥能力': 0.05
            },
            "层级": 3,
            "说明": "全事业部的销售战略制定和销售团队的统括管理",
            "关键词": ["销售战略", "销售管理", "团队培养"]
        },
        "A事业部・总部・技术统括部长": {
            "必需技能": {s: 4 for s in skills},
            "权重": {
                '业务能力': 0.4, '职务经历和业绩': 0.25, '身心健康': 0.1,
                '人格和志向': 0.1, '价值观和信念': 0.1, '经营实务的知识和发挥能力': 0.05
            },
            "层级": 3,
            "说明": "技术开发战略的制定和技术部门的综合管理",
            "关键词": ["技术开发", "创新", "研究管理"]
        },
        "A事业部・总部・海外统括部长": {
            "必需技能": {s: 4 for s in skills},
            "权重": {
                '价值观和信念': 0.3, '业务能力': 0.25, '职务经历和业绩': 0.2,
                '身心健康': 0.1, '人格和志向': 0.1, '经营实务的知识和发挥能力': 0.05
            },
            "层级": 3,
            "说明": "海外事业的展开战略制定和全球事业的统括",
            "关键词": ["全球战略", "跨文化应对", "海外展开"]
        },
        "B事业部・销售统括部长": {
            "必需技能": {s: 4 for s in skills},
            "权重": {
                '业务能力': 0.35, '职务经历和业绩': 0.3, '身心健康': 0.15,
                '人格和志向': 0.1, '价值观和信念': 0.05, '经营实务的知识和发挥能力': 0.05
            },
            "层级": 3,
            "说明": "B事业部整体的销售战略制定和销售目标的达成管理",
            "关键词": ["销售战略", "客户管理", "销售促进"]
        },
        "B事业部・技术统括部长": {
            "必需技能": {s: 4 for s in skills},
            "权重": {
                '业务能力': 0.4, '职务经历和业绩': 0.25, '身心健康': 0.1,
                '人格和志向': 0.1, '价值观和信念': 0.1, '经营实务的知识和发挥能力': 0.05
            },
            "层级": 3,
            "说明": "B事业部的技术开发战略和技术团队的综合管理",
            "关键词": ["技术管理", "开发战略", "质量管理"]
        },
        "B事业部・开发统括部长": {
            "必需技能": {s: 4 for s in skills},
            "权重": {
                '业务能力': 0.45, '职务经历和业绩': 0.25, '身心健康': 0.1,
                '人格和志向': 0.1, '价值观和信念': 0.05, '经营实务的知识和发挥能力': 0.05
            },
            "层级": 3,
            "说明": "新产品开发战略的制定和开发过程的优化",
            "关键词": ["产品开发", "项目管理", "创新"]
        },
        "海外子公司总经理": {
            "必需技能": {s: 4 for s in skills},
            "权重": {
                '价值观和信念': 0.3, '业务能力': 0.25, '职务经历和业绩': 0.2,
                '身心健康': 0.15, '人格和志向': 0.1, '经营实务的知识和发挥能力': 0.05
            },
            "层级": 3,
            "说明": "海外子公司的经营管理和当地战略的执行统筹",
            "关键词": ["全球经营", "本地适应", "海外据点管理"]
        },
        "学习力者": {
            "必需技能": {s: 4 for s in skills},
            "权重": {
                '价值观和信念': 0.3, '业务能力': 0.25, '职务经历和业绩': 0.2,
                '身心健康': 0.15, '人格和志向': 0.1, '经营实务的知识和发挥能力': 0.05
            },
            "层级": 3,
            "说明": "学习新技术和愿景，并将其应用于经营战略",
            "关键词": ["学习", "新技术", "愿景"]
        }
    }

    # 強化されたフィードバックライブラリ（新しいスキル体系に合わせて更新）
    # 强化的反馈库（根据新的技能体系更新）
    feedback_library = {
        '身心健康': [
            "全年没有请病假，保持了高水平的表现",
            "在压力环境下始终保持冷静，稳定了团队",
            "严格管理健康，在公司马拉松比赛中获胜，体力出众",
            "实践工作与生活的平衡，展示了高生产力的工作方式"
        ],
        '人格和志向': [
            "无私奉献于公司的发展，受到全体员工的尊敬",
            "拥有强烈的社会贡献志向，积极推动CSR活动",
            "始终以公平的立场判断事物，受到所有人的信赖",
            "明确描述未来十年的愿景，激励周围的人"
        ],
        '价值观和信念': [
            "坚持‘客户第一’的信念，在困难局面中保持伦理观",
            "体现公司的核心价值观，成为新员工的榜样",
            "在不确定的情况下也有明确的判断标准，行为一致",
            "实践多样性和包容性，充分利用多样化的人才"
        ],
        '业务能力': [
            "在负责的工作中始终取得高成果，对部门目标的达成做出重大贡献",
            "改善复杂的业务流程，使部门整体效率提高了30%",
            "拥有深厚的专业知识，多次成功解决困难问题",
            "迅速适应新工作，短时间内成为高生产力的成员"
        ],
        '经营实务的知识和发挥能力': [
            "深入理解财务报表，提出适当的资源分配建议，改善收益",
            "参与高层管理决策，提供建设性意见",
            "负责评估并购项目，进行适当的企业价值评估",
            "擅长制定将经营战略付诸实施的具体行动计划"
        ],
        '职务经历和业绩': [
            "在过去五年中经历了三个不同部门，并在每个部门都取得了成绩",
            "成功启动新业务，三年内实现了10亿日元的销售额",
            "有海外派遣经验，领导国际项目的业绩",
            "担任行业协会的委员，在社外有广泛的网络"
        ],
        '潜在能力': [
            "虽然只发挥了一部分，但如果有适当的指导，可能会有飞跃性的成长",
            "迅速适应新工作，短时间内成为高生产力的成员",
            "在面对困难问题时表现出的韧性非同寻常",
            "自学能力强，积极在工作之外取得相关资格"
        ],
        '业绩': [
            "在XX项目中，在预算减少15%的情况下提前完成了计划",
            "在新客户开发中，通过独特的方法使年销售额增加了25%",
            "引入提高团队生产力的措施，使项目执行时间平均缩短了30%"
        ],
        '成长': [
            "入职三年，已经担任了五名后辈员工的导师",
            "集中改善了不擅长的演讲技巧，现在在公司比赛中获胜",
            "英语能力有问题，但通过自学将TOEIC分数提高了300分，参与了国际项目"
        ]
    }
    # 生成150名员工数据
    np.random.seed(42)
    data = []
    departments = ['销售', '开发', '生产', '人事', '经营企划']

    for i in range(150):
        dept = np.random.choice(departments)
        emp = {
            'ID': f'E{i + 1:03d}',
            '姓名': f'员工{i + 1}',
            '年龄': np.random.randint(30, 56),
            '部门': dept,
            '经验年数': np.random.randint(3, 26),
            '表现': np.clip(np.random.normal(3.8, 0.8), 1, 5),
            '潜在力': np.clip(np.random.normal(0.7, 0.15), 0.3, 1.0),
            '成长意愿': np.clip(np.random.normal(0.8, 0.1), 0.5, 1.0),
            '性格特性': random.choice(['慎重', '积极', '协调性', '分析性', '创造性', '热情', '冷静', '挑战性'])
        }

        # 基于新的技能体系进行评价
        for skill in skills:
            base = np.random.normal(3.8, 0.8) * (0.7 + emp['潜在力'] * 0.5)
            if dept == '经营企划' and skill in ['经营实务的知识和发挥能力', '价值观和信念']:
                base += 1.5
            elif dept == '销售' and skill in ['业务能力', '职务经历和业绩']:
                base += 1.0
            elif dept == '开发' and skill in ['业务能力', '身心健康']:
                base += 0.7
            emp[skill] = np.clip(base, 1, 5)

        # 生成反馈（根据新的技能体系更新）
        feedback_parts = []

        # 1. 基于技能的评论（从强项技能中选择2条）
        strong_skills = [s for s in skills if emp[s] >= 4]
        if len(strong_skills) >= 2:
            selected_skills = np.random.choice(strong_skills, 2, replace=False)
            for skill in selected_skills:
                feedback_parts.append(random.choice(feedback_library[skill]))

        # 2. 潜在能力评论（60%的概率添加）
        if random.random() < 0.6:
            feedback_parts.append(random.choice(feedback_library['潜在能力']))

        # 3. 性格特性评论
        feedback_parts.append(f"性格特性: {emp['性格特性']}，{random.choice(feedback_library['人格和志向'])}")

        # 4. 业绩或成长故事（添加其中一条）
        feedback_parts.append(random.choice([
            "【业绩】" + random.choice(feedback_library['业绩']),
            "【成长】" + random.choice(feedback_library['成长'])
        ]))

        # 随机打乱反馈顺序
        random.shuffle(feedback_parts)
        emp['反馈'] = "■ " + "\n■ ".join(feedback_parts)
        data.append(emp)

        # 验证技能生成逻辑
        print("生成的技能值统计:")
        for skill in skills:
            values = [emp[skill] for emp in data]
            # print(f"{skill}: 平均={np.mean(values):.2f} 最大={np.max(values):.2f} 最小={np.min(values):.2f}")

    # 将数据写入到excel文件中
    df = pd.DataFrame(data)
    df.to_excel('data.xlsx', index=False)

    return pd.DataFrame(data), skills, position_requirements


# 2. 推荐引擎（支持新的技能体系）
class SuccessionPlanner:
    def __init__(self, df, skills, position_reqs):
        self.df = df
        self.skills = skills
        self.position_reqs = position_reqs

        # 初始化缩放器
        self.scalers = {
            'skill': MinMaxScaler().fit(df[skills]),
            'performance': MinMaxScaler().fit(df[['表现']])
        }

        # 文本分析设置
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=500,
            stop_words=['が', 'を', 'に', 'の', 'は', 'で', 'た'],
            token_pattern=r'(?u)\b\w+\b'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(df['反馈'])

        # 技能关键词映射
        self.skill_keywords = {
            '身心健康': ['健康', '体力', '压力', '耐力', '持久力', '心理'],
            '人格和志向': ['人格', '志向', '信念', '伦理', '理念', '领导力'],
            '价值观和信念': ['价值观', '信念', '伦理', '判断', '标准', '哲学'],
            '业务能力': ['业务', '效率', '生产力', '专业', '执行', '技能'],
            '经营实务的知识和发挥能力': ['经营', '财务', '战略', '决策', '并购', '治理'],
            '职务经历和业绩': ['业绩', '经验', '项目', '成果', '职业', '实务']
        }

        # 培养计划
        self.development_plans = {
            '身心健康': [
                "健康管理计划（6个月）",
                "压力管理培训",
                "心理健康训练"
            ],
            '人格和志向': [
                "领导力哲学培训",
                "高管指导计划",
                "社会贡献项目领导经验"
            ],
            '价值观和信念': [
                "企业理念深化研讨会",
                "高管伦理培训",
                "多样性培训"
            ],
            '业务能力': [
                "专业领域高级培训",
                "业务流程改进项目",
                "跨部门轮岗"
            ],
            '经营实务的知识和发挥能力': [
                "经营模拟计划",
                "财务分析专家培训课程",
                "董事会观察员经验"
            ],
            '职务经历和业绩': [
                "内部公开招聘项目参与",
                "海外派遣计划",
                "新业务启动任务小组"
            ]
        }

    @functools.lru_cache(maxsize=32)
    def get_text_similarity(self, position):
        """文本相似度计算（带缓存）"""
        reqs = self.position_reqs[position]
        pos_keywords = " ".join(reqs.get("关键词", []))
        pos_vector = self.vectorizer.transform([pos_keywords])
        return cosine_similarity(self.tfidf_matrix, pos_vector)

    def extract_potential(self, feedback):
        """强化的潜在能力提取逻辑"""
        signals = [
            ('可能性', 0.3), ('潜力', 0.3), ('成长', 0.2),
            ('提升空间', 0.4), ('适应力', 0.2), ('期望', 0.3),
            ('飞跃', 0.4), ('素质', 0.3), ('前景', 0.3)
        ]

        score = 0
        for keyword, weight in signals:
            if keyword in feedback:
                score += weight

        # 具体事例的有无
        if any(word in feedback for word in ['业绩', '成果', '达成', '成功']):
            score += 0.2

        # 成长故事
        if any(word in feedback for word in ['改进', '提升', '掌握', '成长']):
            score += 0.2

        return min(1.0, score)

    def recommend_candidates(self, position, top_n=3):
        try:
            reqs = self.position_reqs[position]
            print(f"\n=== 开始推荐 {position} ===")

            # 复制所有候选人
            candidates_df = self.df.copy()

            # 计算技能不足度（与必需技能的差距）
            skill_shortfalls = []
            for skill, min_level in reqs["必需技能"].items():
                candidates_df[f'{skill}_不足'] = np.clip(min_level - 0.5 - candidates_df[skill], 0, None)
                skill_shortfalls.append(f'{skill}_不足')

            # 计算技能评分
            skill_weights = np.array([reqs["权重"].get(s, 0) for s in self.skills])
            candidates_df['技能评分'] = candidates_df[self.skills].values.dot(skill_weights)

            # 缩放
            candidates_df['技能评分_缩放'] = self.scalers['skill'].transform(candidates_df[self.skills]).mean(axis=1)
            candidates_df['表现_缩放'] = self.scalers['performance'].transform(candidates_df[['表现']]).flatten()

            # 文本匹配评分
            text_sim = self.get_text_similarity(position)
            candidates_df['文本评分'] = text_sim[candidates_df.index].flatten()

            # 潜在能力评分
            candidates_df['潜在评分'] = candidates_df['反馈'].apply(self.extract_potential)

            # 计算综合评分（应用技能不足的惩罚）
            skill_penalty = candidates_df[skill_shortfalls].sum(axis=1) * 0.1  # 每不足1点扣10%
            candidates_df['综合评分'] = (
                0.4 * candidates_df['技能评分_缩放'] +
                0.3 * candidates_df['表现_缩放'] +
                0.2 * candidates_df['文本评分'] +
                0.1 * candidates_df['潜在评分']
            ) * (1 - skill_penalty)  # 应用惩罚

            # 生成推荐理由
            candidates_df['推荐理由'] = candidates_df.apply(
                lambda x: self.generate_reason(x, position), axis=1)

            result_cols = ['ID', '姓名', '部门'] + self.skills + ['表现', '技能评分', '文本评分', '潜在评分', '综合评分', '推荐理由', '反馈']
            result_df = candidates_df[result_cols].nlargest(top_n, '综合评分').reset_index(drop=True)
            return result_df
        except Exception as e:
            print(f"推荐过程中发生错误: {str(e)}")
            return pd.DataFrame()

    def generate_reason(self, candidate, position):
        reqs = self.position_reqs[position]
        reasons = []

        # 技能适配性（明确不足点）
        missing_skills = []
        for skill, min_level in reqs["必需技能"].items():
            actual = candidate[skill]
            required = min_level - 0.5
            if actual >= required:
                level = "◎" if actual >= 4.5 else "○"
                reasons.append(f"{skill}{level}({actual:.1f}/5)")
            else:
                missing_skills.append(f"{skill}(不足:{required - actual:.1f})")

        if missing_skills:
            reasons.append(f"※需要改进的技能: {', '.join(missing_skills)}")

        # 反馈分析
        feedback_analysis = []

        # 关键词提取
        matched_keywords = []
        for skill in reqs["必需技能"]:
            if any(kw in candidate['反馈'] for kw in self.skill_keywords.get(skill, [])):
                matched_keywords.append(skill)
        if matched_keywords:
            feedback_analysis.append(f"检测到{len(matched_keywords)}个技能关键词")

        # 具体事例
        if any(c in candidate['反馈'] for c in ['业绩', '成果', '达成']):
            feedback_analysis.append("有具体事例")

        # 成长记录
        if any(c in candidate['反馈'] for c in ['成长', '改进', '提升']):
            feedback_analysis.append("有成长记录")

        if feedback_analysis:
            reasons.append("定性分析: " + ", ".join(feedback_analysis))

        # 潜在能力
        potential_score = self.extract_potential(candidate['反馈'])
        if potential_score > 0.6:
            reasons.append(f"潜在能力: {potential_score:.0%}")
            reasons.append(f"成长意愿: {candidate['成长意愿']:.1f}/1.0")

        return f"【{position}适配性】\n" + "\n".join(f"・{r}" for r in reasons)

    def predict_growth(self, candidate_id, position):
        candidate = self.df[self.df['ID'] == candidate_id].iloc[0]
        reqs = self.position_reqs[position]

        predictions = {
            'year': ['现在', '1年后', '2年后', '3年后'],
            '表现': [candidate['表现']],
            '技能': {},
            '培养计划': []
        }

        # 技能成长预测
        for skill in self.skills:
            current = candidate[skill]
            growth = 0

            if skill in reqs["必需技能"]:
                plan = random.choice(self.development_plans[skill])
                predictions['培养计划'].append(f"{skill}: {plan}")
                growth = min(0.7, 0.2 + candidate['成长意愿'] * 0.5)

            predictions['技能'][skill] = [
                current,
                np.clip(current + growth * 0.5, 1, 5),
                np.clip(current + growth * 0.8, 1, 5),
                np.clip(current + growth, 1, 5)
            ]

        # 表现预测
        perf_growth = sum(
            (predictions['技能'][s][-1] - predictions['技能'][s][0]) * w
            for s, w in reqs["权重"].items()
        ) / sum(reqs["权重"].values())

        predictions['表现'].extend([
            np.clip(candidate['表现'] + perf_growth * 0.3, 1, 5),
            np.clip(candidate['表现'] + perf_growth * 0.6, 1, 5),
            np.clip(candidate['表现'] + perf_growth, 1, 5)
        ])

        return predictions

    def analyze_optimal_team(self):
        """第1层级和第2层级的独特最佳组合提供3个方案"""
        team_options = []
        used_combinations = set()

        # 候选人池的生成（添加调试日志）
        print("\n=== 团队生成过程开始 ===")
        print("生成候选人池中...")
        with ThreadPoolExecutor() as executor:
            candidate_futures = {
                pos: executor.submit(self.recommend_candidates, pos, 5)
                for pos, reqs in self.position_reqs.items()
                if reqs["层级"] <= 2
            }
            candidates = {}
            for pos, future in candidate_futures.items():
                try:
                    result = future.result()
                    if not result.empty:
                        candidates[pos] = result['ID'].tolist()
                        print(f"✅ {pos}: 获取了{len(candidates[pos])}名候选人")
                    else:
                        print(f"⚠️ {pos}: 无候选人")
                        candidates[pos] = []
                except Exception as e:
                    print(f"🔥 {pos} 候选人获取错误: {str(e)}")
                    candidates[pos] = []

        # 总裁候选人检查
        if not candidates.get("总裁"):
            print("🛑 致命错误: 无总裁候选人")
            return []

        # print(f"\n总裁候选人数: {len(candidates['总裁']}")

        print("开始生成团队...")

        # 团队生成逻辑
        max_attempts =100
        attempt_count = 0
        generated_teams = 0

        while generated_teams < 3 and attempt_count < max_attempts:
            attempt_count += 1
            pres = random.choice(candidates["总裁"])
            other_positions = [p for p in candidates.keys() if p != "总裁"]

            team = self._generate_team(pres, candidates, other_positions)
            team_hash = self._create_team_hash(team)

            if team_hash not in used_combinations:
                self._evaluate_team(team)
                team_options.append(team)
                used_combinations.add(team_hash)
                generated_teams += 1
                print(f"🎯 生成新团队 ({generated_teams}/3) - 哈希: {team_hash}")

        # 团队评估选择前3名
        top_teams = sorted(
            team_options,
            key=lambda x: (x["技能覆盖率"], x["多样性"], -x["风险"]),
            reverse=True
        )[:3]

        print("\n=== 团队生成结果 ===")
        print(f"生成团队候选数: {len(team_options)}")
        print(f"选择最佳团队数: {len(top_teams)}")

        return self._remove_duplicate_teams(top_teams)

    def _generate_team(self, pres, candidates, other_positions):
        """团队生成辅助函数"""
        team = {
            "总裁": pres,
            "成员": {},
            "技能覆盖率": 0,
            "多样性": 0,
            "风险": 0,
            "优势": [],
            "劣势": []
        }
        selected = {pres}

        for pos in other_positions:
            available = [c for c in candidates[pos] if c not in selected]
            if available:
                choice = random.choice(available)
                team["成员"][pos] = choice
                selected.add(choice)
                print(f"  → {pos}: 添加 {choice}")
            else:
                print(f"  → {pos}: 无合适候选人")

        return team

    def _evaluate_team(self, team):
        """团队评估逻辑（包括优势/劣势分析）"""
        member_ids = [team["总裁"]] + list(team["成员"].values())
        team_df = self.df[self.df['ID'].isin(member_ids)]

        # 技能分析
        max_skills = team_df[self.skills].max()
        mean_skills = team_df[self.skills].mean()

        # 优势（前3技能）
        top_skills = mean_skills.nlargest(3)
        team["优势"] = [
            f"{skill}（平均:{value:.1f}/最大:{max_skills[skill]:.1f})"
            for skill, value in top_skills.items()
        ]

        # 劣势（后2技能）
        weak_skills = mean_skills.nsmallest(2)
        team["劣势"] = [
            f"{skill}（平均:{value:.1f}/最大:{max_skills[skill]:.1f})"
            for skill, value in weak_skills.items()
        ]

        # 数值指标
        team["技能覆盖率"] = max_skills.mean() / 5
        team["多样性"] = len(team_df['部门'].unique()) / len(self.df['部门'].unique())
        team["风险"] = 1 - team_df['潜在力'].mean()

    def _create_team_hash(self, team):
        """生成团队的唯一哈希"""
        members = tuple(sorted([team["总裁"]] + list(team["成员"].values())))
        return hash(members)

    def _remove_duplicate_teams(self, teams):
        """去除重复团队"""
        seen = set()
        unique = []
        for team in teams:
            team_hash = self._create_team_hash(team)
            if team_hash not in seen:
                seen.add(team_hash)
                unique.append(team)
        return unique


# 3. 可视化函数
def plot_growth(predictions, position):
    # 表现预测
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=predictions['year'],
        y=predictions['表现'],
        name='表现',
        line=dict(width=4, color='#1f77b4'),
        marker=dict(size=10)
    ))
    fig_perf.update_layout(
        title=f'{position}的成长预测',
        yaxis=dict(range=[1, 5], title='表现评价'),
        xaxis=dict(title='年度'),
        template='plotly_white'
    )

    # 技能预测
    fig_skill = go.Figure()
    for skill, values in predictions['技能'].items():
        fig_skill.add_trace(go.Scatter(
            x=predictions['year'],
            y=values,
            name=skill,
            mode='lines+markers'
        ))
    fig_skill.update_layout(
        title='技能进化预测',
        yaxis=dict(range=[1, 5], title='技能水平'),
        xaxis=dict(title='年度'),
        template='plotly_white'
    )

    return fig_perf, fig_skill
def display_team_analysis(planner, team, df):
    pres = df[df['ID'] == team["总裁"]].iloc[0]
    members = {pos: df[df['ID'] == eid].iloc[0] for pos, eid in team["成员"].items()}

    st.markdown(f"""
    ## 团队方案的特点
    **总裁候选人**: {pres['姓名']} ({pres['部门']})  
    **技能覆盖率**: {team["技能覆盖率"]:.0%}  
    **部门多样性**: {team["多样性"]:.0%}  
    **风险评估**: {'低' if team["风险"] < 0.3 else '中' if team["风险"] < 0.6 else '高'}
    """)

    # 团队成员表
    member_data = []
    # for pos, member in members.items():
    #     print("#"*100)
    #     print(member[planner.skills])
    #     print("#"*100)
    #
    #     member_data.append({
    #         "职位": pos,
    #         "姓名": member['姓名'],
    #         "部门": member['部门'],
    #         "主要技能": ", ".join(member[planner.skills].nlargest(3).index.tolist()),
    #         "综合评分": f"{planner.recommend_candidates(pos, top_n=10).set_index('ID').loc[member['ID'], '综合评分']:.2f}"
    #     })
    # st.table(pd.DataFrame(member_data))
    for pos, member in members.items():
        # 确认技能是数值数据类型
        skills = member[planner.skills].astype(float)

        member_data.append({
            "职位": pos,
            "姓名": member['姓名'],
            "部门": member['部门'],
            "主要技能": ", ".join(skills.nlargest(3).index.tolist()),
            "综合评分": f"{planner.recommend_candidates(pos, top_n=10).set_index('ID').loc[member['ID'], '综合评分']:.2f}"
        })
    st.table(pd.DataFrame(member_data))

    # 优势和劣势
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 团队的优势")
        for strength in team["优势"]:
            st.markdown(f"- {strength}")
    with col2:
        st.markdown("### 改进点")
        for weakness in team["劣势"]:
            st.markdown(f"- {weakness}")

    # 推荐理由
    st.markdown("""
    ### 推荐理由
    该团队组合在以下方面表现出色:
    - 覆盖了所需的广泛技能
    - 从多个部门选拔人才，确保多样性
    - 平均潜在力高，未来成长可期
    """)

    # 风险因素
    st.markdown("""
    ### 注意点
    需要注意以下几点:
    - 可能依赖于特定技能
    - 成员之间的经验年数平衡
    - 不同的沟通风格
    """)
# 4. 主应用程序
def main():
    try:
        st.set_page_config(layout="wide", page_title="AI继任者规划师 Pro")

        # 数据准备
        df, skills, position_reqs = prepare_data()
        planner = SuccessionPlanner(df, skills, position_reqs)

        # 标题
        st.title('🏢 AI继任者规划师 Pro')
        st.markdown("""
        **基于新技能体系的下一代继任者规划系统**  
        从身心健康、人格和志向、价值观和信念等六个方面进行适应性评估
        """)

        # 主界面
        tab1, tab2 = st.tabs(["🧑 个人分析", "👥 团队优化"])

        with tab1:
            st.subheader("按职位推荐候选人")

            col1, col2 = st.columns([1, 2])
            with col1:
                position_level = st.radio(
                    "职位层级",
                    ["第1层级", "第2层级", "第3层级"],
                    horizontal=True,
                    key='pos_level'
                )

                positions = [p for p, req in position_reqs.items()
                             if req["层级"] == (1 if position_level == "第1层级" else 2 if position_level == "第2层级" else 3)]

                selected_position = st.selectbox("选择职位", positions, key='pos_select')

                if st.button("分析候选人", type="primary", key='analyze_btn'):
                    with st.spinner('正在分析候选人...'):
                        st.session_state.recommendations = planner.recommend_candidates(selected_position)
                        st.session_state.selected_position = selected_position

            # 显示部分的修正
            with col2:
                if 'recommendations' in st.session_state:
                    st.subheader(f"⭐ {st.session_state.selected_position} 候选人前3名")

                    # 调试信息显示
                    # st.write("调试信息（原始数据）:")
                    st.write(st.session_state.recommendations)

                    if not st.session_state.recommendations.empty:
                        # fig = px.histogram(st.session_state.recommendations, x='综合评分',
                        # # fig = px.histogram(st.session_state.recommendations, x='count',
                        #                    title='综合评分分布', nbins=20)
                        # st.plotly_chart(fig, use_container_width=True)

                        fig = px.bar(
                            st.session_state.recommendations.sort_values('综合评分', ascending=False),
                            x='姓名',
                            y='综合评分',
                            color='综合评分',
                            color_continuous_scale='Bluered',
                            title='按候选人综合评分',
                            labels={'综合评分': '综合适应性评分', '姓名': '候选人'},
                            hover_data=['部门', '表现', '潜在评分', '技能评分']
                        )
                        # 图表布局调整
                        fig.update_layout(
                            xaxis_tickangle=-45,
                            xaxis_title=None,
                            yaxis_range=[0, 1],
                            height=500,
                            hovermode='x unified',
                            coloraxis_showscale=False,
                            margin=dict(b=150))

                        fig.update_traces(
                            texttemplate='%{y:.2f}',
                            textposition='outside',
                            hovertemplate="<b>%{x}</b><br>评分: %{y:.2f}<br>部门: %{customdata[0]}<br>"
                                          "表现: %{customdata[1]:.1f}<br>潜在评分: %{customdata[2]:.2f}<br>"
                                          "技能评分: %{customdata[3]:.2f}"
                        )
                        st.plotly_chart(fig, use_container_width=True)


                        for idx, row in st.session_state.recommendations.iterrows():
                            with st.expander(
                                    f"{idx + 1}位: {row['姓名']} ({row['部门']}) 评分: {row['综合评分']:.2f}",
                                    expanded=(idx == 0)):
                                col_a, col_b = st.columns([1, 2])
                                with col_a:
                                    # 技能雷达图
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatterpolar(
                                        r=row[skills].values,
                                        theta=skills,
                                        fill='toself',
                                        name='当前技能',
                                        line_color='#636efa'
                                    ))

                                    # 职位要求
                                    reqs = position_reqs[st.session_state.selected_position]
                                    fig.add_trace(go.Scatterpolar(
                                        r=[reqs["必需技能"].get(s, 0) for s in skills],
                                        theta=skills,
                                        name='职位要求',
                                        line=dict(color='#FFA15A', dash='dot')
                                    ))

                                    fig.update_layout(
                                        polar=dict(radialaxis=dict(range=[0, 5], visible=True)),
                                        title='技能比较',
                                        width=400,
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                with col_b:
                                    st.markdown(f"""
                                    ### 推荐理由的详细信息
                                    {row['推荐理由']}
                                    #### 定性数据分析:
                                    """)
                                    # 反馈显示（带转义处理）
                                    if '反馈' in row and pd.notnull(row['反馈']):
                                        feedback_text = row['反馈'].replace('```', 'ﾌﾟﾛｯﾄ')  # 转义Markdown符号
                                        st.markdown(f"```\n{feedback_text}\n```")
                                    else:
                                        st.warning("没有反馈信息")

                                    # {row['反馈']}
                                    # 详细分析
                        selected_candidate = st.selectbox(
                            "选择要详细分析的候选人",
                            st.session_state.recommendations['ID'].tolist(),
                            key='detail_select'
                        )

                        if selected_candidate:
                            st.subheader("📈 3年成长预测和培养计划")
                            predictions = planner.predict_growth(selected_candidate, st.session_state.selected_position)

                            # 表现预测
                            st.plotly_chart(plot_growth(predictions, st.session_state.selected_position)[0],
                                            use_container_width=True)

                            # 培养计划
                            st.markdown("### 推荐培养计划")
                            for plan in predictions['培养计划']:
                                st.markdown(f"- {plan}")

                            # 技能进化预测
                            st.plotly_chart(plot_growth(predictions, st.session_state.selected_position)[1],
                                            use_container_width=True)
                    else:
                        st.warning("未找到符合条件的候选人。请放宽条件。")

        with tab2:
            st.subheader("管理层最佳组合分析")
            st.info("AI将为您推荐最佳的总裁候选人和第2层级职位的组合")

            if st.button("生成最佳团队", type="primary", key='team_btn'):
                with st.spinner('AI正在分析最佳团队组合...'):
                    st.session_state.top_teams = planner.analyze_optimal_team()

            if 'top_teams' in st.session_state:
                for i, team in enumerate(st.session_state.top_teams, 1):
                    with st.expander(f"🏆 最佳团队方案 {i}", expanded=i == 1):
                        display_team_analysis(planner, team, df)
    except Exception as e:
        st.error(f"发生严重错误: {str(e)}")
        st.write("错误详情:")
        st.exception(e)


if __name__ == "__main__":
    main()

