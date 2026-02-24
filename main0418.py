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


# 1. データ準備（新しいスキル体系）
def prepare_data():
    skills = ['心身健全性', '人格と志', '価値観・信念', '業務能力', '経営実務の知識と発揮能力', '職務経歴・実績']

    position_requirements = {
        "社長": {
            "必須スキル": {s: 5 for s in skills if s != '業務能力'},
            "重み": {'心身健全性': 0.2, '人格と志': 0.25, '価値観・信念': 0.2, '業務能力': 0.1,
                     '経営実務の知識と発揮能力': 0.15, '職務経歴・実績': 0.1},
            "階層": 1,
            "説明": "全社のビジョン策定と経営判断が主要な役割",
            "キーワード": ["経営", "ビジョン", "判断力", "決断力"]
        },
        "経営企画本部長": {
            "必須スキル": {s: 4 for s in skills},
            "重み": {'心身健全性': 0.15, '人格と志': 0.2, '価値観・信念': 0.15, '業務能力': 0.15,
                     '経営実務の知識と発揮能力': 0.25, '職務経歴・実績': 0.1},
            "階層": 2,
            "説明": "中長期経営計画の策定と実行管理",
            "キーワード": ["計画", "シナリオ", "分析", "戦略"]
        },
        "事業部長": {
            "必須スキル": {s: 4 for s in skills},
            "重み": {'心身健全性': 0.15, '人格と志': 0.15, '価値観・信念': 0.15, '業務能力': 0.25,
                     '経営実務の知識と発揮能力': 0.2, '職務経歴・実績': 0.1},
            "階層": 2,
            "説明": "事業単位のP&L責任と戦略実行",
            "キーワード": ["事業", "収益", "顧客", "市場"]
        },
        "内部監査部長": {
            "必須スキル":  {s: 3 for s in skills},
            "重み": {
                '価値観・信念': 0.3,
                '経営実務の知識と発揮能力': 0.25,
                '職務経歴・実績': 0.2,
                '心身健全性': 0.1,
                '人格と志': 0.1,
                '業務能力': 0.05
            },
            "階層": 3,
            "説明": "企業統治とコンプライアンスの徹底的な監査・管理",
            "キーワード": ["監査", "コンプライアンス", "リスク管理", "ガバナンス"]
        },
        "A事業部・東京支社長": {
            "必須スキル": {s: 3 for s in skills},
            "重み": {
                '業務能力': 0.3,
                '職務経歴・実績': 0.25,
                '心身健全性': 0.15,
                '人格と志': 0.15,
                '価値観・信念': 0.1,
                '経営実務の知識と発揮能力': 0.05
            },
            "階層": 3,
            "説明": "東京地域における事業戦略の実行と営業目標の達成管理",
            "キーワード": ["地域戦略", "営業管理", "チーム統率", "市場開拓"]
        },
        # その他のポジション定義
        "A事業部・大阪支社長": {
            "必須スキル": {s: 4 for s in skills},
            "重み": {
                '業務能力': 0.3, '職務経歴・実績': 0.25, '心身健全性': 0.15,
                '人格と志': 0.15, '価値観・信念': 0.1, '経営実務の知識と発揮能力': 0.05
            },
            "階層": 3,
            "説明": "大阪地域における事業展開と収益目標の達成管理",
            "キーワード": ["地域マネジメント", "収益管理", "拠点運営"]
        },
        "A事業部・本部・営業統括部長": {
            "必須スキル": {s: 4 for s in skills if s != '経営実務の知識と発揮能力'},
            "重み": {
                '業務能力': 0.35, '職務経歴・実績': 0.3, '心身健全性': 0.15,
                '人格と志': 0.1, '価値観・信念': 0.05, '経営実務の知識と発揮能力': 0.05
            },
            "階層": 3,
            "説明": "全事業部の営業戦略策定と営業部隊の統括管理",
            "キーワード": ["営業戦略", "販売管理", "チーム育成"]
        },
        "A事業部・本部・技術統括部長": {
            "必須スキル": {s: 4 for s in skills},
            "重み": {
                '業務能力': 0.4, '職務経歴・実績': 0.25, '心身健全性': 0.1,
                '人格と志': 0.1, '価値観・信念': 0.1, '経営実務の知識と発揮能力': 0.05
            },
            "階層": 3,
            "説明": "技術開発戦略の策定と技術部門の総合管理",
            "キーワード": ["技術開発", "イノベーション", "研究管理"]
        },
        "A事業部・本部・海外統括部長": {
            "必須スキル": {s: 4 for s in skills},
            "重み": {
                '価値観・信念': 0.3, '業務能力': 0.25, '職務経歴・実績': 0.2,
                '心身健全性': 0.1, '人格と志': 0.1, '経営実務の知識と発揮能力': 0.05
            },
            "階層": 3,
            "説明": "海外事業の展開戦略策定とグローバル事業の統括",
            "キーワード": ["グローバル戦略", "異文化対応", "海外展開"]
        },
        "B事業部・営業統括部長": {
            "必須スキル": {s: 4 for s in skills},
            "重み": {
                '業務能力': 0.35, '職務経歴・実績': 0.3, '心身健全性': 0.15,
                '人格と志': 0.1, '価値観・信念': 0.05, '経営実務の知識と発揮能力': 0.05
            },
            "階層": 3,
            "説明": "B事業部全体の営業戦略策定と営業目標の達成管理",
            "キーワード": ["営業戦略", "顧客管理", "販売促進"]
        },
        "B事業部・技術統括部長": {
            "必須スキル": {s: 4 for s in skills},
            "重み": {
                '業務能力': 0.4, '職務経歴・実績': 0.25, '心身健全性': 0.1,
                '人格と志': 0.1, '価値観・信念': 0.1, '経営実務の知識と発揮能力': 0.05
            },
            "階層": 3,
            "説明": "B事業部の技術開発戦略と技術チームの総合管理",
            "キーワード": ["技術管理", "開発戦略", "品質管理"]
        },
        "B事業部・開発統括部長": {
            "必須スキル": {s: 4 for s in skills},
            "重み": {
                '業務能力': 0.45, '職務経歴・実績': 0.25, '心身健全性': 0.1,
                '人格と志': 0.1, '価値観・信念': 0.05, '経営実務の知識と発揮能力': 0.05
            },
            "階層": 3,
            "説明": "新製品開発戦略の策定と開発プロセスの最適化",
            "キーワード": ["製品開発", "プロジェクト管理", "イノベーション"]
        },
        "海外子会社長": {
            "必須スキル": {s: 4 for s in skills},
            "重み": {
                '価値観・信念': 0.3, '業務能力': 0.25, '職務経歴・実績': 0.2,
                '心身健全性': 0.15, '人格と志': 0.1, '経営実務の知識と発揮能力': 0.05
            },
            "階層": 3,
            "説明": "海外子会社の経営管理と現地戦略の実行統括",
            "キーワード": ["グローバル経営", "現地適応", "海外拠点管理"]
        },
        "勉強力者": {
            "必須スキル": {s: 4 for s in skills},
            "重み": {
                '価値観・信念': 0.3, '業務能力': 0.25, '職務経歴・実績': 0.2,
                '心身健全性': 0.15, '人格と志': 0.1, '経営実務の知識と発揮能力': 0.05
            },
            "階層": 3,
            "説明": "新しい技術やビジョンを学び、それに応用した経営戦略",
            "キーワード": ["学習", "新しい技術", "ビジョン"]
        }
    }

    # 強化されたフィードバックライブラリ（新しいスキル体系に合わせて更新）
    feedback_library = {
        '心身健全性': [
            "年間を通じて病気休暇を1日も取得せず、高いパフォーマンスを維持",
            "ストレスフルな環境下でも常に冷静さを保ち、チームを安定させた",
            "健康管理を徹底し、社内マラソン大会で優勝するなど体力面でも優れる",
            "ワークライフバランスを実践し、生産性の高い働き方を模範として示した"
        ],
        '人格と志': [
            "私利私欲なく会社の発展に尽くす姿勢が全社員から尊敬されている",
            "社会貢献への強い志を持ち、CSR活動を積極的に推進している",
            "常に公平な立場で物事を判断し、誰からも信頼される人物",
            "10年後のあるべき姿を明確に語り、周囲を鼓舞する力がある"
        ],
        '価値観・信念': [
            "『顧客第一』の信念を貫き、難しい局面でも倫理観を堅持した",
            "会社のコアバリューを体現し、新入社員のロールモデルとなっている",
            "不確実な状況でもブレない判断基準を持ち、一貫した行動を取る",
            "ダイバーシティ&インクルージョンを実践し、多様な人材を活かす"
        ],
        '業務能力': [
            "担当業務で常に高い成果を上げ、部門の目標達成に大きく貢献",
            "複雑な業務プロセスを改善し、部門全体の効率を30%向上させた",
            "専門分野の深い知識を持ち、困難な課題解決に繰り返し成功",
            "新しい業務にも迅速に適応し、短期間で生産性の高いメンバーとなった"
        ],
        '経営実務の知識と発揮能力': [
            "財務諸表を深く理解し、適切な資源配分を提案して収益を改善",
            "経営陣レベルの意思決定に参加し、建設的な意見を提供している",
            "M&A案件の評価を担当し、適切な企業価値算定を行った",
            "経営戦略を実行に移す際の具体的なアクションプラン作成が得意"
        ],
        '職務経歴・実績': [
            "過去5年間で3つの異なる部門を経験し、それぞれで実績を残した",
            "新規事業立ち上げを成功させ、3年で売上10億円を達成",
            "海外赴任経験があり、国際プロジェクトをリードした実績がある",
            "業界団体の委員を務め、社外でのネットワークが広い"
        ],
        '潜在能力': [
            "まだ一部しか発揮されていないが、適切な指導があれば飛躍的に成長する可能性を感じる",
            "新しい業務にもすぐに適応し、短期間で生産性の高いメンバーとなった",
            "困難な課題に直面した時に見せる粘り強さは並外れている",
            "自己学習能力が高く、業務外でも関連する資格取得を積極的に行っている"
        ],
        '業績': [
            "XXプロジェクトで予算を15%削減しながら、スケジュールを前倒しで完了させた",
            "新規顧客開拓において、独自のアプローチで年間売上を25%増加させた",
            "チームの生産性向上施策を導入し、プロジェクト遂行期間を平均30%短縮"
        ],
        '成長': [
            "入社3年目にして既に後輩社員5名のメンターを務めている",
            "苦手だったプレゼンテーションスキルを集中的に改善し、現在では社内コンテストで優勝するまでに成長",
            "英語力に課題があったが、自主学習でTOEICスコアを300点上げ国際プロジェクトに参画"
        ]
    }

    # 150名の従業員データ生成
    np.random.seed(42)
    data = []
    departments = ['営業', '開発', '生産', '人事', '経営企画']

    for i in range(150):
        dept = np.random.choice(departments)
        emp = {
            'ID': f'E{i + 1:03d}',
            '氏名': f'従業員{i + 1}',
            '年齢': np.random.randint(30, 56),
            '部署': dept,
            '経験年数': np.random.randint(3, 26),
            'パフォーマンス': np.clip(np.random.normal(3.8, 0.8), 1, 5),
            '潜在力': np.clip(np.random.normal(0.7, 0.15), 0.3, 1.0),
            '成長意欲': np.clip(np.random.normal(0.8, 0.1), 0.5, 1.0),
            '性格特性': random.choice(['慎重', '積極的', '協調性', '分析的', '創造的', '情熱的', '冷静', '挑戦的'])
        }

        # 新しいスキル体系に基づく評価
        for skill in skills:
            base = np.random.normal(3.8, 0.8) * (0.7 + emp['潜在力'] * 0.5)
            if dept == '経営企画' and skill in ['経営実務の知識と発揮能力', '価値観・信念']:
                base += 1.5
            elif dept == '営業' and skill in ['業務能力', '職務経歴・実績']:
                base += 1.0
            elif dept == '開発' and skill in ['業務能力', '心身健全性']:
                base += 0.7
            emp[skill] = np.clip(base, 1, 5)

        # フィードバック生成（新しいスキル体系に合わせて更新）
        feedback_parts = []

        # 1. スキルに基づくコメント（強みスキルから2件選択）
        strong_skills = [s for s in skills if emp[s] >= 4]
        if len(strong_skills) >= 2:
            selected_skills = np.random.choice(strong_skills, 2, replace=False)
            for skill in selected_skills:
                feedback_parts.append(random.choice(feedback_library[skill]))

        # 2. 潜在能力コメント（確率60%で追加）
        if random.random() < 0.6:
            feedback_parts.append(random.choice(feedback_library['潜在能力']))

        # 3. 性格特性コメント
        feedback_parts.append(f"性格特性: {emp['性格特性']}で、{random.choice(feedback_library['人格と志'])}")

        # 4. 業績または成長エピソード（どちらか1件追加）
        feedback_parts.append(random.choice([
            "【実績】" + random.choice(feedback_library['業績']),
            "【成長】" + random.choice(feedback_library['成長'])
        ]))

        # フィードバックをランダムに並び替え
        random.shuffle(feedback_parts)
        emp['フィードバック'] = "■ " + "\n■ ".join(feedback_parts)
        data.append(emp)

        # スキル生成ロジックの検証
        print("生成されたスキル値の統計:")
        for skill in skills:
            values = [emp[skill] for emp in data]
            # print(f"{skill}: 平均={np.mean(values):.2f} 最大={np.max(values):.2f} 最小={np.min(values):.2f}")

    # 把data写入到excel文件中
    df = pd.DataFrame(data)
    df.to_excel('data.xlsx', index=False)

    return pd.DataFrame(data), skills, position_requirements


# 2. 推薦エンジン（新しいスキル体系に対応）
class SuccessionPlanner:
    def __init__(self, df, skills, position_reqs):
        self.df = df
        self.skills = skills
        self.position_reqs = position_reqs

        # スケーラーの初期化
        self.scalers = {
            'skill': MinMaxScaler().fit(df[skills]),
            'performance': MinMaxScaler().fit(df[['パフォーマンス']])
        }

        # テキスト分析の設定
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=500,
            stop_words=['が', 'を', 'に', 'の', 'は', 'で', 'た'],
            token_pattern=r'(?u)\b\w+\b'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(df['フィードバック'])

        # スキルキーワードマップ
        self.skill_keywords = {
            '心身健全性': ['健康', '体力', 'ストレス', '忍耐', '持久力', 'メンタル'],
            '人格と志': ['人格', '志', '信念', '倫理', '理念', 'リーダーシップ'],
            '価値観・信念': ['価値観', '信念', '倫理', '判断', '基準', '哲学'],
            '業務能力': ['業務', '効率', '生産性', '専門', '遂行', 'スキル'],
            '経営実務の知識と発揮能力': ['経営', '財務', '戦略', '意思決定', 'M&A', 'ガバナンス'],
            '職務経歴・実績': ['実績', '経験', 'プロジェクト', '成果', 'キャリア', '実務']
        }

        # 育成計画
        self.development_plans = {
            '心身健全性': [
                "健康管理プログラム(6ヶ月)",
                "ストレスマネジメント研修",
                "メンタルヘルスケアトレーニング"
            ],
            '人格と志': [
                "リーダーシップ哲学研修",
                "経営者メンタリングプログラム",
                "社会貢献プロジェクトリーダー経験"
            ],
            '価値観・信念': [
                "企業理念深化ワークショップ",
                "エグゼクティブ倫理研修",
                "ダイバーシティトレーニング"
            ],
            '業務能力': [
                "専門分野アドバンストトレーニング",
                "業務プロセス改善プロジェクト",
                "他部門ローテーション"
            ],
            '経営実務の知識と発揮能力': [
                "経営シミュレーションプログラム",
                "財務分析スペシャリスト養成講座",
                "役員会オブザーバー経験"
            ],
            '職務経歴・実績': [
                "社内公募プロジェクトへの参加",
                "海外派遣プログラム",
                "新規事業立ち上げタスクフォース"
            ]
        }

    @functools.lru_cache(maxsize=32)
    def get_text_similarity(self, position):
        """テキスト類似度計算（キャッシュ付き）"""
        reqs = self.position_reqs[position]
        pos_keywords = " ".join(reqs.get("キーワード", []))
        pos_vector = self.vectorizer.transform([pos_keywords])
        return cosine_similarity(self.tfidf_matrix, pos_vector)

    def extract_potential(self, feedback):
        """強化された潜在能力抽出ロジック"""
        signals = [
            ('可能性', 0.3), ('潜在', 0.3), ('成長', 0.2),
            ('伸びしろ', 0.4), ('適応力', 0.2), ('期待', 0.3),
            ('飛躍', 0.4), ('素質', 0.3), ('将来性', 0.3)
        ]

        score = 0
        for keyword, weight in signals:
            if keyword in feedback:
                score += weight

        # 具体的事例の有無
        if any(word in feedback for word in ['実績', '成果', '達成', '成功']):
            score += 0.2

        # 成長エピソード
        if any(word in feedback for word in ['改善', '向上', '習得', '成長']):
            score += 0.2

        return min(1.0, score)

    def recommend_candidates(self, position, top_n=3):
        try:
            reqs = self.position_reqs[position]
            print(f"\n=== {position}の推薦処理開始 ===")

            # 全候補者をコピー
            candidates_df = self.df.copy()

            # スキル不足度を計算（必須スキルとの差）
            skill_shortfalls = []
            for skill, min_level in reqs["必須スキル"].items():
                candidates_df[f'{skill}_不足'] = np.clip(min_level - 0.5 - candidates_df[skill], 0, None)
                skill_shortfalls.append(f'{skill}_不足')

            # スキルスコア計算
            skill_weights = np.array([reqs["重み"].get(s, 0) for s in self.skills])
            candidates_df['スキルスコア'] = candidates_df[self.skills].values.dot(skill_weights)

            # スケーリング
            candidates_df['スキルスコア_scaled'] = self.scalers['skill'].transform(candidates_df[self.skills]).mean(
                axis=1)
            candidates_df['パフォーマンス_scaled'] = self.scalers['performance'].transform(
                candidates_df[['パフォーマンス']]).flatten()

            # テキストマッチングスコア
            text_sim = self.get_text_similarity(position)
            candidates_df['テキストスコア'] = text_sim[candidates_df.index].flatten()

            # 潜在能力スコア
            candidates_df['潜在スコア'] = candidates_df['フィードバック'].apply(self.extract_potential)

            # 総合スコア計算（スキル不足にペナルティを適用）
            skill_penalty = candidates_df[skill_shortfalls].sum(axis=1) * 0.1  # 不足1ポイントごとに10%減点
            candidates_df['総合スコア'] = (
                                                  0.4 * candidates_df['スキルスコア_scaled'] +
                                                  0.3 * candidates_df['パフォーマンス_scaled'] +
                                                  0.2 * candidates_df['テキストスコア'] +
                                                  0.1 * candidates_df['潜在スコア']
                                          ) * (1 - skill_penalty)  # ペナルティ適用

            # 推薦理由生成
            candidates_df['推薦理由'] = candidates_df.apply(
                lambda x: self.generate_reason(x, position), axis=1)

            # 結果を整形して返す
            # result_cols = ['ID', '氏名', '部署'] + self.skills + ['パフォーマンス', 'スキルスコア',
            #                                                       'テキストスコア', '潜在スコア', '総合スコア',
            #                                                       '推薦理由']
            # result_df = candidates_df[result_cols].nlargest(top_n, '総合スコア').reset_index(drop=True)
            #
            # print("\n推薦結果トップ3:")
            # print(result_df[['ID', '総合スコア']])
            # return result_df
            result_cols = ['ID', '氏名', '部署'] + self.skills + ['パフォーマンス', 'スキルスコア',
                                                                  'テキストスコア', '潜在スコア', '総合スコア',
                                                                  '推薦理由',
                                                                  'フィードバック']
            result_df = candidates_df[result_cols].nlargest(top_n, '総合スコア').reset_index(drop=True)
            return result_df
        except Exception as e:
            print(f"推薦処理でエラー発生: {str(e)}")
            return pd.DataFrame()

    def generate_reason(self, candidate, position):
        reqs = self.position_reqs[position]
        reasons = []

        # スキル適合性（不足ポイントを明示）
        missing_skills = []
        for skill, min_level in reqs["必須スキル"].items():
            actual = candidate[skill]
            required = min_level - 0.5
            if actual >= required:
                level = "◎" if actual >= 4.5 else "○"
                reasons.append(f"{skill}{level}({actual:.1f}/5)")
            else:
                missing_skills.append(f"{skill}(不足:{required - actual:.1f})")

        if missing_skills:
            reasons.append(f"※要改善スキル: {', '.join(missing_skills)}")

        # フィードバック分析
        feedback_analysis = []

        # キーワード抽出
        matched_keywords = []
        for skill in reqs["必須スキル"]:
            if any(kw in candidate['フィードバック']
                   for kw in self.skill_keywords.get(skill, [])):
                matched_keywords.append(skill)
        if matched_keywords:
            feedback_analysis.append(f"{len(matched_keywords)}スキルキーワード検出")

        # 具体的事例
        if any(c in candidate['フィードバック'] for c in ['実績', '成果', '達成']):
            feedback_analysis.append("具体的事例あり")

        # 成長記録
        if any(c in candidate['フィードバック'] for c in ['成長', '改善', '向上']):
            feedback_analysis.append("成長記録あり")

        if feedback_analysis:
            reasons.append("定性分析: " + ", ".join(feedback_analysis))

        # 潜在能力
        potential_score = self.extract_potential(candidate['フィードバック'])
        if potential_score > 0.6:
            reasons.append(f"潜在能力: {potential_score:.0%}")
            reasons.append(f"成長意欲: {candidate['成長意欲']:.1f}/1.0")

        return f"【{position}適性】\n" + "\n".join(f"・{r}" for r in reasons)

    def predict_growth(self, candidate_id, position):
        candidate = self.df[self.df['ID'] == candidate_id].iloc[0]
        reqs = self.position_reqs[position]

        predictions = {
            'year': ['現在', '1年後', '2年後', '3年後'],
            'パフォーマンス': [candidate['パフォーマンス']],
            'スキル': {},
            '育成計画': []
        }

        # スキル成長予測
        for skill in self.skills:
            current = candidate[skill]
            growth = 0

            if skill in reqs["必須スキル"]:
                plan = random.choice(self.development_plans[skill])
                predictions['育成計画'].append(f"{skill}: {plan}")
                growth = min(0.7, 0.2 + candidate['成長意欲'] * 0.5)

            predictions['スキル'][skill] = [
                current,
                np.clip(current + growth * 0.5, 1, 5),
                np.clip(current + growth * 0.8, 1, 5),
                np.clip(current + growth, 1, 5)
            ]

        # パフォーマンス予測
        perf_growth = sum(
            (predictions['スキル'][s][-1] - predictions['スキル'][s][0]) * w
            for s, w in reqs["重み"].items()
        ) / sum(reqs["重み"].values())

        predictions['パフォーマンス'].extend([
            np.clip(candidate['パフォーマンス'] + perf_growth * 0.3, 1, 5),
            np.clip(candidate['パフォーマンス'] + perf_growth * 0.6, 1, 5),
            np.clip(candidate['パフォーマンス'] + perf_growth, 1, 5)
        ])

        return predictions

    # def analyze_optimal_team(self):
    #     """第1階層と第2階層の最適な組み合わせを3案提示"""
    #     with ThreadPoolExecutor() as executor:
    #         candidates = {
    #             pos: executor.submit(self.recommend_candidates, pos, 2).result()['ID'].tolist()
    #             for pos, reqs in self.position_reqs.items() if reqs["階層"] <= 2
    #         }
    #
    #     president_candidates = candidates.get("社長", [])
    #     other_positions = [p for p in candidates.keys() if p != "社長"]
    #
    #     team_options = []
    #     for pres in president_candidates:
    #         for _ in range(3):
    #             team = {
    #                 "社長": pres,
    #                 "メンバー": {},
    #                 "スキルカバレッジ": 0,
    #                 "多様性": 0,
    #                 "リスク": 0,
    #                 "強み": [],
    #                 "弱み": []
    #             }
    #
    #             selected = set([pres])
    #             for pos in other_positions:
    #                 available = [c for c in candidates[pos] if c not in selected]
    #                 if available:
    #                     choice = np.random.choice(available)
    #                     team["メンバー"][pos] = choice
    #                     selected.add(choice)
    #
    #             team_df = self.df[self.df['ID'].isin([pres] + list(team["メンバー"].values()))]
    #
    #             # スキルカバレッジ
    #             max_skills = team_df[self.skills].max()
    #             team["スキルカバレッジ"] = max_skills.mean() / 5
    #
    #             # 部署多様性
    #             team["多様性"] = len(team_df['部署'].unique()) / len(self.df['部署'].unique())
    #
    #             # リスク評価
    #             team["リスク"] = 1 - team_df['潜在力'].mean()
    #
    #             # 強み/弱み分析
    #             team_skills = team_df[self.skills].mean()
    #             top_skills = team_skills.nlargest(3)
    #             weak_skills = team_skills.nsmallest(2)
    #
    #             team["強み"] = [f"{s}({v:.1f}/5)" for s, v in top_skills.items()]
    #             team["弱み"] = [f"{s}({v:.1f}/5)" for s, v in weak_skills.items()]
    #
    #             team_options.append(team)
    #
    #     # トップ3チーム選出
    #     top_teams = sorted(team_options, key=lambda x: (
    #         x["スキルカバレッジ"],
    #         x["多様性"],
    #         -x["リスク"]
    #     ), reverse=True)[:3]
    #
    #     return top_teams
    def analyze_optimal_team(self):
        """第1階層と第2階層のユニークな最適組み合わせを3案提示"""
        team_options = []
        used_combinations = set()

        # 候補者プールの生成（デバッグ用ログ追加）
        print("\n=== チーム生成プロセス開始 ===")
        print("候補者プール生成中...")
        with ThreadPoolExecutor() as executor:
            candidate_futures = {
                pos: executor.submit(self.recommend_candidates, pos, 5)
                for pos, reqs in self.position_reqs.items()
                if reqs["階層"] <= 2
            }
            candidates = {}
            for pos, future in candidate_futures.items():
                try:
                    result = future.result()
                    if not result.empty:
                        candidates[pos] = result['ID'].tolist()
                        print(f"✅ {pos}: {len(candidates[pos])}名の候補者を取得")
                    else:
                        print(f"⚠️ {pos}: 候補者なし")
                        candidates[pos] = []
                except Exception as e:
                    print(f"🔥 {pos} 候補者取得エラー: {str(e)}")
                    candidates[pos] = []

        # 社長候補チェック
        if not candidates.get("社長"):
            print("🛑 致命的エラー: 社長候補が存在しません")
            return []

        # print(f"\n社長候補数: {len(candidates['社長']}")
        
        print("チーム生成を開始します...")

        # チーム生成ロジック
        max_attempts =100
        attempt_count = 0
        generated_teams = 0

        while generated_teams < 3 and attempt_count < max_attempts:
            attempt_count += 1
            pres = random.choice(candidates["社長"])
            other_positions = [p for p in candidates.keys() if p != "社長"]

            team = self._generate_team(pres, candidates, other_positions)
            team_hash = self._create_team_hash(team)

            if team_hash not in used_combinations:
                self._evaluate_team(team)
                team_options.append(team)
                used_combinations.add(team_hash)
                generated_teams += 1
                print(f"🎯 新規チーム生成 ({generated_teams}/3) - ハッシュ: {team_hash}")

        # チーム評価でトップ3を選出
        top_teams = sorted(
            team_options,
            key=lambda x: (x["スキルカバレッジ"], x["多様性"], -x["リスク"]),
            reverse=True
        )[:3]

        print("\n=== チーム生成結果 ===")
        print(f"生成チーム候補数: {len(team_options)}")
        print(f"最適チーム選出数: {len(top_teams)}")

        return self._remove_duplicate_teams(top_teams)

    def _generate_team(self, pres, candidates, other_positions):
        """チーム生成ヘルパー関数"""
        team = {
            "社長": pres,
            "メンバー": {},
            "スキルカバレッジ": 0,
            "多様性": 0,
            "リスク": 0,
            "強み": [],
            "弱み": []
        }
        selected = {pres}

        for pos in other_positions:
            available = [c for c in candidates[pos] if c not in selected]
            if available:
                choice = random.choice(available)
                team["メンバー"][pos] = choice
                selected.add(choice)
                print(f"  → {pos}: {choice} を追加")
            else:
                print(f"  → {pos}: 適任候補なし")

        return team

    def _evaluate_team(self, team):
        """チーム評価ロジック（強み/弱み分析含む）"""
        member_ids = [team["社長"]] + list(team["メンバー"].values())
        team_df = self.df[self.df['ID'].isin(member_ids)]

        # スキル分析
        max_skills = team_df[self.skills].max()
        mean_skills = team_df[self.skills].mean()

        # 強み（上位3スキル）
        top_skills = mean_skills.nlargest(3)
        team["強み"] = [
            f"{skill}（平均:{value:.1f}/最大:{max_skills[skill]:.1f})"
            for skill, value in top_skills.items()
        ]

        # 弱み（下位2スキル）
        weak_skills = mean_skills.nsmallest(2)
        team["弱み"] = [
            f"{skill}（平均:{value:.1f}/最大:{max_skills[skill]:.1f})"
            for skill, value in weak_skills.items()
        ]

        # 数値指標
        team["スキルカバレッジ"] = max_skills.mean() / 5
        team["多様性"] = len(team_df['部署'].unique()) / len(self.df['部署'].unique())
        team["リスク"] = 1 - team_df['潜在力'].mean()

    def _create_team_hash(self, team):
        """チームのユニークハッシュ生成"""
        members = tuple(sorted([team["社長"]] + list(team["メンバー"].values())))
        return hash(members)

    def _remove_duplicate_teams(self, teams):
        """重複チーム排除"""
        seen = set()
        unique = []
        for team in teams:
            team_hash = self._create_team_hash(team)
            if team_hash not in seen:
                seen.add(team_hash)
                unique.append(team)
        return unique
    # def analyze_optimal_team(self):
    #     """第1階層と第2階層のユニークな最適組み合わせを3案提示"""
    #     used_combinations = set()
    #     team_options = []
    #     # 候補者プールの生成（並列処理）
    #     with ThreadPoolExecutor() as executor:
    #         candidate_futures = {
    #             pos: executor.submit(self.recommend_candidates, pos, 5)
    #             for pos, reqs in self.position_reqs.items()
    #             if reqs["階層"] <= 2
    #         }
    #         candidates = {
    #             pos: future.result()['ID'].tolist()
    #             for pos, future in candidate_futures.items()
    #         }
    #     # 社長候補ごとに最大5チーム生成
    #     for pres in candidates.get("社長", []):
    #         team_count = 0
    #         other_positions = [p for p in candidates.keys() if p != "社長"]
    #         while team_count < 5 and len(team_options) < 15:
    #             team = self._generate_team(pres, candidates, other_positions)
    #             team_hash = self._create_team_hash(team)
    #             if team_hash not in used_combinations:
    #                 self._evaluate_team(team)
    #                 team_options.append(team)
    #                 used_combinations.add(team_hash)
    #                 team_count += 1
    #     # チーム評価でトップ3を選出
    #     top_teams = sorted(
    #         team_options,
    #         key=lambda x: (x["スキルカバレッジ"], x["多様性"], -x["リスク"]),
    #         reverse=True
    #     )[:3]
    #     return self._remove_duplicate_teams(top_teams)
    # def _generate_team(self, pres, candidates, other_positions):
    #     """チーム生成のヘルパー関数"""
    #     team = {
    #         "社長": pres,
    #         "メンバー": {},
    #         "スキルカバレッジ": 0,
    #         "多様性": 0,
    #         "リスク": 0,
    #         "強み": [],
    #         "弱み": []
    #     }
    #     selected = {pres}
    #     for pos in other_positions:
    #         available = [c for c in candidates[pos] if c not in selected]
    #         if available:
    #             choice = self._select_unique_candidate(available, selected)
    #             team["メンバー"][pos] = choice
    #             selected.add(choice)
    #     return team
    # def _select_unique_candidate(self, candidates, used_set):
    #     """重複しない候補者選択"""
    #     for candidate in candidates:
    #         if candidate not in used_set:
    #             return candidate
    #     return None
    # # def _evaluate_team(self, team):
    # #     """チーム評価ロジック"""
    # #     team_df = self.df[self.df['ID'].isin([team["社長"]] + list(team["メンバー"].values()))]
    # #     team["スキルカバレッジ"] = team_df[self.skills].max().mean() / 5
    # #     team["多様性"] = len(team_df['部署'].unique()) / len(self.df['部署'].unique())
    # #     team["リスク"] = 1 - team_df['潜在力'].mean()
    # #     team_skills = team_df[self.skills].mean()
    # #     team["強み"] = [f"{s}({v:.1f}/5)" for s, v in team_skills.nlargest(3).items()]
    # #     team["弱み"] = [f"{s}({v:.1f}/5)" for s, v in team_skills.nsmallest(2).items()]
    # def _evaluate_team(self, team):
    #     """チーム評価ロジック"""
    #     members = [team["社長"]] + list(team.get("メンバー", {}).values())
    #     members = [m for m in members if m is not None]
    #
    #     team_df = self.df[self.df['ID'].isin(members)]
    #
    #     if team_df.empty:
    #         # メンバーが存在しない場合のフォールバック
    #         team["スキルカバレッジ"] = 0
    #         team["多様性"] = 0
    #         team["リスク"] = 1
    #         team["強み"] = []
    #         team["弱み"] = []
    #         return
    #
    #     team["スキルカバレッジ"] = team_df[self.skills].max().mean() / 5
    #
    #     total_departments = len(self.df['部署'].unique())
    #     team["多様性"] = len(team_df['部署'].unique()) / total_departments if total_departments > 0 else 0
    #
    #     team["リスク"] = 1 - team_df['潜在力'].mean()
    #
    #     team_skills = team_df[self.skills].mean()
    #     team["強み"] = [f"{s}({v:.1f}/5)" for s, v in team_skills.nlargest(3).items()]
    #     team["弱み"] = [f"{s}({v:.1f}/5)" for s, v in team_skills.nsmallest(2).items()]
    #
    # def _create_team_hash(self, team):
    #     """チームのユニークハッシュ生成"""
    #     members = tuple(sorted([team["社長"]] + list(team["メンバー"].values())))
    #     return hash(members)
    # def _remove_duplicate_teams(self, teams):
    #     """完全重複チームの排除"""
    #     seen = set()
    #     unique_teams = []
    #     for team in teams:
    #         team_hash = self._create_team_hash(team)
    #         if team_hash not in seen:
    #             seen.add(team_hash)
    #             unique_teams.append(team)
    #     return unique_teams[:3]

# 3. 可視化関数
def plot_growth(predictions, position):
    # パフォーマンス予測
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=predictions['year'],
        y=predictions['パフォーマンス'],
        name='パフォーマンス',
        line=dict(width=4, color='#1f77b4'),
        marker=dict(size=10)
    ))
    fig_perf.update_layout(
        title=f'{position}としての成長予測',
        yaxis=dict(range=[1, 5], title='パフォーマンス評価'),
        xaxis=dict(title='年度'),
        template='plotly_white'
    )

    # スキル予測
    fig_skill = go.Figure()
    for skill, values in predictions['スキル'].items():
        fig_skill.add_trace(go.Scatter(
            x=predictions['year'],
            y=values,
            name=skill,
            mode='lines+markers'
        ))
    fig_skill.update_layout(
        title='スキル進化予測',
        yaxis=dict(range=[1, 5], title='スキルレベル'),
        xaxis=dict(title='年度'),
        template='plotly_white'
    )

    return fig_perf, fig_skill


def display_team_analysis(planner, team, df):
    pres = df[df['ID'] == team["社長"]].iloc[0]
    members = {pos: df[df['ID'] == eid].iloc[0] for pos, eid in team["メンバー"].items()}

    st.markdown(f"""
    ## チーム案の特徴
    **社長候補**: {pres['氏名']} ({pres['部署']})  
    **スキルカバレッジ**: {team["スキルカバレッジ"]:.0%}  
    **部署多様性**: {team["多様性"]:.0%}  
    **リスク評価**: {'低' if team["リスク"] < 0.3 else '中' if team["リスク"] < 0.6 else '高'}
    """)

    # チームメンバー表
    member_data = []
    # for pos, member in members.items():
    #     print("#"*100)
    #     print(member[planner.skills])
    #     print("#"*100)
    #
    #     member_data.append({
    #         "ポジション": pos,
    #         "氏名": member['氏名'],
    #         "部署": member['部署'],
    #         "主要スキル": ", ".join(member[planner.skills].nlargest(3).index.tolist()),
    #         "総合スコア": f"{planner.recommend_candidates(pos, top_n=10).set_index('ID').loc[member['ID'], '総合スコア']:.2f}"
    #     })
    # st.table(pd.DataFrame(member_data))
    for pos, member in members.items():
        # スキルが数値データ型であることを確認
        skills = member[planner.skills].astype(float)

        member_data.append({
            "ポジション": pos,
            "氏名": member['氏名'],
            "部署": member['部署'],
            "主要スキル": ", ".join(skills.nlargest(3).index.tolist()),
            "総合スコア": f"{planner.recommend_candidates(pos, top_n=10).set_index('ID').loc[member['ID'], '総合スコア']:.2f}"
        })
    st.table(pd.DataFrame(member_data))

    # 強みと弱み
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### チームの強み")
        for strength in team["強み"]:
            st.markdown(f"- {strength}")
    with col2:
        st.markdown("### 改善点")
        for weakness in team["弱み"]:
            st.markdown(f"- {weakness}")

    # 推奨理由
    st.markdown("""
    ### 推奨理由
    このチーム編成は以下の点で優れています:
    - 必要なスキルを広範囲にカバー
    - 複数の部署から人材を選出し多様性を確保
    - 平均潜在力が高く今後の成長が見込める
    """)

    # リスク要因
    st.markdown("""
    ### 注意点
    以下の点に注意が必要です:
    - 特定のスキルに依存している可能性
    - メンバー間の経験年数のバランス
    - コミュニケーションスタイルの違い
    """)


# 4. メインアプリケーション
def main():
    try:
        st.set_page_config(layout="wide", page_title="AI後継者プランナー Pro")

        # データ準備
        df, skills, position_reqs = prepare_data()
        planner = SuccessionPlanner(df, skills, position_reqs)

        # タイトル
        st.title('🏢 AI後継者プランナー Pro')
        st.markdown("""
        **新しいスキル体系に基づく次世代型後継者計画システム**  
        心身健全性、人格と志、価値観・信念など6つの観点から適性を評価
        """)

        # メイン画面
        tab1, tab2 = st.tabs(["🧑 個人分析", "👥 チーム最適化"])

        with tab1:
            st.subheader("ポジション別候補者推薦")

            col1, col2 = st.columns([1, 2])
            with col1:
                position_level = st.radio(
                    "ポジション階層",
                    ["第1階層", "第2階層", "第3階層"],
                    horizontal=True,
                    key='pos_level'
                )

                positions = [p for p, req in position_reqs.items()
                             if req["階層"] == (1 if position_level == "第1階層" else 2 if position_level == "第2階層" else 3)]

                selected_position = st.selectbox("ポジション選択", positions, key='pos_select')

                if st.button("候補者を分析", type="primary", key='analyze_btn'):
                    with st.spinner('候補者を分析中...'):
                        st.session_state.recommendations = planner.recommend_candidates(selected_position)
                        st.session_state.selected_position = selected_position

            # 表示部分の修正
            with col2:
                if 'recommendations' in st.session_state:
                    st.subheader(f"⭐ {st.session_state.selected_position} 候補者トップ3")

                    # デバッグ情報を表示
                    # st.write("デバッグ情報（生データ）:")
                    st.write(st.session_state.recommendations)

                    if not st.session_state.recommendations.empty:
                        # fig = px.histogram(st.session_state.recommendations, x='総合スコア',
                        # # fig = px.histogram(st.session_state.recommendations, x='count',
                        #                    title='総合スコア分布', nbins=20)
                        # st.plotly_chart(fig, use_container_width=True)

                        fig = px.bar(
                            st.session_state.recommendations.sort_values('総合スコア', ascending=False),
                            x='氏名',
                            y='総合スコア',
                            color='総合スコア',
                            color_continuous_scale='Bluered',
                            title='候補者別総合スコア',
                            labels={'総合スコア': '総合適性スコア', '氏名': '候補者'},
                            hover_data=['部署', 'パフォーマンス', '潜在スコア', 'スキルスコア']
                        )
                        # グラフのレイアウト調整
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
                            hovertemplate="<b>%{x}</b><br>スコア: %{y:.2f}<br>部署: %{customdata[0]}<br>"
                                          "パフォーマンス: %{customdata[1]:.1f}<br>潜在スコア: %{customdata[2]:.2f}<br>"
                                          "スキルスコア: %{customdata[3]:.2f}"
                        )
                        st.plotly_chart(fig, use_container_width=True)


                        # 修正前のヒストグラムコードを削除し、以下のバーチャートに変更
                        # fig = px.bar(
                        #     st.session_state.recommendations.sort_values('総合スコア', ascending=False),
                        #     x='氏名',
                        #     y='総合スコア',
                        #     color='総合スコア',
                        #     color_continuous_scale='Bluered',
                        #     title='候補者別総合スコア',
                        #     labels={'総合スコア': '総合適性スコア', '氏名': '候補者'},
                        #     hover_data=['部署', '経験年数', 'パフォーマンス']
                        # )
                        # # グラフのレイアウト調整
                        # fig.update_layout(
                        #     xaxis_tickangle=-45,
                        #     xaxis_title=None,
                        #     yaxis_range=[0, 1],
                        #     height=500,
                        #     hovermode='x unified',
                        #     coloraxis_showscale=False,
                        #     margin=dict(b=150)  # 下部マージン拡大
                        # )
                        # # バーのテキスト表示
                        # fig.update_traces(
                        #     texttemplate='%{y:.2f}',
                        #     textposition='outside'
                        # )
                        # # デフォルトのツールチップ改善
                        # fig.update_traces(
                        #     hovertemplate="<b>%{x}</b><br>"
                        #                   "スコア: %{y:.2f}<br>"
                        #                   "部署: %{customdata[0]}<br>"
                        #                   "経験年数: %{customdata[1]}年<br>"
                        #                   "パフォーマンス: %{customdata[2]:.1f}"
                        # )
                        # st.plotly_chart(fig, use_container_width=True)

                        for idx, row in st.session_state.recommendations.iterrows():
                            with st.expander(
                                    f"{idx + 1}位: {row['氏名']} ({row['部署']}) スコア: {row['総合スコア']:.2f}",
                                    expanded=(idx == 0)):
                                col_a, col_b = st.columns([1, 2])
                                with col_a:
                                    # スキルレーダーチャート
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatterpolar(
                                        r=row[skills].values,
                                        theta=skills,
                                        fill='toself',
                                        name='現在のスキル',
                                        line_color='#636efa'
                                    ))

                                    # ポジション要件
                                    reqs = position_reqs[st.session_state.selected_position]
                                    fig.add_trace(go.Scatterpolar(
                                        r=[reqs["必須スキル"].get(s, 0) for s in skills],
                                        theta=skills,
                                        name='ポジション要件',
                                        line=dict(color='#FFA15A', dash='dot')
                                    ))

                                    fig.update_layout(
                                        polar=dict(radialaxis=dict(range=[0, 5], visible=True)),
                                        title='スキル比較',
                                        width=400,
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                # with col_b:
                                #     st.markdown(f"""
                                #     ### 推薦理由の詳細
                                #     {row['推薦理由']}
                                #
                                #     #### 定性データ分析:
                                #     ```
                                #     フィードバック
                                #     ```
                                #     """)
                                with col_b:
                                    st.markdown(f"""
                                    ### 推薦理由の詳細
                                    {row['推薦理由']}
                                    #### 定性データ分析:
                                    """)
                                    # フィードバック表示（エスケープ処理付き）
                                    if 'フィードバック' in row and pd.notnull(row['フィードバック']):
                                        feedback_text = row['フィードバック'].replace('```', 'ﾌﾟﾛｯﾄ')  # マークダウン記号をエスケープ
                                        st.markdown(f"```\n{feedback_text}\n```")
                                    else:
                                        st.warning("フィードバック情報がありません")

                                    # {row['フィードバック']}
                                    # 詳細分析
                        selected_candidate = st.selectbox(
                            "詳細分析する候補者を選択",
                            st.session_state.recommendations['ID'].tolist(),
                            key='detail_select'
                        )

                        if selected_candidate:
                            st.subheader("📈 3年間の成長予測と育成計画")
                            predictions = planner.predict_growth(selected_candidate, st.session_state.selected_position)

                            # パフォーマンス予測
                            st.plotly_chart(plot_growth(predictions, st.session_state.selected_position)[0],
                                            use_container_width=True)

                            # 育成計画
                            st.markdown("### 推奨育成計画")
                            for plan in predictions['育成計画']:
                                st.markdown(f"- {plan}")

                            # スキル進化予測
                            st.plotly_chart(plot_growth(predictions, st.session_state.selected_position)[1],
                                            use_container_width=True)
                    else:
                        st.warning("該当する候補者が見つかりませんでした。条件を緩和してください。")

        with tab2:
            st.subheader("経営陣最適組み合わせ分析")
            st.info("社長候補と第2階層ポジションの最適な組み合わせをAIが提案します")

            if st.button("最適チームを生成", type="primary", key='team_btn'):
                with st.spinner('AIが最適なチーム組み合わせを分析中...'):
                    st.session_state.top_teams = planner.analyze_optimal_team()

            if 'top_teams' in st.session_state:
                for i, team in enumerate(st.session_state.top_teams, 1):
                    with st.expander(f"🏆 最適チーム案 {i}", expanded=i == 1):
                        display_team_analysis(planner, team, df)
    except Exception as e:
        st.error(f"重大なエラーが発生しました: {str(e)}")
        st.write("エラー詳細:")
        st.exception(e)


if __name__ == "__main__":
    main()
