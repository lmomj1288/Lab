from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import time 

start = time.time()
# 메뉴 리스트
menu_list = ['아메리카노', '카페라테', '카푸치노', '카라멜마키아토', '바닐라라테', '초코라테', '녹차라테', '얼그레이', 
             '얼그레이밀크티', '자몽에이드', '레몬에이드', '오렌지주스', '딸기스무디', '블루베리스무디', '요거트스무디', 
             '티라미수', '치즈케이크', '초코케이크', '마카롱', '쿠키', '베이글', '샌드위치', '토스트', '크로와상', '스콘',
             '카레', '파스타', '피자', '리조또', '샐러드']

# 전체 사용자 구매 이력 데이터
all_purchase_history = [
    ['아메리카노', '카페라테', '카푸치노', '카라멜마키아토', '바닐라라테', '카페라테', '카라멜마키아토', 'age_60under', 'mood_good'],
    ['카페라테', '카라멜마키아토', '초코라테', '녹차라테', '얼그레이', '카라멜마키아토', '녹차라테', 'age_60over', 'mood_good'],
    ['아메리카노', '카푸치노', '바닐라라테', '얼그레이', '얼그레이밀크티', '카푸치노', '얼그레이', 'age_60under', 'mood_bad'],
    ['카푸치노', '초코라테', '녹차라테', '자몽에이드', '레몬에이드', '초코라테', '자몽에이드', 'age_60over', 'mood_bad'],
    ['아메리카노', '바닐라라테', '얼그레이밀크티', '오렌지주스', '딸기스무디', '바닐라라테', '오렌지주스', 'age_60under', 'mood_good'],
    ['카페라테', '얼그레이', '얼그레이밀크티', '블루베리스무디', '요거트스무디', '얼그레이', '블루베리스무디', 'age_60over', 'mood_good'],
    ['카라멜마키아토', '녹차라테', '오렌지주스', '티라미수', '치즈케이크', '녹차라테', '티라미수', 'age_60under', 'mood_bad'],
    ['초코라테', '자몽에이드', '딸기스무디', '초코케이크', '마카롱', '자몽에이드', '초코케이크', 'age_60over', 'mood_bad'],
    ['얼그레이밀크티', '레몬에이드', '블루베리스무디', '쿠키', '베이글', '레몬에이드', '쿠키', 'age_60under', 'mood_good'],
    ['얼그레이', '얼그레이밀크티', '초코케이크', '샌드위치', '토스트', '얼그레이밀크티', '샌드위치', 'age_60over', 'mood_good'],
    ['오렌지주스', '티라미수', '마카롱', '크로와상', '스콘', '티라미수', '크로와상', 'age_60under', 'mood_bad'],
    ['딸기스무디', '치즈케이크', '쿠키', '카레', '파스타', '치즈케이크', '카레', 'age_60over', 'mood_bad']
]

# 추천 알고리즘
def recommend_menu(age, mood, purchase_history):
    # 사용자 구매 이력 인코딩
    te = TransactionEncoder()
    te_ary = te.fit([purchase_history]).transform([purchase_history])
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # 사용자 구매 이력에 대한 Apriori 알고리즘 적용
    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # 연관 규칙 기반 메뉴 추천
    recommended_menu_purchase = []
    for item in rules['consequents']:
        if len(item) == 1:
            recommended_menu_purchase.append(list(item)[0])
    recommended_menu_purchase = recommended_menu_purchase[:3]  # 최대 3개까지 추천

    # 동일한 나이 또는 감정을 가진 사용자들의 구매 이력 추출
    similar_user_history = []
    for history in all_purchase_history:
        if (f'age_{age}' in history) or (f'mood_{mood}' in history):
            similar_user_history.append([item for item in history if item in menu_list])

    # 데이터 인코딩
    te = TransactionEncoder()
    te_ary = te.fit(similar_user_history).transform(similar_user_history)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Apriori 알고리즘 적용
    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # 연관 규칙 기반 메뉴 추천
    recommended_menu_similar = []
    for item in rules['consequents']:
        if len(item) == 1 and list(item)[0] not in recommended_menu_purchase:
            recommended_menu_similar.append(list(item)[0])
    recommended_menu_similar = recommended_menu_similar[:3]  # 최대 3개까지 추천

    # 추천 메뉴 합치기
    recommended_menu = recommended_menu_purchase + recommended_menu_similar

    # 추천 메뉴가 6개 미만인 경우, 구매 이력에 없는 메뉴 중 랜덤으로 추천
    if len(recommended_menu) < 6:
        non_purchased_menu = list(set(menu_list) - set(purchase_history))
        recommended_menu.extend(non_purchased_menu[:6 - len(recommended_menu)])

    # 추천 메뉴가 6개를 초과하는 경우, 6개까지 제한
    recommended_menu = recommended_menu[:6]
    
    print("비슷한 사용자 구매 이력 기반:",recommended_menu_similar)

    return recommended_menu

age = '60over'
mood = 'good'
purchase_history = ['카푸치노', '바닐라라테', '녹차라테', '오렌지주스', '초코케이크', '크로와상', '카푸치노', '녹차라테', '오렌지주스']

recommended_menu = recommend_menu(age, mood, purchase_history)
print("Recommended Menu:", recommended_menu)
end = time.time()

print("time :",end - start)