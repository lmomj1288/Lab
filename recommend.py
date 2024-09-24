import random

# 메뉴 리스트
menu_list = ['menu1', 'menu2', 'menu3', 'menu4', 'menu5', 'menu6', 'menu7', 'menu8', 'menu9', 'menu10']

# 추천 알고리즘
def recommend_menu(age, mood, purchase_history):
    # 나이에 따른 메뉴 필터링
    if age == '60under':
        age_filtered_menu = ['menu1', 'menu3', 'menu5', 'menu7', 'menu9']
    else:
        age_filtered_menu = ['menu2', 'menu4', 'menu6', 'menu8', 'menu10']

    # 기분에 따른 메뉴 필터링
    if mood == 'good':
        mood_filtered_menu = ['menu1', 'menu2', 'menu5', 'menu6', 'menu9', 'menu10']
    else:
        mood_filtered_menu = ['menu3', 'menu4', 'menu7', 'menu8']

    # 나이와 기분에 따라 필터링된 메뉴 리스트
    filtered_menu = list(set(age_filtered_menu) & set(mood_filtered_menu))

    # 구매 이력에 없는 메뉴 추천
    recommended_menu = []
    non_purchased_menu = list(set(filtered_menu) - set(purchase_history))
    recommended_menu.extend(random.sample(non_purchased_menu, min(3, len(non_purchased_menu))))
    
    # 구매 이력에 없는 메뉴로부터 추천된 메뉴가 3개 미만일 때, 
    # 구매 이력에 있는 메뉴 추천
    if len(recommended_menu) < 3:
        purchased_menu = list(set(filtered_menu) & set(purchase_history))
        recommended_menu.extend(random.sample(purchased_menu, min(3 - len(recommended_menu), len(purchased_menu))))

    return recommended_menu

# example 
age = '60under'
mood = 'good'
purchase_history = ['menu1', 'menu5', 'menu8']

recommended_menu = recommend_menu(age, mood, purchase_history)
print("Recommended Menu:", recommended_menu)