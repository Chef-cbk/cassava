import numpy as np
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

def calculate(mask_leaf, mask_disease):
    total_leaf_pixels = np.sum(mask_leaf == 1)
    if total_leaf_pixels == 0:
        return 0.0
    damage_pixels = np.sum((mask_leaf == 1) & (mask_disease == 1))
    percent_damage = (damage_pixels / total_leaf_pixels) * 100
    return percent_damage

def trainwithregressionmodel():
    days = np.linspace(1, 25, 200)
    L = 85
    k = 0.4 
    x0 = 12
    percent_damage = L / (1 + np.exp(-k * (days - x0)))
    noise = np.random.normal(0, 2, 200)
    percent_damage = np.clip(percent_damage + noise, 0, 100)
    X = percent_damage.reshape(-1, 1)
    y = days
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predicttheday(model, percent_damage):
    if percent_damage <= 0:
        return 1
    day_pred = model.predict([[percent_damage]])
    return max(int(day_pred[0]), 1)

def makeitbetter(percent_damage):
    states = 24
    T = np.zeros((states, states))
    normalized_damage = percent_damage / 100.0

    stay_prob = 0.7 * (1 - normalized_damage)
    same_dir_prob = 0.15 + 0.4 * normalized_damage
    other_dir_prob = 0.05 + 0.15 * normalized_damage

    stay_prob = np.clip(stay_prob, 0.1, 0.7)
    same_dir_prob = np.clip(same_dir_prob, 0.15, 0.55)
    other_dir_prob = np.clip(other_dir_prob, 0.05, 0.2)

    for i in range(states):
        dir_idx = i // 3
        dist_idx = i % 3
        T[i, i] = stay_prob
        if dist_idx < 2:
            T[i, i+1] = same_dir_prob
        
        for j in range(8):
            if j != dir_idx:
                T[i, j*3 + dist_idx] = other_dir_prob / 7

        if np.sum(T[i]) > 0:
            T[i] /= np.sum(T[i])

    return T

def calculate_spread(matrix, days_to_predict):
    return np.linalg.matrix_power(matrix, days_to_predict)

def describe_spread(T_powered, days_to_predict):
    directions = ['เหนือ','ตะวันออกเฉียงเหนือ','ตะวันออก','ตะวันออกเฉียงใต้','ใต้','ตะวันตกเฉียงใต้','ตะวันตก','ตะวันตกเฉียงเหนือ']
    distances = ['ใกล้','กลาง','ไกล']

    st.subheader(f"ทำนายการลามของโรคในอีก {days_to_predict} วันถัดไป")

    avg_probs = T_powered.mean(axis=0)
    top_states = np.argsort(-avg_probs)[:3]

    top_dirs = sorted(list(set([directions[idx//3] for idx in top_states])))
    top_dists = sorted(list(set([distances[idx%3] for idx in top_states])), key=lambda x: distances.index(x))

    dir_text = ", ".join(top_dirs)
    if len(top_dirs) > 1:
        dir_text = " และ ".join([", ".join(top_dirs[:-1]), top_dirs[-1]])

    dist_text = " ถึง ".join(top_dists)

    st.write(f"ตรวจพบว่าโรคมีแนวโน้มลามไป **ทิศ{dir_text}** มากที่สุด")
    st.write(f"การแพร่กระจายส่วนใหญ่อยู่ใน **ระยะ{dist_text}**")
    st.warning(f"**คำแนะนำ:** ควรระวังและจัดการใบที่อยู่ **ทิศ{dir_text}** ของต้นนี้ เพื่อป้องกันการระบาด")

def main(mask_leaf, mask_disease):
    percent_damage = calculate(mask_leaf, mask_disease)
    st.write(f"ตรวจพบแผล **{percent_damage:.2f}%**")

    model = trainwithregressionmodel()
    day_infected = predicttheday(model, percent_damage)
    st.write(f"คาดว่าติดโรคมาแล้ว **{day_infected}** วัน")

    T = makeitbetter(percent_damage)
    days_to_predict = 5
    T_powered = calculate_spread(T, days_to_predict)
    describe_spread(T_powered, days_to_predict)