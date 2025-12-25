
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

# Menu Dataset
data = {
    'Dish': ['Veg Sandwich', 'Chicken Burger', 'Fruit Salad', 'Pasta', 'Grilled Fish', 'Chocolate Cake', 'Smoothie', 'Paneer Wrap'],
    'Type': ['Vegetarian', 'Non-Vegetarian', 'Vegetarian', 'Vegetarian', 'Non-Vegetarian', 'Vegetarian', 'Vegetarian', 'Vegetarian'],
    'Calories': [250, 500, 150, 400, 350, 450, 200, 300],
    'Meal': ['Breakfast', 'Lunch', 'Snack', 'Lunch', 'Lunch', 'Snack', 'Breakfast', 'Lunch']
}

df = pd.DataFrame(data)

# Encode categorical columns
le_type = LabelEncoder()
le_meal = LabelEncoder()
df['Type_num'] = le_type.fit_transform(df['Type'])
df['Meal_num'] = le_meal.fit_transform(df['Meal'])

# Features and Labels
X = df[['Type_num', 'Calories', 'Meal_num']]
y = df['Dish']

# Train k-NN Recommender
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# Streamlit UI
st.title("🍽️ AI Cafeteria Menu Recommender")
st.write("Get dish recommendations based on your preferences!")

# User input
user_type = st.selectbox("Choose Type:", df['Type'].unique())
user_calories = st.slider("Preferred Calories:", 100, 600, 300)
user_meal = st.selectbox("Meal Time:", df['Meal'].unique())

if st.button("Recommend Dish"):
    user_type_num = le_type.transform([user_type])[0]
    user_meal_num = le_meal.transform([user_meal])[0]
    recommended_dish = knn.predict([[user_type_num, user_calories, user_meal_num]])
    st.success(f"Recommended Dish: {recommended_dish[0]}")
