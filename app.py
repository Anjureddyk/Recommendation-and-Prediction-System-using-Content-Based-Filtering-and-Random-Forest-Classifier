import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def recommend(book_name):
    with open('pt.pkl', 'rb') as file:
        piv = pickle.load(file)
    with open('similarity_scores.pkl', 'rb') as file:
        similarity_scores = pickle.load(file)
    
    if book_name not in piv.index:
        st.write(f"Book '{book_name}' not found.")
        return []
    index = np.where(piv.index == book_name)[0][0]
    similar_books = sorted(enumerate(similarity_scores[index]), key=lambda x: x[1], reverse=True)[1:6]
    recommendations = [piv.index[i[0]] for i in similar_books]
    return recommendations

def reverse_encode(label, value):
    if label == 'PreferredLoginDevice':
        mapping = {0: 'Mobile Phone', 1: 'Computer', 2: 'Phone'}
    elif label == 'PreferredPaymentMode':
        mapping = {0: 'Debit Card', 1: 'Credit Card', 2: 'E wallet', 3: 'UPI', 4: 'COD', 5: 'CC', 6: 'Cash on Delivery'}
    elif label == 'Gender':
        mapping = {0: 'Male', 1: 'Female'}
    elif label == 'PreferedOrderCat':
        mapping = {0: 'Laptop & Accessory', 1: 'Mobile Phone', 2: 'Fashion', 3: 'Mobile', 4: 'Grocery', 5: 'Others'}
    elif label == 'MaritalStatus':
        mapping = {0: 'Married', 1: 'Single', 2: 'Divorced'}
    else:
        mapping = {}
    return mapping.get(value, value)

with open('random_forest.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

st.title('Recommendation and Prediction System')

st.header('Book Recommendation System')

with open('popular_books.pkl', 'rb') as file:
    popular_books = pickle.load(file)

st.header('Popular Books')
st.dataframe(popular_books)

st.header('Search for a Book')
book_name = st.text_input('Enter book name:')

if st.button('Get Recommendations'):
    if book_name:
        recommendations = recommend(book_name)
        if recommendations:
            st.write(f"Recommendations for '{book_name}':")
            for book in recommendations:
                st.write(book)
    else:
        st.write("Please enter a book name.")

st.header('Explore Books')
book_search = st.text_input('Search for a book:')
if book_search:
    with open('books.pkl', 'rb') as file:
        books = pickle.load(file)
    filtered_books = books[books['Book-Title'].str.contains(book_search, case=False)]
    st.dataframe(filtered_books)

st.header('Repeat Order Prediction')

tenure = st.slider('Tenure (months)', 1, 30, 15)
preferred_login_device_options = ['Mobile Phone', 'Computer', 'Phone']
preferred_login_device = st.selectbox('Preferred Login Device', preferred_login_device_options)
city_tier = st.select_slider('City Tier', options=[1, 2, 3])
warehouse_to_home = st.slider('Warehouse to Home Distance (km)', 1, 20, 10)
preferred_payment_mode_options = ['Debit Card', 'Credit Card', 'E wallet', 'UPI', 'COD', 'CC', 'Cash on Delivery']
preferred_payment_mode = st.selectbox('Preferred Payment Mode', preferred_payment_mode_options)
gender_options = ['Male', 'Female']
gender = st.selectbox('Gender', gender_options)
hour_spend_on_app = st.slider('Hour Spend on App', 0.5, 5.0, 2.5)
number_of_device_registered = st.slider('Number of Devices Registered', 1, 10, 5)
preferred_order_cat_options = ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Mobile', 'Grocery', 'Others']
prefered_order_cat = st.selectbox('Preferred Order Category', preferred_order_cat_options)
satisfaction_score = st.slider('Satisfaction Score', 1, 5, 3)
marital_status_options = ['Married', 'Single', 'Divorced']
marital_status = st.selectbox('Marital Status', marital_status_options)
number_of_address = st.slider('Number of Addresses', 1, 5, 3)
complain = st.radio('Complain', ['No', 'Yes'])
order_amount_hike_from_last_year = st.slider('Order Amount Hike From Last Year (%)', 1.0, 3.0, 2.0)
coupon_used = st.slider('Number of Coupons Used', 0, 5, 2)
order_count = st.slider('Order Count', 1, 10, 5)
day_since_last_order = st.slider('Days Since Last Order', 1, 50, 25)
cashback_amount = st.slider('Cashback Amount', 5.0, 50.0, 25.0)

encoded_values = {
    'PreferredLoginDevice': preferred_login_device_options.index(preferred_login_device),
    'PreferredPaymentMode': preferred_payment_mode_options.index(preferred_payment_mode),
    'Gender': gender_options.index(gender),
    'PreferedOrderCat': preferred_order_cat_options.index(prefered_order_cat),
    'MaritalStatus': marital_status_options.index(marital_status),
}

def predict_churn(tenure, preferred_login_device, city_tier, warehouse_to_home,
                  preferred_payment_mode, gender, hour_spend_on_app, number_of_device_registered,
                  prefered_order_cat, satisfaction_score, marital_status, number_of_address,
                  complain, order_amount_hike_from_last_year, coupon_used, order_count,
                  day_since_last_order, cashback_amount):

    reversed_values = {key: reverse_encode(key, value) for key, value in encoded_values.items()}

    input_data = pd.DataFrame({
        'Tenure': [tenure],
        'PreferredLoginDevice': [encoded_values['PreferredLoginDevice']],
        'CityTier': [city_tier],
        'WarehouseToHome': [warehouse_to_home],
        'PreferredPaymentMode': [encoded_values['PreferredPaymentMode']],
        'Gender': [encoded_values['Gender']],
        'HourSpendOnApp': [hour_spend_on_app],
        'NumberOfDeviceRegistered': [number_of_device_registered],
        'PreferedOrderCat': [encoded_values['PreferedOrderCat']],
        'SatisfactionScore': [satisfaction_score],
        'MaritalStatus': [encoded_values['MaritalStatus']],
        'NumberOfAddress': [number_of_address],
        'Complain': [1 if complain == 'Yes' else 0],
        'OrderAmountHikeFromlastYear': [order_amount_hike_from_last_year],
        'CouponUsed': [coupon_used],
        'OrderCount': [order_count],
        'DaySinceLastOrder': [day_since_last_order],
        'CashbackAmount': [cashback_amount]
    })

    prediction = loaded_model.predict(input_data)
    probability = loaded_model.predict_proba(input_data)[:, 1]
    
    return prediction, probability

if st.button('Predict'):
    prediction, probability = predict_churn(tenure, preferred_login_device, city_tier, warehouse_to_home,
                                           preferred_payment_mode, gender, hour_spend_on_app, number_of_device_registered,
                                           prefered_order_cat, satisfaction_score, marital_status, number_of_address,
                                           complain, order_amount_hike_from_last_year, coupon_used, order_count,
                                           day_since_last_order, cashback_amount)
    
    st.write(f"Prediction: {'Not Repeated Purchase' if prediction[0] == 0 else 'Repeated Purchase'}")
    st.write(f"Probability: {probability[0]:.2f}")
