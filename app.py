import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('kmeansclusterassignment.pkl', 'rb'))
dataset = pd.read_csv('Wholesale customers data.csv')
X = dataset.iloc[:, 2:].values
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)


def predict_note_authentication(chanel, region, fresh, milk, grocery, frozen, detergents, delicassen):
    predict = model.predict(sc.transform([[fresh, milk, grocery, frozen, detergents, delicassen]]))
    print("cluster number", predict)
    if predict == [0]:
        prediction="Customer is misor"

    elif predict == [1]:
        prediction="Customer is standard"
    elif predict == [2]:
        prediction="Customer is Target"
    elif predict == [3]:
        prediction="Customer is careful"

    else:
        prediction="Custmor is sensible"

    print(prediction)
    return prediction


def main():
    st.header("Customer Segmenation on wholesale data ")

    chanel = st.selectbox(
        "Chanel",
        ("1", "2")
    )
    region = st.selectbox(
        "Region",
        ("1", "2", "3")
    )
    fresh = st.number_input("enter fresh no.")
    milk = st.number_input("Enter milk Quantity")
    grocery = st.number_input("Enter grocery Quantity")
    frozen = st.number_input("Enter frozen Quantity")
    detergents = st.number_input("Enter detergents Quantity")
    delicassen = st.number_input("Enter delicassen Quantity")
    if st.button("K-Means Prediction"):
      result=predict_note_authentication(chanel, region, fresh, milk, grocery, frozen, detergents, delicassen)
      st.success('K-means has predicted {}'.format(result))
    html_temp = """
        <div class="" style="background-color:orange;" >
        <div class="clearfix">           
        <div class="col-md-12">
        <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Experiment 8: K-Means Algo</p></center> 
        </div>
        </div>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
