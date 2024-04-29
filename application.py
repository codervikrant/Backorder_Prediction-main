from flask import Flask, request, jsonify, render_template, url_for, request
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Import dataset 
'''df = pd.read_csv('data/cleaned_data.csv')
print(df.head())


# Label Encoding
le = LabelEncoder()
df['product'] = le.fit_transform(df['product'])'''

application = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict',methods=['POST'])
def predict_data():
    product = request.form['product']
    forecast3 = request.form['forecast3']
    forecast6 = request.form['forecast6']
    forecast9 = request.form['forecast9']
    sales1 = request.form['sales1']
    sales3 = request.form['sales3']
    sales6 = request.form['sales6']
    sales9 = request.form['sales9']
    perf6 = request.form['perf6']
    perf12 = request.form['perf12']

    product = str(product)
    forecast3 = float(forecast3)
    forecast6 = float(forecast6)
    forecast9 = float(forecast9)
    sales1 = float(sales1)
    sales3 = float(sales3)
    sales6 = float(sales6)
    sales9 = float(sales9)
    perf6 = float(perf6)
    perf12 = float(perf12)

    x1 = [product, forecast3, forecast6, forecast9, sales1, sales3, sales6, sales9, perf6, perf12]
    df1 = pd.DataFrame(data= [x1], columns = ['product', 'forecast3', 'forecast6', 'forecast9', 'sales1', 'sales3', 'sales6', 'sales9', 'perf6', 'perf12'])
    le = LabelEncoder()
    df1['product'] = le.fit_transform(df1['product'])
    
    #df1['product'] = le.transform(df1['product'])
    x = df1.iloc[:, :10].values
    ans = model.predict(x)
    output = ans

    print(output)
    
    return render_template('index.html', prediction_text=output)
    
if __name__ == '__main__':
	application.run(host="0.0.0.0")
