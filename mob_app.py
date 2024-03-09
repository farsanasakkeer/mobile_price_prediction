from flask import Flask, render_template, request
import pickle
import numpy as np

filename='mobile_rate_pred.pkl'
classifier = pickle.load(open(filename,'rb'))

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('mob_home.html')


@app.route('/predict', methods=['GET','POST'])
def man():
   if request.method == 'POST':
       Ram=float(request.form['Ram'])
       Rom=float(request.form['Rom'])
       Extended_Memory=float(request.form['Extended_Memory'])
       Battery_capacity=float(request.form['Battery_capacity(mAh)'])
       print("ram"+Ram)
       
       arr = np.array([[Ram,Rom,Extended_Memory,Battery_capacity]])
	
       pred = classifier.predict(arr)
       
       return render_template('mob_after.html',pred)
       


if __name__ == "__main__":
    app.run(debug=True)

