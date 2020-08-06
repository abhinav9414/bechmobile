from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)



model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("predictprice.html")



@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict(final)
    output='{}'.format(prediction)
    if output==1:
        return render_template('predictprice.html',pred='Your predicted price is upto 5000')
    if output==2:
	    return render_template('predictprice.html',pred='Your phone predicted  price is in range 5000-10000')
    if output==3:
	    return render_template('predictprice.html',pred='Your phone predicted price is in range 10000-15000')
    else:
	    return render_template('predictprice.html',pred='Your preditited price is over 20,000')

if __name__ == '__main__':
    app.run(debug=True)
