from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)



model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("sell.html")



@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[x for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)


if __name__ == '__main__':
    app.run(debug=True)
