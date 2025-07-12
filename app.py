from flask import Flask , render_template,request
import numpy as np
import pandas as pd
import pickle
df=pd.read_csv("Production System Dataset.csv")
app=Flask(__name__)
models={'M001':pickle.load(open('model1.pkl',"rb")),
       'M002': pickle.load(open('model2.pkl', 'rb')),
       'M003': pickle.load(open('model3.pkl', 'rb')),
       'M004': pickle.load(open('model4.pkl', 'rb')),
       }
@app.route('/',methods={"GET"})
def home():
    machine_ids=df["machine_id"].unique().tolist()
    return render_template("index.html",machine_ids=machine_ids)
@app.route('/predict',methods=['POST'])
def predict():
    machine_id=request.form.get("machine_id")
    temperature=float(request.form.get("Temperature"))
    power_consumption=float(request.form.get("Power_consumption"))
    vibration=float(request.form.get("Vibration level"))
    query = np.array([[temperature,power_consumption,vibration]])
    model=models[machine_id]
    pred=model.predict(query)[0]
    result="maintenance required" if pred==1 else "no maintenance required"
    return render_template("index.html",prediction=result)

if __name__=="__main__":
    app.run(debug=True)
