from flask import Flask, request,jsonify,render_template
import numpy as np
import pickle
import os

#creating app name
app=Flask(__name__)

#function to load the model
def Load():
	return pickle.load(open('player_rating.pkl','rb'))

#loading defalut page
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	gk1=float(request.form.get("gk_diving",""))
	gk2=float(request.form.get("gk_reflexes",""))
	gk3=float(request.form.get("gk_positioning",""))
	gk4=float(request.form.get("gk_handling",""))
	gk5=float(request.form.get("movement_reactions",""))
	gk6=float(request.form.get("gk_kicking",""))
	gk7=float(request.form.get("mentality_composure",""))
	gk8=float(request.form.get("passing",""))
	gk9=float(request.form.get("potential",""))
	gk10=float(request.form.get("value_eur",""))
	gk11=float(request.form.get("release_clause_eur",""))
	gk12=float(request.form.get("dribbling",""))
	gk13=float(request.form.get("wage_eur",""))
	gk14=float(request.form.get("power_shot_power",""))
	features=[gk1,gk2,gk3,gk4,gk5,gk6,gk7,gk8,gk9,gk10,gk11,gk12,gk13,gk14]
	values=[np.array(features)]
	model=Load()
	y_pred=model.predict(values)
	return render_template('index.html',output='The player`s predicted overal score is:{}'.format(y_pred))

if __name__=='__main__':
	port=int(os.environ.get('PORT',5000))
	app.run(port=port,debug=True,use_reloader=False)