from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import uuid


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():


	request_type_str = request.method
	if request_type_str == 'GET':
		return render_template('index.html', href='static/thumbnail.jpg')

	else:
		#text = request.form['text']
		random_string = uuid.uuid4().hex
		#match = floats_str_to_np_arr(text)
		Team1 = request.form['Team1']
		Team2 = request.form['Team2']
		overnum = int(request.form['Over'])
		PP_rr = float(request.form['PP_rr'])
		PP_wkt = float(request.form['PP_wkt'])
		mid_rr = float(request.form['mid_rr'])
		mid_wkt = float(request.form['mid_wkt'])
		RR = float(request.form['RR'])
		rrr = float(request.form['rrr'])
		wkt_rem = float(request.form['wkt_rem'])

		match = np.array([PP_rr, PP_wkt, mid_rr, mid_wkt, RR, rrr, wkt_rem])

		path = 'static/' + random_string + '.svg'
		make_pred(Team1,Team2,overnum,match,path)
		return render_template('index.html', href=path)


def floats_str_to_np_arr(floats_str):
	def is_float(s):
		try:
			float(s)
			return True
		except:
			return False
	floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
	return floats

def make_pred(Team1, Team2, overnum, match, outfile):
	filename = 'model%d.joblib'%overnum
	model_in = load(filename)

	if overnum == 0:
		test = match[[0,1,2,3,4]].reshape(1,-1)
	else:
		test = match[[0,1,2,3,4,5,6]].reshape(1,-1)
	win_prob = model_in.predict_proba(test)[0][1]


	teams = ['KK', 'LQ', 'PZ', 'IU', 'QG', 'MS']
	cols = ['tab:blue', 'greenyellow', 'gold', 'tab:orange', 'tab:purple', 'tab:green']
	fig, ax = plt.subplots(figsize=(10,8))
	sizes = [np.mean(win_prob), 1-np.mean(win_prob)]
	ax.pie(sizes, labels=[Team1, Team2], autopct='%1.1f%%', wedgeprops=dict(width=0.65),
	colors = [cols[teams.index(Team1)], cols[teams.index(Team2)]], textprops={'fontsize': 14, 'fontweight': 'bold'},
	shadow=True, startangle=90)

	if overnum !=0:
		ax.set_title('%s vs %s, Prediction After %d overs 2nd Innings'%(Team1,Team2,overnum), fontsize = 16, fontweight= 'bold')
	else:
		ax.set_title('%s vs %s, Prediction After 1st Innings'%(Team1,Team2), fontsize = 16, fontweight= 'bold')

	
	fig.savefig(outfile)
	fig.show()	

