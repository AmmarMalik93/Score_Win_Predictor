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
		return render_template('index.html', href='static/index.jpg')
	else:
		random_string = uuid.uuid4().hex
		overnum = int(request.form['Over'])
		rr = float(request.form['RR'])
		wkt_rem = float(request.form['Wkt_Rem'])
		match = np.array([rr, wkt_rem]).reshape(-1,1)
		path = 'static/' + random_string + '.svg'
		make_pred(overnum, match, path)
		return render_template('index.html', href=path)

def make_pred(overnum,match,outfile):
	modelname = 'saved_models/model%d.joblib'%overnum
	model_in = load(modelname)

	pred_score = int(model_in.predict(match)[0])

	fig, ax = plt.subplots(figsize=(10,6))
	scores = np.round(match[0]*overnum)
	#ax.plot(x,scores,'r-', linewidth=3, label='Original')
	ax.plot([overnum,20],[scores,pred_score], 'rx:', linewidth=5, markersize=10, markeredgewidth = 5, label='Predicted')
	ax.grid(True,linestyle='-.')
	#ax.legend(fontsize=16)
	ax.set_xticks(np.arange(overnum,21))
	ax.set_xlabel('Over Number', fontsize=20, fontweight='bold')
	ax.set_ylabel('Score', fontsize=20, fontweight='bold')
	ax.set_title('Overs Completed: %d, Score Prediction: %d'%(overnum,pred_score), fontsize=20, fontweight='bold')
	ax.tick_params(labelsize=16)
	fig.savefig(outfile)
	fig.show()

