from flask import *
from werkzeug.utils import secure_filename
import os  
from test import main
from train import training
app = Flask(__name__)  

@app.route('/')  
def upload():  
    return render_template("topic.html")

@app.route('/train', methods=['GET', 'POST'])
def train():
    training()
    return render_template("topic.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text=request.form['text'] 
        outfile=open("data.txt","w")
        outfile.write(text)
        outfile.close()
    prediction=main()
    if prediction==-1:
        my_variable="you got a phishing website"
    else:
        my_variable="safe"
    return render_template("success.html",my_variable=my_variable)





# @app.route('/success', methods = ['POST'])  
# def success():  
#     if request.method == 'POST':
#         text=request.form['text'] 
#         outfile=open("data.txt","w")
#         outfile.write(text)
#         outfile.close()
#         return render_template("success.html")



        

if __name__ == '__main__':  
    app.run(debug = True)  


