from flask import Flask,render_template,redirect,request
from SentimentalAnalysis import Twitter_Classifier

t = Twitter_Classifier()
app = Flask(__name__)
@app.route('/')
def check():
    return render_template("index.html")

@app.route('/submit',methods = ['POST'])
def submit_data():
    if request.method == 'POST':
        feedback = request.form['tweet']
        l, pos, neg = t.NB_predict(feedback)
        l = round(l,2)
        if l > 0:
            return render_template("feedback1.html",label = l, positive = pos)
        else:
            l = abs(l)
            return render_template("feedback2.html",label = l, negative = neg)
if __name__ == '__main__':
    app.run(debug = True)