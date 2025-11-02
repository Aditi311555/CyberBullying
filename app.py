from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# load stopwords
with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

# load saved vocabulary
loaded_vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# create vectorizer object with same vocabulary
vectorizer = loaded_vectorizer

comments=[]

# load Linear SVC model
model = pickle.load(open("LinearSVCTuned.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['text']
        
        transformed_input = vectorizer.fit_transform([user_input])
        prediction = model.predict(transformed_input)[0]

        if prediction==0:
            comments.append(user_input)

    return render_template('index.html', prediction=prediction,user_text=user_input, comments=comments)

if __name__ == '__main__':
    app.run(debug=True)
