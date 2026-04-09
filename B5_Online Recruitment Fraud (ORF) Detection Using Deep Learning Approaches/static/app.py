from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('heart.csv')  # Make sure your dataset is in the same folder or provide the correct path

# Assume the target variable is 'target' and all other columns are features
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model as a .pkl file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model has been saved as 'model.pkl'")


app = Flask(__name__)
app.secret_key = '371023ed2754119d0e5d086d2ae7736b'

# Load the trained model
def load_model():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Sign up route
@app.route('/signup')
def signup():
    return render_template('signup.html')

# Login route
@app.route('/login')
def login():
    return render_template('login.html')

# Submit signup form
@app.route('/submit_signup', methods=['POST'])
def submit_signup():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    return render_template('signup_success.html', username=username)

# Submit login form
@app.route('/submit_login', methods=['POST'])
def submit_login():
    username = request.form.get('username')
    password = request.form.get('password')
    session['username'] = username
    return render_template('dashboard.html')

# Dashboard route
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Create a DataFrame with input data
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                                  columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

        # Load the model and predict
        model = load_model()
        prediction = model.predict(input_data)

        # Return the result page with the prediction
        return render_template('result.html', prediction=prediction[0])

    # If the method is GET, render the prediction form
    return render_template('predict.html')


# Algorithm Routes
@app.route('/run_rf')
def run_rf():
    # Placeholder for Random Forest implementation or logic
    return render_template('rf_algorithm.html', algorithm_name="Random Forest")

@app.route('/run_genetic_algorithm')
def run_genetic_algorithm():
    # Placeholder for Genetic Algorithm implementation or logic
    return render_template('genetic_algorithm.html', algorithm_name="Genetic Algorithm")

@app.route('/run_bat')
def run_bat():
    # Placeholder for Bat Algorithm implementation or logic
    return render_template('bat_algorithm.html', algorithm_name="Bat Algorithm")

@app.route('/run_bee')
def run_bee():
    # Placeholder for Bee Algorithm implementation or logic
    return render_template('bee_algorithm.html', algorithm_name="Bee Algorithm")

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
