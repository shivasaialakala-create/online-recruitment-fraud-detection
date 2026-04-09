import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
df = pd.read_csv('balanced_recruitment_fraud.csv')  # Ensure your dataset path is correct
model = pickle.load(open("dt_model.pkl", "rb"))
# Feature columns (input variables)
features = ['Job_Title','Job_Type','Salary_Offered','Job_Location','Remote','Required_Experience','Company_Rating','Job_Description_Length','Number_of_Applicants',
            'Email_Domain','Phone_Contact']

# Target variable
target = 'Fraudulent'
users = []

# Handle categorical columns (Land Cover, Soil Type) using Label Encoding
label_encoder_land_cover = LabelEncoder()
label_encoder_soil_type = LabelEncoder()
label_encoder_job_location = LabelEncoder()
label_encoder_remote = LabelEncoder()
label_encoder_email_domain = LabelEncoder()
label_encoder_phone_contact = LabelEncoder()


df['Job_Title'] = label_encoder_land_cover.fit_transform(df['Job_Title'])

df['Job_Type'] = label_encoder_soil_type.fit_transform(df['Job_Type'])
df['Job_Location'] = label_encoder_job_location.fit_transform(df['Job_Location'])

df['Remote'] = label_encoder_remote.fit_transform(df['Remote'])
df['Email_Domain'] = label_encoder_email_domain.fit_transform(df['Email_Domain'])

df['Phone_Contact'] = label_encoder_phone_contact.fit_transform(df['Phone_Contact'])

# Handle missing values (NaN values) and infinite values before proceeding with modeling
#df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
df.dropna(inplace=True)  # Remove rows with NaN values
print(df.isnull().sum())
print(df.duplicated().sum())

# Prepare the features (X) and target (y)
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)



# Evaluate the model accuracy on the test set
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f'Random Forest Accuracy: {rf_accuracy * 100:.2f}%')

# Save the trained model to disk
with open('rf_model.pkl', 'wb') as rf_model_file:
    pickle.dump(rf_model, rf_model_file)

# Train Decision Tree (DT) model
def train_decision_tree(X_train, y_train, X_test, y_test):
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)

    dt_y_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_y_pred)

    # Save the trained model to disk
    with open('dt_model.pkl', 'wb') as dt_model_file:
        pickle.dump(dt_model, dt_model_file)

    return dt_accuracy

# Train Convolutional Neural Network (CNN) model
def train_cnn(X_train, y_train, X_test, y_test):
    # Pad sequences to the same length for CNN input
    X_train_padded = pad_sequences(X_train.to_numpy(), padding='post', maxlen=1000)
    X_test_padded = pad_sequences(X_test.to_numpy(), padding='post', maxlen=1000)

    # Build CNN model
    model = Sequential()
    model.add(Conv1D(128, 5, activation='relu', input_shape=(X_train_padded.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Reshape data for CNN
    X_train_padded = X_train_padded.reshape((X_train_padded.shape[0], X_train_padded.shape[1], 1))
    X_test_padded = X_test_padded.reshape((X_test_padded.shape[0], X_test_padded.shape[1], 1))

    # Train the model
    model.fit(X_train_padded, y_train, epochs=5, batch_size=64, verbose=1)

    # Evaluate the model
    _, cnn_accuracy = model.evaluate(X_test_padded, y_test, verbose=0)

    # Save the trained model to disk
    model.save('cnn_model.h5')

    return cnn_accuracy

# Flask app setup
app = Flask(__name__)
app.secret_key = '371023ed2754119d0e5d086d2ae7736b'

# Load the trained models (Random Forest, Decision Tree, CNN models)
def load_rf_model():
    with open('rf_model.pkl', 'rb') as rf_model_file:
        rf_model = pickle.load(rf_model_file)
    return rf_model

def load_dt_model():
    with open('dt_model.pkl', 'rb') as dt_model_file:
        dt_model = pickle.load(dt_model_file)
    return dt_model

def load_cnn_model():
    from tensorflow.keras.models import load_model
    cnn_model = load_model('cnn_model.h5')
    return cnn_model

# Function to clean input data
def clean_input_data(df):
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    

    # Handle NaN values: Drop rows with NaN values
    df.dropna(inplace=True)

    # Clip extreme values to avoid overflow in float64
    df = df.clip(upper=1e10)  # You can adjust this threshold based on your dataset

    # Ensure that all values are of type float64
    df = df.astype(np.float64)

    return df

# Flask routes
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/signup') 
def signup(): 
    return render_template('signup.html')

@app.route("/submit_signup", methods=["GET", "POST"])
def submit_signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        email = request.form["email"].strip()

        if any(u["username"] == username for u in users):
            return redirect(url_for("submit_signup"))  # Username exists

        if any(u["email"] == email for u in users):
            return redirect(url_for("submit_signup"))  # Email exists

        users.append({
            "username": username,
            "password": password,
            "email": email,
        })

        return redirect(url_for("signup_success"))  # <- This requires the route below

    return render_template("submit_signup.html")
@app.route("/signup_success")
def signup_success():
    return render_template("signup_success.html")


@app.route("/submit_login", methods=["GET", "POST"])
def submit_login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        user = next((u for u in users if u["username"] == username and u["password"] == password), None)
        if user:
            return redirect(url_for("dashboard"))  # or 'dashboard' if that's your home

        return redirect(url_for("login"))  # Invalid credentials

    return render_template("login.html")




@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model
    if request.method == 'POST':
        # Get the input job-related data from the form (Updated columns)
        job_title = request.form['job_title']       # Job Title as string
        job_type = request.form['job_type']         # Job Type as string
        salary_offered = float(request.form['salary_offered'])  # Salary Offered (numeric)
        job_location = request.form['job_location']  # Job Location as string
        remote = request.form['remote']  # Remote as string (e.g., 'Yes' or 'No')
        required_experience = float(request.form['required_experience'])  # Required Experience (numeric)
        company_rating = float(request.form['company_rating'])  # Company Rating (numeric)
        job_description_length = float(request.form['job_description_length'])  # Job Description Length (numeric)
        number_of_applicants = int(request.form['number_of_applicants'])  # Number of Applicants (numeric)
        email_domain = request.form['email_domain']  # Email Domain as string
        phone_contact = request.form['phone_contact']  # Phone Contact as string

        # Convert categorical data to numeric using pre-trained label encoders
        job_title_numeric = label_encoder_land_cover.transform([job_title])
        job_type_numeric = label_encoder_soil_type.transform([job_type])
        job_location_numeric = label_encoder_job_location.transform([job_location])
        remote_numeric = label_encoder_remote.transform([remote])
        email_domain_numeric = label_encoder_email_domain.transform([email_domain])
        phone_contact_numeric = label_encoder_phone_contact.transform([phone_contact])

        # Create a DataFrame for input data (Updated columns)
        input_data = pd.DataFrame({
            'Job_Title': job_title_numeric,
            'Job_Type': job_type_numeric,
            'Salary_Offered': [salary_offered],
            'Job_Location': job_location_numeric,
            'Remote': remote_numeric,
            'Required_Experience': [required_experience],
            'Company_Rating': [company_rating],
            'Job_Description_Length': [job_description_length],
            'Number_of_Applicants': [number_of_applicants],
            'Email_Domain': email_domain_numeric,
            'Phone_Contact': phone_contact_numeric
        })

        # Clean the input data (handle NaN, Infinity, and large values)
        input_data = clean_input_data(input_data)

        # Preprocess and vectorize the input data (handle dummy variables, etc.)
        input_data = pd.get_dummies(input_data, drop_first=True)

        # Load the trained model (you can use Random Forest, Decision Tree, or CNN here)
        model_type = request.form.get('model_type')  # Choose the model type
        
        if model_type == 'rf':
            model = load_rf_model()
        elif model_type == 'dt':
            model = load_dt_model()
        elif model_type == 'cnn':
            model = load_cnn_model()

        # Make the prediction (returns a numpy array)
        if model_type == 'cnn':
            # Reshape input for CNN model
            input_data_padded = pad_sequences(input_data.to_numpy(), padding='post', maxlen=1000)
            input_data_padded = input_data_padded.reshape((input_data_padded.shape[0], input_data_padded.shape[1], 1))
            prediction = model.predict(input_data_padded)
        else:
            # For RF and DT models, just use the input data directly
            prediction = model.predict(input_data)

        # Interpret and return the result
        if prediction[0] == 1:
            result = 'Fraudulent Job Posting'
        else:
            result = 'Non-Fraudulent Job Posting'

        return render_template('result.html', prediction=result)

    # If GET request, render the input form
    return render_template('predict.html')

@app.route('/run_rf')
def run_rf():
    # Simulating Random Forest result here
    return render_template('rf_algorithm.html', rf_accuracy=rf_accuracy)

@app.route('/run_dt')
def run_dt():
    # Train Decision Tree model and get accuracy
    dt_accuracy = train_decision_tree(X_train, y_train, X_test, y_test)
    return render_template('run_dt.html', dt_accuracy=dt_accuracy)

@app.route('/run_cnn')
def run_cnn():
    # Train CNN model and get accuracy
    cnn_accuracy = train_cnn(X_train, y_train, X_test, y_test)
    return render_template('run_cnn.html', cnn_accuracy=cnn_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
