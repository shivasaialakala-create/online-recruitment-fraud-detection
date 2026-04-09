from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import json

app = Flask(__name__)
app.secret_key = '371023ed2754119d0e5d086d2ae7736b'  # Replace with a generated key

# Load songs data
with open('songs.json', 'r') as f:
    songs = json.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/submit_signup', methods=['POST'])
def submit_signup():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    # Here you would typically handle saving the user data to a database
    return render_template('signup_success.html', username=username)

@app.route('/submit_login', methods=['POST'])
def submit_login():
    username = request.form.get('username')
    password = request.form.get('password')
    session['username'] = username
    return render_template('dashboard.html')

@app.route('/all_songs')
def all_songs():
    moods = {
        "happy": [],
        "sad": [],
        "romantic": [],
        "energetic": [],
        "chill": [],
        "motivational": [],
        "angry": [],
        "nostalgic": []
    }

    # Organize songs by mood
    for song in songs:
        mood = song['mood']
        if mood in moods:
            moods[mood].append(song)

    return render_template('all_songs.html', moods=moods)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_message = request.json.get('message', '').lower()
        response = ""
        recommendations = []

        # Responses for greetings and other phrases
        if "hi" in user_message or "hello" in user_message:
            response = "Hello! How can I assist you today?"
        elif "how are you" in user_message:
            response = "I'm just a bot, but I'm here to help you! How about you?"
        elif "good morning" in user_message:
            response = "Good morning! Hope you have a great day ahead!"
        elif "good afternoon" in user_message:
            response = "Good afternoon! What can I do for you today?"
        elif "good night" in user_message:
            response = "Good night! Sleep well and come back anytime!"
        elif "bye" in user_message:
            response = "Goodbye! Have a great day! Feel free to return anytime."
        elif "MellagaMellaga" or "Chi La Sow movie song" in user_message:
            response = "Sure! Here's the song 'Mellaga Mellaga': <a href='https://www.youtube.com/watch?v=2mDCVzruYzQ' target='_blank'>Watch on YouTube</a>"
        elif "play chuttamalle chuttesave" in user_message:
            response = "Sure! Here's the song 'chuttamalle chuttesave': <a href='https://www.youtube.com/watch?v=5vsOv_bcnhs' target='_blank'>Watch on YouTube</a>"
        elif "play butta bomma" in user_message:
            response = "Sure! Here's the song 'Butta Bomma': <a href='https://www.youtube.com/watch?v=2mDCVzruYzQ' target='_blank'>Watch on YouTube</a>"
        elif "Vettaiyan - Manasilaayo" in user_message:
            response = "Sure! Here's the song 'Vettaiyan - Manasilaayo': <a href='https://www.youtube.com/watch?v=AiD6SOOBKZI' target='_blank'>Watch on YouTube</a>"
        elif "play Ayudha Pooja" in user_message:
            response = "Sure! Here's the song 'Ayudha Pooja': <a href='https://www.youtube.com/watch?v=HZ_Q20ir-gg' target='_blank'>Watch on YouTube</a>"
        elif "play Hey Rangule" or "Amaran song" in user_message:
            response = "Sure! Here's the song 'Amaran': <a href='https://www.youtube.com/watch?v=qaf4cDPsW68' target='_blank'>Watch on YouTube</a>"
        elif "motivational" in user_message:
            response = "Here are some motivational songs to inspire you:"
            recommendations = [song for song in songs if song['mood'] == "motivational"]
        elif "nostalgic" in user_message:
            response = "Feeling nostalgic? Here are some songs that might bring back memories:"
            recommendations = [song for song in songs if song['mood'] == "nostalgic"]
        elif "angry" in user_message:
            response = "I understand that you're feeling angry. Here are some songs to let it out:"
            recommendations = [song for song in songs if song['mood'] == "angry"]
        elif "happy" in user_message:
            response = "That's wonderful to hear! Here are some happy songs for you:"
            recommendations = [song for song in songs if song['mood'] == "happy"]
        elif "sad" in user_message:
            response = "I’m sorry to hear that. Here are some songs that might resonate with your feelings:"
            recommendations = [song for song in songs if song['mood'] == "sad"]
        elif "romantic" in user_message or "love" in user_message:
            response = "Looking for something romantic? Here are some lovely suggestions:"
            recommendations = [song for song in songs if song['mood'] == "romantic"]
        elif "energetic" in user_message or "workout" in user_message:
            response = "Great choice! Here are some energetic tracks for your workout:"
            recommendations = [song for song in songs if song['mood'] == "energetic"]
        elif "chill" in user_message or "relax" in user_message:
            response = "Here are some chill tracks to help you relax:"
            recommendations = [song for song in songs if song['mood'] == "chill"]
        elif "thank you" in user_message:
            response = "You're welcome! Let me know if you want more recommendations."
        else:
            response = "I didn't quite catch that. Can you tell me how you're feeling or what genre you like?"

        # Append song recommendations if applicable
        if recommendations:
            response += "<br><br>Here are some recommendations:<br><br>"
            for song in recommendations:
                response += f"{song['title']} by {song['artist']}: <a href='{song['links']['YouTube']['url']}' target='_blank'>{song['links']['YouTube']['name']}</a><br>"

        return jsonify({"response": response}), 200

    return jsonify({"message": "Welcome to the Song Recommender Bot! Use POST /chat with a JSON payload containing your message."}), 200

@app.route('/chat_page')
def chat_page():
    return render_template('index.html')

@app.route('/review')
def review():
    return render_template('review.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
