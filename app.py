from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from services import ocr_service

import time
from flask import jsonify 
import logging
import speech_recognition as sr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import tempfile


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

# MongoDB Setup
app.config['MONGO_URI'] = 'mongodb://localhost:27017/medcare'  # Adjust URI if using MongoDB Atlas
mongo = PyMongo(app)
# logging.basicConfig(level=logging.DEBUG)

# Home Page Route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Check user credentials in MongoDB
        user = mongo.db.users.find_one({'email': email})
        
        if user and check_password_hash(user['password'], password):  # Validate hashed password
            session['user_id'] = str(user['_id'])  # Store user ID in session
            return redirect(url_for('dashboard'))
        else:
            error= "Invalid credentials, try again"
    
    return render_template('signin.html', error=error)

# Sign-Up Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)  # Hash the password
        
        existing_user = mongo.db['users'].find_one({'email': email})  # Correct way to access the 'users' collection
        if existing_user:
            error = "Email already exists! Please choose a different one."
        # Insert new user into MongoDB
        else:
            # Insert new user into MongoDB
            mongo.db['users'].insert_one({'email': email, 'password': hashed_password})
            return redirect(url_for('signin'))  # Redirect to login page after successful signup
    return render_template('signup.html', error=error)

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('signin'))

    return render_template('dashboard.html')
 


# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
vectorstore_file = "vectorstoreexample.index"
if os.path.exists(vectorstore_file):
    logging.info("Loading existing vector store...")
    vectorstore = FAISS.load_local(vectorstore_file, embeddings, index_name="index",allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
else:
    logging.error("Vector store file does not exist.")

# Initialize the language model
llm = ChatGroq(
    groq_api_key='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',
    model_name = "llama-3.3-70b-specdec",
    temperature=0.2,
    max_tokens=3000,
    model_kwargs={"top_p": 1}
)

# Define the chat prompt template
template = """
<|context|>
You are a professional emotional medical doctor who provides accurate and empathetic advice based on the patient's symptoms and health concerns. Your primary focus is on offering tailored solutions that address the patient's condition effectively.

Responsibilities:
- Accept user-provided symptoms to suggest possible health improvements.
- Diagnose the patient accurately and offer direct, truthful answers.
- Recommend appropriate medications, diet plans, and exercises tailored to the user's condition.
- Share tips for maintaining a healthy lifestyle based on data and symptom analysis.

Formatting Guidelines:
- Do NOT use markdown (`###`, `**bold**`, `- bullet points`, etc.).  
- Provide responses in a clear and natural language without markdown formatting.  
- Present information in full sentences rather than structured lists or bullet points.  
- Maintain proper spacing to ensure readability while keeping the response conversational.  
- Avoid using headings, numbered points, or bold text. Keep it natural and flowing.  
- Ensure responses are empathetic, concise, and informative, similar to how a doctor would explain to a patient.  

</s>
<|user|>
{query}
</s>
<|assistant|>


"""
prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain
rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
from langchain_core.messages import AIMessage

# Define the response function
import re

def clean_text(response):
    """Remove markdown-style formatting from the response."""
    response = re.sub(r'#+\s*', '', response)  # Remove headings (###, ##, #)
    response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)  # Remove bold (**text**)
    response = re.sub(r'-\s+', '', response)  # Remove bullet points
    response = response.replace("\n", " ")  # Convert new lines to spaces
    return response.strip()

def get_response(user_input,conversation_context):
    logging.info(f"Getting response for user input: {user_input}")
    try:
        max_tokens = 5000  # Set a safe limit within Groq API's 6000-token max
        conversation_history = conversation_context.split()  # Split into words
        if len(conversation_history) > max_tokens:
            conversation_context = " ".join(conversation_history[-max_tokens:])
        full_prompt = "Chat History: " + conversation_context + " User Input: " + user_input

        result = rag_chain.invoke(full_prompt)
        result = clean_text(result)
        time.sleep(1)  # Simulate processing delay
        return result
    except Exception as e:
        logging.error(f"Error while fetching response: {e}")
        return "Sorry, something went wrong. Please try again."


# Add Health Report Route
@app.route('/add_report', methods=['GET', 'POST'])
def add_report():
    # Define database name and collection name
    database_name = "healthcare"  # Replace with your database name
    collection_name = "health_data"  # Replace with your collection name
    if request.method == 'POST':
        # Process uploaded report (using OCR service)
        file = request.files['report']
        # Analyze the report using the OCR service
        
        analysis = ocr_service.analyze_report(file)
        # Check if analysis is an AIMessage and serialize it
        if isinstance(analysis, AIMessage):
            analysis_data = analysis.content 
        else:
            # If analysis is not AIMessage, store it as is
            analysis_data = analysis
        # Add the report analysis to the database
    
        return render_template('add_report.html', analysis=analysis_data)  # Render the page with analysis
    return render_template('add_report.html')  # Render the page with the form to upload the report
# Initialize embeddings and vector store

# Chatbot Route (Symptom Analysis)
@app.route('/chatbot')
def chatbot():
    
    return render_template('chatbot01.html')


@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']  # Get user input from form
    user_input = user_input[:500]  # Limit to 500 characters

    user_id = session.get('user_id')  # Assuming user_id is stored in the session after login

    if not user_id:
        return jsonify({'error': 'User not logged in.'}), 403
    # Fetch the user's chat history document or create one if it doesn't exist
    chat_history_collection = mongo.db.chat_history
    chat_record = chat_history_collection.find_one({'user_id': user_id})

    if not chat_record:
        chat_record = {'user_id': user_id, 'history': []}
        chat_history_collection.insert_one(chat_record)

    # Append the user's message to the history
    chat_record['history'].append({'user': user_input})
    # Prepare the conversation context for the bot to use
    conversation_context = " ".join([entry.get('user', '') + " " + entry.get('bot', '') for entry in chat_record['history']])

    # Get response
    response = get_response(user_input, conversation_context)
    clean_response = response.replace('"', '').strip()

    # Append the chatbot's response to the history
    chat_record['history'].append({'bot': clean_response})

    # Update the chat history in the database
    chat_history_collection.update_one(
        {'user_id': user_id},
        {'$set': {'history': chat_record['history']}}
    )

    return jsonify({'response': clean_response})

@app.route('/history', methods=['GET'])
def chat_history():
    user_id = session.get('user_id')  # Assuming user_id is stored in the session after login
    if not user_id:
        return jsonify({'error': 'User not logged in.'}), 403

    # Retrieve the chat history from MongoDB
    chat_history_collection = mongo.db.chat_history
    chat_record = chat_history_collection.find_one({'user_id': user_id})

    if not chat_record:
        return jsonify({'history': []})

    return jsonify({'history': chat_record['history']})
    
@app.route('/clear_history', methods=['POST'])
def clear_history():
    user_id = session.get('user_id')  # Get the user ID from session
    if not user_id:
        return jsonify({'error': 'User not logged in.'}), 403

    # Access the chat history collection in MongoDB
    chat_history_collection = mongo.db.chat_history
    
    # Delete the chat record for the logged-in user
    result = chat_history_collection.delete_one({'user_id': user_id})

    if result.deleted_count > 0:
        return jsonify({'message': 'Chat history cleared.'})
    else:
        return jsonify({'error': 'No chat history found.'}), 404


@app.route('/voice-input', methods=['POST'])
def voice_input():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Get the audio file from the form
    audio_file = request.files['audio']

    # Save the audio file temporarily
    with tempfile.NamedTemporaryFile(delete=True) as temp_audio:
        audio_file.save(temp_audio.name)

        # Recognize the speech using Google Web Speech API (you can replace this with other services)
        with sr.AudioFile(temp_audio.name) as source:
            audio = recognizer.record(source)
            try:
                user_input = recognizer.recognize_google(audio)
                return jsonify({'user_input': user_input})
            except sr.UnknownValueError:
                return jsonify({'error': 'Could not understand the audio.'})
            except sr.RequestError as e:
                return jsonify({'error': f'Could not request results; {e}'})



# Counter for repeating alarm sound


@app.route('/logout')
def logout():
    # Clear user session
    session.clear()  # Ensure the session is cleared

    # Redirect to login page (adjust the route as per your app's login page)
    return redirect('/')

if __name__ == '__main__':

    app.run(debug=False)
