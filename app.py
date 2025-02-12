import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import requests

# Load environment variables from .env file
load_dotenv()
#client = ChatGoogleGenerativeAI(api_key=openai_api_key)

app = Flask(__name__)
CORS(app)

# Global variables to replace session
chat_history = []  # To store chat history
qa_chain = None  # To store the QA chain

qdrant_url = os.getenv('QDRANT_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

@app.route('/initialize', methods=['POST'])
def initialize_conversation():
    global qa_chain
    try:
        if qa_chain is None:
            vectorstore = get_vectorstore()
            if isinstance(vectorstore, str):  # Check if an error occurred
                return jsonify({"error": vectorstore}), 500
            num_chunks = 3
            qa_chain = get_qa_chain(vectorstore, num_chunks)  # Initialize QA chain globally
            return jsonify({"status": "initialized"}), 200
        else:
            return jsonify({"status": "already initialized"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/asks', methods=['POST'])
def ask():
    data = request.get_json()  # Ensure we get JSON data from the request
    user_question = data.get('question') 
    return jsonify({"response": user_question+"Response", "history": [user_question,"How are you"]}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    global chat_history, qa_chain
    data = request.get_json()  # Ensure we get JSON data from the request
    user_question = data.get('question')
    language = data.get('language')
    if user_question:
        # Initialize QA chain if it's not initialized yet
        if qa_chain is None:
            vectorstore = get_vectorstore()
            qa_chain = get_qa_chain(vectorstore, 3)

        handle_user_input(user_question, language)

        # Return both the new response and the entire chat history
        return jsonify({"response": chat_history[-1], "history": chat_history}), 200  # Return only the latest response and full chat history
    else:
        return jsonify({"error": "No question provided"}), 400

def get_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
        knowledge_base = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name="lawyer",
            timeout=120
        )
        return knowledge_base
    except Exception as e:
        return str(e)

def get_qa_chain(vectorstore, num_chunks):
    try:
        prompt_template = PromptTemplate(
            template="""
            You are a humble Law Bot trained on legal texts to assist lawyers.
            In your response, call the user 'dear lawyer'. Greet politely and 
            provide precise answers based on relevant data. If no answer is found, 
            politely inform with 'Sorry, I couldn't find any information on that topic.' 
            Context: {context}
            Question: {question}
            """,
            input_variables=["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key),  # Update model if necessary
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks}),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
        return qa
    except Exception as e:
        return str(e)

def handle_user_input(user_question, language):
    global chat_history
    greetings = ['hi', 'hello', 'hey', 'hy', 'asslam o aliakum', 'helo']

    if user_question.lower().strip() in greetings:
        response = "Welcome to the Law Bot! You can ask questions from key legal texts."
        if language=='ur':
            response = translate_text(response)
        chat_history.append({"user": user_question, "bot": response})
    else:
        result = qa_chain({"query": user_question})  # Use the qa_chain stored globally
        response = result['result']
        if language=='ur':
            response = translate_text(response)
        source = result['source_documents'][0].metadata['source']
        page_no = result['source_documents'][0].metadata['page_no']

        # Append both user input and bot response as a dictionary to the chat history
        chat_history.append({
            "user": user_question,
            "bot": f"{response} \n\n Book Name: {source} \n Page No. {page_no}"
        })
    return response

def translate_text(text, target_lang='ur'):
    url = "https://api.mymemory.translated.net/get"
    
    # Split the text into smaller chunks if it exceeds 500 characters
    max_length = 500
    translations = []

    # Process text in chunks
    for i in range(0, len(text), max_length):
        chunk = text[i:i + max_length]
        params = {
            'q': chunk,
            'langpair': f'en|{target_lang}'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check if the response is valid
        if 'responseData' in data and 'translatedText' in data['responseData']:
            translation = data['responseData']['translatedText']
            translations.append(translation)
        else:
            print(f"Error translating chunk: {chunk}")
            translations.append(chunk)  # Append the original chunk if translation fails

    # Join all translated chunks into a single string
    return ' '.join(translations)


if __name__ == '__main__':
    app.run(debug=False)
