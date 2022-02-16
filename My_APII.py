from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
import nltk
import os
import pandas as pd
import pickle
# import request
import streamlit as st
import string

def file_selector(folder_path = '.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def set_str(title_text, body_ext):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    errors_counts = 0
    selection = ['NN', 'NNS', 'NNP', 'NNPS']
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    punct = string.punctuation
    punct_space = " "*len(string.punctuation)
    
    digits_space = " "*len(string.digits)

    vec_title = []
    vec_body = []
        
    title = BeautifulSoup(title_text, 'html.parser').get_text()
    body = BeautifulSoup(body_ext, 'html.parser').get_text()
    
    title = title.translate(str.maketrans(punct, punct_space,''))
    body = body.translate(str.maketrans(punct, punct_space,''))

    title = title.translate(str.maketrans(string.digits, digits_space,''))
    body = body.translate(str.maketrans(string.digits, digits_space,''))
    
    title = title.encode('ascii',errors='ignore').decode('ascii')
    body = body.encode('ascii',errors='ignore').decode('ascii')
    
    while '  ' in title:
        title = title.replace('  ', ' ')

    while '  ' in body:
        body = body.replace('  ', ' ')
        
#     try:
    if detect(body) == 'en':

        # Suppression des mots qui ne sont pas des noms
        tok_title = nltk.word_tokenize(title)
        tok_body = nltk.word_tokenize(body)

        tag_title = nltk.pos_tag(tok_title)
        tag_body = nltk.pos_tag(tok_body)

        vec_title = [lemmatizer.lemmatize(token.lower()) for (token, tag) in tag_title if (tag in selection) & (len(token) > 1)& (token not in stop_words)]
        vec_body = [lemmatizer.lemmatize(token.lower()) for (token, tag) in tag_body if (tag in selection) & (len(token) > 1) & (token not in stop_words)]

    else :
        st.write(
            'Problem of language'
            )
            
#     except:
#         st.write("\nThis body and langag detect error: \n {0:s}\n".format(body))
#         errors_counts += 0
    
    data = vec_title + vec_body
        
    df = pd.DataFrame([data])
    
    return df

# def request_prediction(model_url, title_text, body_ext):
    
#     data = set_str(title_text, body_ext)
    
#     headers = {'Conten-type': 'application/json'}
    
#     data_json = {'data' : data}
#     response = requests.request[
#         methods = 'POST',
#         headers = headers,
#         url = model_url,
#         json = data_json
#     ]
        
#     if response.status_code != 200:
#         raise Exception{
#             'Request failed with status {}, {}'.format(response.status_code, response.text)
#         }
        
#     return response.json()

def prediction(title_text, body_ext, load_pipe, MLB):
    data = set_str(title_text, body_ext)
    pred = load_pipe.predict(data.values)
    
    prediction = MLB.inverse_transform(pred)
        
    return prediction

def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
    DetectorFactory.seed = 0
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')
    
    filename = file_selector()
    st.write('You selected `%s`' % filename)
    
    load_pipe = pickle.load(open('./APP/Pipe_LogReg/model.pkl', 'rb'))
    MLB = pickle.load(open('./APP/LabelBinarizer.pkl', 'rb'))

    st.title('Find Best tags for your Stackoverflow question')

    user_input_title = st.text_area("Title :")
    user_input_Body = st.text_area("Body :")

    predict_btn = st.button('Tags suggestion')
    if predict_btn:
        pred = None

#         pred = request_prediction(MLFLOW_URI, user_input_title, user_input_Body)
        pred = prediction(user_input_title, user_input_Body, load_pipe, MLB)
        st.write(
            'Suggested Tags : {0}'.format(pred)
        )


if __name__ == '__main__':
    main()
    