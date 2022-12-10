# %%
import pandas as pd
import requests
import re
import numpy as np
import time
import tensorflow as tf

from bs4 import BeautifulSoup
from math import ceil

import translators.server as ts
from langdetect import detect

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import model_from_json

import warnings
from IPython.display import clear_output

import nltk
import spacy.cli
from nltk.corpus import stopwords
import en_core_web_sm

nltk.download('stopwords')
stopwords_en = stopwords.words("english")
spacy.cli.download("en_core_web_sm")
spc_en = en_core_web_sm.load()


warnings.filterwarnings("ignore")


# %%
import gradio as gr

# %%
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36', 'Accept-Language':'pt-BR,pt;q=0.9,en;q=0.8'}
s = requests.Session()

# %%
with gr.Blocks() as demo:
        
    input = gr.Textbox(label="Nome do produto")

    # reviews = []
    # produtos = []
    product = ""
    btn_pesq = gr.Button("Pesquisar")
    
    output = gr.outputs.Textbox(label="console")
    state = gr.State(value=product)

    def produto(input, stats):
        reviews = []
        produtos = []
        product = ""
        stats = ''
        text = str(input) + ' amazon'
        url = "https://www.google.com/search?q=" + text

        request_result=s.get(url).content

        soup = BeautifulSoup(request_result,"html.parser")


        for a in soup.find_all('a', href=True):    
            if 'amazon.com' in a['href'] and '/dp/' in a['href']:
                href888 = a['href']
                produtos.append(a.text)
                new_url = href888.replace('/url?q=', '') # remove /url?q=
                url_coments = new_url.split('&')[0].replace('/dp/', '/product-reviews/').replace('.br', '')
                if '//'  in url_coments:
                    url_coments = url_coments.split('//')[1]
                reviews.append('https://' + url_coments)

        coments = []
        cont=0

        for r in reviews:
            
            try:
                if len(coments) >= 1000:
                    break
                html = s.get(r + f'/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber=1', headers=headers).content
                soup = BeautifulSoup(html, 'html.parser')

                a_soupi = soup.find_all('div', class_='a-row a-spacing-base a-size-base')
                
                if len(a_soupi) == 0: # se não encontrar a div, pula para o próximo produto
                    continue
                
                stats += f"Analisando produto {produtos[cont]}\n"
                stats += a_soupi[0].text.replace('\n', '').replace('  ', '') + '\n' + '\n'
                yield stats, stats
                
                cont+=1 
                
                i = int(a_soupi[0].text.replace('\n', '').replace('  ', '').replace(',', '').split(' ')[3])/10 # pegando o numero de páginas
                i = ceil(i)


                if i == 1: # se só tiver uma página roda o for só uma vez
                    i+=1
                

                for j in range(1, i):
                    if len(coments) >= 1000:
                        break    
                    html = s.get(r + f'/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber={j}', headers=headers).content
                    soup = BeautifulSoup(html, 'html.parser')
                    a_soup = soup.find_all('div', class_='a-row a-spacing-small review-data')
                    
                    for a in a_soup:
                        for coment in a.text.split('\n'):
                            # limite de comentários, para o programa não ficar muito lento
                            if len(coments) >= 1000:
                                break
                            if coment != '':
                                coments.append(coment)
                
            
            except Exception as e:
                stats += e
                yield stats, stats
                continue

             
        
        time.sleep(1)

        coments_translate = []
        cont1 = 1

        # pelo fato de ter comentários de diferentes linguas, só iremos pegar os comentários em inglês

        for c in coments:
            stats=''
            stats+=f'Detectando comentários em inglês\n'
            yield stats, stats
            try:
                det = detect(c)
                if det != 'en' or 'The media could not be loaded.' in c:
                    continue
                else:
                    coments_translate.append(c)
                    cont1+=1
            except:
                continue
            
            stats+=f'{cont1} de {len(coments)} comentários\n'
            yield stats, stats
        
        stats=''
        stats+=f'{len(coments_translate)} comentários válidos\n'
        yield stats, stats
        time.sleep(1)



            

        
        
        
        ## Analisando modelo
        

        


        def limpa_texto(texto):
            '''(str) -> str
            Essa funcao recebe uma string, deixa tudo em minusculo, filtra apenas letras,
            retira stopwords, lemmatiza e retorna a string resultante.
            '''
            texto = texto.lower()

            texto = re.sub(r"[\W\d_]+", " ", texto)

            texto = [pal for pal in texto.split() if pal not in stopwords_en]

            spc_texto = spc_en(" ".join(texto))
            tokens = [word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in spc_texto]
            
            return " ".join(tokens)



        try:
            df_alexa = pd.read_csv('df_preprocessed.csv')
            df_alexa.dropna(inplace=True)
    
        except:
            stats=''
            stats+='Dataset preprocessado nao encontrado, criando novo dataset...\n'
            yield stats, stats

            try:
                df = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')

                df_alexa = df.drop(['id','dateAdded', 'dateUpdated', 'asins', 'keys', 'manufacturer', 'manufacturerNumber', 
                'reviews.date', 'reviews.dateSeen', 'reviews.didPurchase', 'reviews.doRecommend', 'reviews.id',
                'reviews.numHelpful', 'reviews.sourceURLs', 'reviews.username', 'imageURLs', 'primaryCategories', 'categories',
                'brand', 'name', 'sourceURLs'], axis=1)

                df_alexa.fillna('', inplace = True) # para nao ter problemas com nulos na concatenacao

                # concatenando as duas colunas
                df_alexa['verified_reviews'] = df_alexa['reviews.text'] + ' ' + df_alexa['reviews.title']
                # removendo entradas sem texto
                df_alexa = df_alexa[df_alexa['verified_reviews'] != ' ']


                # transformando rating em feedback 0 e 1
                labels = []
                for score in df_alexa['reviews.rating']:
                    if score > 3:
                        labels.append(1)
                    else:
                        labels.append(0)

                df_alexa['feedback'] = labels
                # Aplica a funcao nas reviews do dataset
                df_alexa['verified_reviews'] = df_alexa['verified_reviews'].apply(limpa_texto)

                # Salva o dataset preprocessado
                
                df_alexa.to_csv('df_preprocessed.csv', index=False)
            except Exception as e:
                
                stats=''
                stats+=e + '\n'
                yield stats, stats
                exit()
            


        from sklearn.feature_extraction.text import TfidfVectorizer

        texto = df_alexa['verified_reviews']
        # Importando o TfidfVectorizer

        # Instanciando o TfidfVectorizer
        tfidf_vect = TfidfVectorizer()

        # Vetorizando
        X_tfidf = tfidf_vect.fit_transform(texto)

        X_train, X_test, y_train, y_test = train_test_split(X_tfidf.toarray(), df_alexa['feedback'], test_size = 0.2)


        try:
            # load json and create model
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            classifier = model_from_json(loaded_model_json)
            # load weights into new model
            classifier.load_weights("model.h5")
            stats=''
            stats+='Modelo carregado com sucesso\n'
            yield stats, stats

            # evaluate loaded model on test data
            classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])
        except:
            # create model and save
            classifier = tf.keras.models.Sequential()
            classifier.add(tf.keras.layers.Dense(units = 10, activation='relu', input_shape=(X_train.shape[1],)))
            classifier.add(tf.keras.layers.Dense(units = 10, activation='relu'))
            classifier.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

            classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])

            # fit model
            epochs_hist = classifier.fit(X_train, y_train, epochs=100, batch_size=50,  verbose=2, validation_split=0.2)
            stats=''
            stats+='Modelo criado com sucesso\n'
            yield stats, stats


            # serialize model to JSON
            model_json = classifier.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            classifier.save_weights("model.h5")
            stats+='Salvo modelo em disco\n'
            yield stats, stats


        if len(coments_translate) == 0:
            stats=''
            stats+='Nenhum comentário para ser classificado\n'
            yield stats, stats
        else:
            c = tfidf_vect.transform(coments_translate).toarray()
            predict = (classifier.predict(c) > 0.5).astype(int)

            cont_pos = 0
            cont_neg = 0
            for p in predict:
                if p == 1:
                    cont_pos += 1
                else:
                    cont_neg += 1

            stats=''
            stats+=f'Positivos: {cont_pos} | Negativos: {cont_neg}\n'
            yield stats, stats

            indexes_neg = np.where(predict == 0)[0] # obtendo indexes dos comentarios negativos

            if len(indexes_neg) == 0:
                stats+='Nenhum comentario negativo encontrado\n'
                yield stats, stats

            else:
                stats+='Comentarios negativos traduzidos para portugues:\n\n'
                yield stats, stats
                for i in indexes_neg:
                    stats+=ts.google(coments_translate[i], to_language='pt') + '\n\n'
                    yield stats, stats


        
    
    btn_pesq.click(produto, [input, state], [state, output])        

demo.queue()

demo.launch(share=False)

# %%



