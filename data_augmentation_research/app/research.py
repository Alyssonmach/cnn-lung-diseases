from fun_utils.image_transformations import filters_transformations, image_manipulation
from fun_utils.image_transformations import geometric_transformations
from streamlit_autorefresh import st_autorefresh
import streamlit as st
import numpy as np
import pandas as pd
import os, random

def change_images(actual_value):
    '''
    it interacts numerically between the 45 images in the database
    
    Args:
        actual_value(int) --> current image number
        new_value(int) --> number of the next image in the database
    '''

    if actual_value != 44: new_value = actual_value + 1
    else: new_value = 0
    
    return new_value

def explanation():
    '''brief explanation of how the research works'''

    with st.expander(label = 'Com relação as imagens geradas a seguir, qualifique cada uma das imagens modificadas tendo como comparação '
                             'a imagem original acima (Clique aqui para saber mais).'):
            st.markdown(body = 'Será analisada diversas imagens médicas de radiografias torácicas, e para cada uma delas, haverá uma '
                               'estratégia de aumento de dados associada. A sua contribuição será dada através da votação da qualidade '
                               'de análise das imagens, na qual ela pode ser qualificada como eficaz, neutra ou deteriorativa através '
                               'da inspeção visual da imagem como forma de diagnóstico. A base de dados utilizada possui 45 radiografias '
                               'de tórax associada a seis distúrbios pulmonares e imagens de pulmões saudáveis. Fique a vontade para gerar '
                               'quantas imagens achar necessário durante a análise de cada uma das estratégias de aumento de dados.')
    
    return

def app(column_progress_bar, verify_send_firebase):

    answers_labels = ['Selecione uma opção', 'Melhora a qualidade visual do diagnóstico', 
                      'Mantém neutra a qualidade visual do diagnóstico',
                      'Deteriora a qualidade visual do diagnóstico']
    help_text = 'Clique aqui para visualizar uma nova imagem de análise'

    if 'image_files' not in st.session_state:
        image_files = os.listdir('data/images/')
        random.shuffle(image_files)
        st.session_state['image_files'] = image_files
        st.session_state['state_image'] = 0 
        st.session_state['state_image_old'] = 99
        st.session_state['rate'] = 0.0
        st.session_state['rate_old'] = 0.0
        st.session_state['question_level'] = 1
        st.session_state['answers_labels'] = np.zeros((5, 3)).astype(np.uint8).tolist()
        st.session_state['reference_image'] = ['', '', '', '', '']
        st.session_state['dataframe'] = pd.read_csv(filepath_or_buffer = 'data/data.csv', delimiter = ',')
        st.session_state['imagea'], st.session_state['imageb'] = '', ''
        st.session_state['imagec'], st.session_state['imaged'] = '', ''

    with column_progress_bar:
        st.text(body = 'Barra de progresso da pesquisa:')
        my_bar = st.progress(value = 0.0)

    st.markdown(body = '***')

    if st.session_state['question_level'] == 1:

        my_bar.progress(0.20)
        
        col1, col2 = st.columns(spec = [8, 2])
        
        with col1:
            st.header(body = 'Estratégia de aumento de dados: ```Geração de Ruídos```')
        with col2:
            state_images = st.button(label = 'Gerar Novas Imagens', key = 'button1', help = help_text)
            if state_images:
                st.session_state.state_image = change_images(st.session_state.state_image)
        
        if (st.session_state.state_image_old != st.session_state.state_image):
            st.session_state.state_image_old = st.session_state.state_image
            st.session_state.rate_old = st.session_state.rate

            st.session_state.imagea = image_manipulation().organize_image(path_image = 'data/images/' + st.session_state.image_files[st.session_state.state_image], dsize = (350, 350))
            st.session_state.imageb = geometric_transformations().sp_noise(image = st.session_state.imagea, rate = random.uniform(0.05, 0.07))
            st.session_state.imagec = geometric_transformations().gaussian_noise(image = st.session_state.imagea, rate = random.uniform(0.05, 0.07))

        col1, col2,  = st.columns(spec = [4, 6])
        
        with col1:
            st.image(image = st.session_state.imagea, use_column_width = True, caption = 'Imagem Original')
        with col2:
            st.markdown(body = '***')
            st.subheader(body = 'Informações da radiografia:')     
            st.markdown(body = '**```Sexo```**: ' + st.session_state.dataframe['Sex'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Idade```**: ' + str(st.session_state.dataframe['Age'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])]) + ' anos')
            st.markdown(body = '**```Posicionamento```**: ' + st.session_state.dataframe['Frontal/Lateral'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Incidência```**: ' + st.session_state.dataframe['AP/PA'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Rotulação```**: ' + st.session_state.dataframe['Label'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '***')
        
        explanation()  
                    
        col1, col2 = st.columns(spec = [4, 6])
        
        with col1: 
            st.image(image = st.session_state.imageb, use_column_width = True, caption = 'Imagem Modificada')
        with col2:
            st.markdown(body = '***')
            st.markdown(body = '**Escolha uma opção adequada de acordo com a imagem em análise** (Ruído Sal e Pimenta):')
            value1 = st.selectbox(label = '', options = answers_labels,
                                  key = 'selectbox1', index = st.session_state.answers_labels[0][0])
            st.markdown(body = '***')
            st.session_state.answers_labels[0][0] = answers_labels.index(value1)
        
        col1, col2 = st.columns(spec = [4, 6])
        
        with col1: 
            st.image(image = st.session_state.imagec, use_column_width = True, caption = 'Imagem Modificada')
        with col2:
            st.markdown(body = '***')
            st.markdown(body = '**Escolha uma opção adequada de acordo com a imagem em análise** (Ruído Gaussiano):')
            value2 = st.selectbox(label = '', options = answers_labels,
                                  key = 'selectbox2', index = st.session_state.answers_labels[0][1])
            st.markdown(body = '***')
            st.session_state.answers_labels[0][1] = answers_labels.index(value2)

            st.session_state.answers_labels[0][2] = 'null'

        st.session_state.reference_image[0] = 'data/images/' + st.session_state.image_files[st.session_state.state_image]
   
    if st.session_state['question_level'] == 2:

        my_bar.progress(0.40)
        
        col1, col2 = st.columns(spec = [11, 2])
        
        with col1:
            st.header(body = 'Estratégia de aumento de dados: ```Transformações Geométricas```')
        with col2:
            state_images = st.button(label = 'Gerar Novas Imagens', key = 'button1', help = help_text)
            if state_images:
                st.session_state.state_image = change_images(st.session_state.state_image)
        
        if (st.session_state.state_image_old != st.session_state.state_image):
            st.session_state.state_image_old = st.session_state.state_image
            st.session_state.rate_old = st.session_state.rate

            st.session_state.imagea = image_manipulation().organize_image(path_image = 'data/images/' + st.session_state.image_files[st.session_state.state_image], dsize = (350, 350))
            st.session_state.imageb = geometric_transformations().rotate_image(image = st.session_state.imagea, rate = random.uniform(0.2, 0.4))
            st.session_state.imagec = geometric_transformations().image_translation(image = st.session_state.imagea, rate = random.uniform(0.2, 0.4))

        col1, col2,  = st.columns(spec = [4, 6])
        
        with col1:
            st.image(image = st.session_state.imagea, use_column_width = True, caption = 'Imagem Original')
        with col2:
            st.markdown(body = '***')
            st.subheader(body = 'Informações da radiografia:')          
            st.markdown(body = '**```Sexo```**: ' + st.session_state.dataframe['Sex'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Idade```**: ' + str(st.session_state.dataframe['Age'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])]) + ' anos')
            st.markdown(body = '**```Posicionamento```**: ' + st.session_state.dataframe['Frontal/Lateral'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Incidência```**: ' + st.session_state.dataframe['AP/PA'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Rotulação```**: ' + st.session_state.dataframe['Label'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '***')
        
        explanation()
                    
        col1, col2 = st.columns(spec = [4, 6])
        
        with col1: 
            st.image(image = st.session_state.imageb, use_column_width = True, caption = 'Imagem Modificada')
        with col2:
            st.markdown(body = '***')
            st.markdown(body = '**Escolha uma opção adequada de acordo com a imagem em análise** (Rotação):')
            value1 = st.selectbox(label = '', options = answers_labels,
                                  key = 'selectbox1', index = st.session_state.answers_labels[1][0])
            st.markdown(body = '***')
            st.session_state.answers_labels[1][0] = answers_labels.index(value1)
        
        col1, col2 = st.columns(spec = [4, 6])
        
        with col1: 
            st.image(image = st.session_state.imagec, use_column_width = True, caption = 'Imagem Modificada')
        with col2:
            st.markdown(body = '***')
            st.markdown(body = '**Escolha uma opção adequada de acordo com a imagem em análise** (Translação):')
            value2 = st.selectbox(label = '', options = answers_labels,
                                  key = 'selectbox2', index = st.session_state.answers_labels[1][1])
            st.markdown(body = '***')
            st.session_state.answers_labels[1][1] = answers_labels.index(value2)

            st.session_state.answers_labels[1][2] = 'null'  
        
        st.session_state.reference_image[1] = 'data/images/' + st.session_state.image_files[st.session_state.state_image]
  
    if st.session_state['question_level'] == 3:

        my_bar.progress(0.60)
        
        col1, col2 = st.columns(spec = [8, 2])
        
        with col1:
            st.header(body = 'Estratégia de aumento de dados: ```Correção Gamma```')
        with col2:
            state_images = st.button(label = 'Gerar Novas Imagens', key = 'button1', help = help_text)
            if state_images:
                st.session_state.state_image = change_images(st.session_state.state_image)
        
        if (st.session_state.state_image_old != st.session_state.state_image):
            st.session_state.state_image_old = st.session_state.state_image
            st.session_state.rate_old = st.session_state.rate

            st.session_state.imagea = image_manipulation().organize_image(path_image = 'data/images/' + st.session_state.image_files[st.session_state.state_image], dsize = (350, 350))
            st.session_state.imageb = filters_transformations().gamma_correction(image = st.session_state.imagea, gamma = 0.5, rate = 1)
            st.session_state.imagec = filters_transformations().gamma_correction(image = st.session_state.imagea, gamma = 2.0, rate = 1)

        col1, col2,  = st.columns(spec = [4, 6])
        
        with col1:
            st.image(image = st.session_state.imagea, use_column_width = True, caption = 'Imagem Original')
        with col2:
            st.markdown(body = '***')
            st.subheader(body = 'Informações da radiografia:')     
            st.markdown(body = '**```Sexo```**: ' + st.session_state.dataframe['Sex'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Idade```**: ' + str(st.session_state.dataframe['Age'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])]) + ' anos')
            st.markdown(body = '**```Posicionamento```**: ' + st.session_state.dataframe['Frontal/Lateral'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Incidência```**: ' + st.session_state.dataframe['AP/PA'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Rotulação```**: ' + st.session_state.dataframe['Label'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '***')
        
        explanation() 
                    
        col1, col2 = st.columns(spec = [4, 6])
        
        with col1: 
            st.image(image = st.session_state.imageb, use_column_width = True, caption = 'Imagem Modificada')
        with col2:
            st.markdown(body = '***')
            st.markdown(body = '**Escolha uma opção adequada de acordo com a imagem em análise**:')
            value1 = st.selectbox(label = '', options = answers_labels,
                                  key = 'selectbox1', index = st.session_state.answers_labels[2][0])
            st.markdown(body = '***')
            st.session_state.answers_labels[2][0] = answers_labels.index(value1)
        
        col1, col2 = st.columns(spec = [4, 6])
        
        with col1: 
            st.image(image = st.session_state.imagec, use_column_width = True, caption = 'Imagem Modificada')
        with col2:
            st.markdown(body = '***')
            st.markdown(body = '**Escolha uma opção adequada de acordo com a imagem em análise**:')
            value2 = st.selectbox(label = '', options = answers_labels,
                                  key = 'selectbox2', index = st.session_state.answers_labels[2][1])
            st.markdown(body = '***')
            st.session_state.answers_labels[2][1] = answers_labels.index(value2)
            st.session_state.answers_labels[2][2] = 'null'
        
        st.session_state.reference_image[2] = 'data/images/' + st.session_state.image_files[st.session_state.state_image]

    if st.session_state['question_level'] == 4:

        my_bar.progress(0.80)
        
        col1, col2 = st.columns(spec = [8, 2])
        
        with col1:
            st.header(body = 'Estratégia de aumento de dados: ```Transformações não Geométricas```')
        with col2:
            state_images = st.button(label = 'Gerar Novas Imagens', key = 'button1', help = help_text)
            if state_images:
                st.session_state.state_image = change_images(st.session_state.state_image)
        
        if (st.session_state.state_image_old != st.session_state.state_image):
            st.session_state.state_image_old = st.session_state.state_image
            st.session_state.rate_old = st.session_state.rate

            st.session_state.imagea = image_manipulation().organize_image(path_image = 'data/images/' + st.session_state.image_files[st.session_state.state_image], dsize = (350, 350))
            st.session_state.imageb = filters_transformations().log_transformation(image = st.session_state.imagea)
            st.session_state.imagec = filters_transformations().adaptative_histogram_equalization(image = st.session_state.imagea)
            st.session_state.imaged = filters_transformations().sharpening(image = st.session_state.imagea)

        col1, col2,  = st.columns(spec = [4, 6])
        
        with col1:
            st.image(image = st.session_state.imagea, use_column_width = True, caption = 'Imagem Original')
        with col2:
            st.markdown(body = '***')
            st.subheader(body = 'Informações da radiografia:')     
            st.markdown(body = '**```Sexo```**: ' + st.session_state.dataframe['Sex'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Idade```**: ' + str(st.session_state.dataframe['Age'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])]) + ' anos')
            st.markdown(body = '**```Posicionamento```**: ' + st.session_state.dataframe['Frontal/Lateral'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Incidência```**: ' + st.session_state.dataframe['AP/PA'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Rotulação```**: ' + st.session_state.dataframe['Label'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '***')
        
        explanation()
                    
        col1, col2 = st.columns(spec = [4, 6])
        
        with col1: 
            st.image(image = st.session_state.imageb, use_column_width = True, caption = 'Imagem Modificada')
        with col2:
            st.markdown(body = '***')
            st.markdown(body = '**Escolha uma opção adequada de acordo com a imagem em análise** (Transformação Logarítmica):')
            value1 = st.selectbox(label = '', options = answers_labels,
                                  key = 'selectbox1', index = st.session_state.answers_labels[3][0])
            st.markdown(body = '***')
            st.session_state.answers_labels[3][0] = answers_labels.index(value1)

        col1, col2 = st.columns(spec = [4, 6])
        
        with col1: 
            st.image(image = st.session_state.imagec, use_column_width = True, caption = 'Imagem Modificada')
        with col2:
            st.markdown(body = '***')
            st.markdown(body = '**Escolha uma opção adequada de acordo com a imagem em análise** (Equalização de Histograma):')
            value2 = st.selectbox(label = '', options = answers_labels,
                                  key = 'selectbox2', index = st.session_state.answers_labels[3][1])
            st.markdown(body = '***')
            st.session_state.answers_labels[3][1] = answers_labels.index(value2)
        
        col1, col2 = st.columns(spec = [4, 6])
        
        with col1: 
            st.image(image = st.session_state.imaged, use_column_width = True, caption = 'Imagem Modificada')
        with col2:
            st.markdown(body = '***')
            st.markdown(body = '**Escolha uma opção adequada de acordo com a imagem em análise** (Aguçamento):')
            value3 = st.selectbox(label = '', options = answers_labels,
                                  key = 'selectbox3', index = st.session_state.answers_labels[3][2])
            st.markdown(body = '***')
            st.session_state.answers_labels[3][2] = answers_labels.index(value3)
        
        st.session_state.reference_image[3] = 'data/images/' + st.session_state.image_files[st.session_state.state_image]
             
    if st.session_state['question_level'] == 5:

        my_bar.progress(1.0)
        
        col1, col2 = st.columns(spec = [8, 2])
        
        with col1:
            st.header(body = 'Estratégia de aumento de dados: ```Filtros de Suavização```')
        with col2:
            state_images = st.button(label = 'Gerar Novas Imagens', key = 'button1', help = help_text)
            if state_images:
                st.session_state.state_image = change_images(st.session_state.state_image)
        
        if (st.session_state.state_image_old != st.session_state.state_image):
            st.session_state.state_image_old = st.session_state.state_image
            st.session_state.rate_old = st.session_state.rate

            st.session_state.imagea = image_manipulation().organize_image(path_image = 'data/images/' + st.session_state.image_files[st.session_state.state_image], dsize = (350, 350))
            st.session_state.imageb = filters_transformations().mean_filter(image = st.session_state.imagea)
            st.session_state.imagec = filters_transformations().median_filter(image = st.session_state.imagea)
            st.session_state.imaged = filters_transformations().gaussian_filter(image = st.session_state.imagea)

        col1, col2,  = st.columns(spec = [4, 6])
        
        with col1:
            st.image(image = st.session_state.imagea, use_column_width = True, caption = 'Imagem Original')
        with col2:
            st.markdown(body = '***')
            st.subheader(body = 'Informações da radiografia:')     
            st.markdown(body = '**```Sexo```**: ' + st.session_state.dataframe['Sex'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Idade```**: ' + str(st.session_state.dataframe['Age'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])]) + ' anos')
            st.markdown(body = '**```Posicionamento```**: ' + st.session_state.dataframe['Frontal/Lateral'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Incidência```**: ' + st.session_state.dataframe['AP/PA'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '**```Rotulação```**: ' + st.session_state.dataframe['Label'][list(st.session_state.dataframe['Path']).index('images/' + st.session_state.image_files[st.session_state.state_image])])
            st.markdown(body = '***')
        
        explanation()  
                    
        col1, col2 = st.columns(spec = [4, 6])
        
        with col1: 
            st.image(image = st.session_state.imageb, use_column_width = True, caption = 'Imagem Modificada')
        with col2:
            st.markdown(body = '***')
            st.markdown(body = '**Escolha uma opção adequada de acordo com a imagem em análise** (Filtro da Média):')
            value1 = st.selectbox(label = '', options = answers_labels,
                                  key = 'selectbox1', index = st.session_state.answers_labels[4][0])
            st.markdown(body = '***')
            st.session_state.answers_labels[4][0] = answers_labels.index(value1)

        col1, col2 = st.columns(spec = [4, 6])
        
        with col1: 
            st.image(image = st.session_state.imagec, use_column_width = True, caption = 'Imagem Modificada')
        with col2:
            st.markdown(body = '***')
            st.markdown(body = '**Escolha uma opção adequada de acordo com a imagem em análise** (Filtro da Mediana):')
            value2 = st.selectbox(label = '', options = answers_labels,
                                  key = 'selectbox2', index = st.session_state.answers_labels[4][1])
            st.markdown(body = '***')
            st.session_state.answers_labels[4][1] = answers_labels.index(value2)
            
        col1, col2 = st.columns(spec = [4, 6])
        
        with col1: 
            st.image(image = st.session_state.imaged, use_column_width = True, caption = 'Imagem Modificada')
        with col2:
            st.markdown(body = '***')
            st.markdown(body = '**Escolha uma opção adequada de acordo com a imagem em análise** (Filtro Gaussiano):')
            value3 = st.selectbox(label = '', options = answers_labels,
                                  key = 'selectbox3', index = st.session_state.answers_labels[4][2])
            st.markdown(body = '***')
            st.session_state.answers_labels[4][2] = answers_labels.index(value3)
        
        st.session_state.reference_image[4] = 'data/images/' + st.session_state.image_files[st.session_state.state_image]
    
    col1, col2, col3 = st.columns(spec = [2, 6, 2])
    
    if st.session_state.question_level != 1:
        with col1:
            state = st.button('Pergunta Anterior')
            if state:
                st.session_state.question_level -= 1
                st.session_state.state_image_old = 99
                with st.spinner(text = 'Aguarde. Estamos preparando tudo para você!'):
                    st_autorefresh(interval = 1, limit = 2, key = 'refresh_back_question')
    if (st.session_state.answers_labels[st.session_state.question_level - 1][0] == 0) or (st.session_state.answers_labels[st.session_state.question_level - 1][1] == 0) or (st.session_state.answers_labels[st.session_state.question_level - 1][2] == 0): 
        with col2:
            st.warning(body = 'Responda todas as perguntas para avançar.')
    elif st.session_state.question_level != 5:
        with col2:
            st.success(body = 'Respostas registradas! Avance para a próxima pergunta.')
    elif not verify_send_firebase:
        with col2:
            st.success(body = 'As respostas já foram enviadas para o servidor. Deseja reenviá-las?')
    else:
        with col2:
            st.success(body = 'Respostas registradas! Por favor, envie suas respostas.')
           
    if (st.session_state.answers_labels[st.session_state.question_level - 1][0] != 0) and (st.session_state.answers_labels[st.session_state.question_level - 1][1] != 0) and (st.session_state.answers_labels[st.session_state.question_level - 1][2] != 0) and (st.session_state.question_level != 5):
        with col3:
            state = st.button(label = 'Próxima Pergunta')
            if state:
                st.session_state.question_level += 1
                st.session_state.state_image_old = 99
                with st.spinner(text = 'Aguarde. Estamos preparando tudo para você!'):
                    st_autorefresh(interval = 1, limit = 2, key = 'refresh_next_question')
    
    end_research = False 
    
    if (st.session_state.question_level == 5) and (st.session_state.answers_labels[st.session_state.question_level - 1][0] != 0) and (st.session_state.answers_labels[st.session_state.question_level - 1][1] != 0) and (st.session_state.answers_labels[st.session_state.question_level - 1][2] != 0) and verify_send_firebase:
        with col3:
            end_research = st.button(label = 'Enviar Respostas')
    elif not verify_send_firebase:
        with col3:
            end_research = st.button(label = 'Reenviar Respostas')
            if end_research: 
                verify_send_firebase = True
    
    if not end_research:
        st.markdown(body = '''<a href='#inicio'>
                                <img src = 'https://user-images.githubusercontent.com/58775072/148143380-98ed5a88-4480-4850-af4f-1aee3e2b829c.png' 
                                width = 50px alt = 'Voltar para o topo da página' style = "float: right;">
                              </a>''', unsafe_allow_html = True)

    return end_research, verify_send_firebase, st.session_state.answers_labels, st.session_state.reference_image
    


    

    