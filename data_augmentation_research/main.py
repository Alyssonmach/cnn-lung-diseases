from streamlit_autorefresh import st_autorefresh
import streamlit as st
from app import homepage, research, end_research
from fun_utils import firebase_connect_api
import time, pyrebase

st.set_page_config(page_title = 'Pesquisa UFCG', layout = 'wide', page_icon = '📚')
st.markdown("<div id='inicio' style='visibility: hidden'></div>", unsafe_allow_html = True)
st.markdown('<style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style>', 
            unsafe_allow_html = True)

padding = 1
st.markdown(f"""<style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style>""", unsafe_allow_html = True)

if 'authorization' not in st.session_state:
    st.session_state['authorization'] = False
    st.session_state['state_research'] = False
    st.session_state['end_research'] = False
    st.session_state['name'] = ''
    st.session_state['university'] = ''
    st.session_state['email'] = ''
    st.session_state['skill'] = ''
    st.session_state['radiology_familiarity'] = 'Selecione uma opção'
    st.session_state['pdi_familiarity'] = 'Selecione uma opção'
    st.session_state['get_data'] = {'user_info': dict(), 'user_data': list(), 'image_reference': list()}
    st.session_state['question_user'] = {'user_info': dict(), 'question': str()}
    st.session_state['verify_send_firebase'] = True
    st.session_state['json_names'] = {'data_filename': str(), 'question_filename': str()}

PAGES = {'homepage': homepage, 'research': research, 'end_research': end_research}

firebase = pyrebase.initialize_app(firebase_connect_api.get_firebase_config())
database, auth, storage = firebase.database(), firebase.auth(), firebase.storage()

if (not st.session_state.authorization or not st.session_state.state_research) and not st.session_state.end_research:
    get_info_homepage = PAGES['homepage'].app(name = st.session_state.name, university = st.session_state.university, 
                                              email = st.session_state.email, skill = st.session_state.skill, 
                                              radiology_familiarity = st.session_state.radiology_familiarity, 
                                              pdi_familiarity = st.session_state.pdi_familiarity)
    
    st.session_state.name = get_info_homepage['name']
    st.session_state.university = get_info_homepage['university'] 
    st.session_state.email = get_info_homepage['email']
    st.session_state.skill = get_info_homepage['skill']
    st.session_state.radiology_familiarity = get_info_homepage['radiology_familiarity']
    st.session_state.pdi_familiarity = get_info_homepage['pdi_familiarity']
    st.session_state.authorization = get_info_homepage['state']
    st.session_state.state_research = get_info_homepage['state_research']
    st.session_state.get_data['user_info'] = get_info_homepage

if st.session_state.state_research:
    with st.spinner(text = 'Aguarde. Estamos preparando tudo para você!'):
        st_autorefresh(interval = 1, limit = 2, key = 'refresh_research')
        time.sleep(1)

    col1, col2 = st.columns(spec = [1, 9])
    with col1:
        back_homepage = st.button(label = 'Voltar', key = 'back_homepage1', help = 'Voltar para a página principal')

    if back_homepage:
        st.session_state.authorization = not back_homepage
        with st.spinner(text = 'Aguarde. Estamos preparando tudo para você!'):
            st_autorefresh(interval = 1, limit = 2, key = 'refresh_homepage1')
    
    st.session_state.end_research, st.session_state.verify_send_firebase, answers_labels, image_reference =  PAGES['research'].app(column_progress_bar = col2, 
                                                                                                                verify_send_firebase = st.session_state.verify_send_firebase)
    
    if st.session_state.end_research:
        st.session_state.state_research = False
    
    st.session_state.get_data['user_data'] = answers_labels
    st.session_state.get_data['image_reference'] = image_reference

if st.session_state.end_research:
    with st.spinner(text = 'Aguarde. Estamos enviando os dados para o Firebase...'):
        st_autorefresh(interval = 1, limit = 2, key = 'refresh_end_research')
        time.sleep(1)

    if st.session_state.verify_send_firebase:
        st.session_state.json_names['data_filename'] = st.session_state.get_data['user_info']['email'].replace(' ', '') + firebase_connect_api.get_random_string(k = 3) + '.json'
        firebase_connect_api.save_dict_in_json_format(data = st.session_state.get_data, json_file = st.session_state.json_names['data_filename'])
        
        storage.child('users_data/' + st.session_state.json_names['data_filename']).put(st.session_state.json_names['data_filename'])
        
        try:
            auth.create_user_with_email_and_password(email = st.session_state.get_data['user_info']['email'], password = firebase_connect_api.get_random_string(k = 8))
        except:
            pass
        
        st.session_state.verify_send_firebase = False

    question_text = PAGES['end_research'].app(user_info = st.session_state.get_data['user_info'])

    back_homepage = st.button('Voltar para a página principal', key = 'back_homepage2')
    if back_homepage:
        st.session_state.authorization = not back_homepage
        st.session_state.end_research = not back_homepage
        with st.spinner(text = 'Aguarde. Estamos preparando tudo para você!'):
            st_autorefresh(interval = 1, limit = 2, key = 'refresh_homepage2')

    if question_text != '' and not back_homepage:
        st.session_state.question_user['user_info'] = st.session_state.get_data['user_info']
        st.session_state.question_user['question'] = question_text
        st.session_state.json_names['question_filename'] = firebase_connect_api.get_random_string(k = 8) + '.json'
        
        firebase_connect_api.save_dict_in_json_format(data = st.session_state.question_user, 
                                                      json_file = st.session_state.json_names['question_filename'])
        
        storage.child('users_questions/' + st.session_state.json_names['question_filename']).put(st.session_state.json_names['question_filename'])