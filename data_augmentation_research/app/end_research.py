import streamlit as st

def app(user_info):

    st.image(image = 'data/homepage-fig.jpg', use_column_width = True)

    try:
        name = user_info['name'].split(' ')[0]
    except:
        name = user_info['name']
    email = user_info['email']   

    st.title(body = f'Obrigado pela contribuição {name}!')
    st.markdown(body = 'A sua ajuda foi fundamental para a nossa pesquisa científica. Em breve entraremos em contato '
                      f'pelo e-mail: [{email}](mailto:{email}).')

    st.markdown(body = '### Referências Bibliográficas')
    st.markdown(body = '* [[1](https://www.ncbi.nlm.nih.gov/pmc/articles/pmc5977656/)] Hussain, Z., Gimenez, F., Yi, D., & Rubin, D. (2017). '
                       'Differential data augmentation techniques for medical imaging classification tasks. In AMIA annual symposium '
                       'proceedings (Vol. 2017, p. 979). American Medical Informatics Association.')
    st.markdown(body = '* [[2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7956964/)] Elgendi, M., Nasir, M., Tang, Q., Smith, D., Grenier, '
                       'J.P., Batte, C., Spieler, B., Leslie, W., Menon, C., Fletcher, R., Howard, N., Ward, R., Parker, W., & Nicolaou, '
                       'S. (2021). The Effectiveness of Image Augmentation in Deep Learning Networks for Detecting COVID-19: A Geometric '
                       'Transformation Perspective. Frontiers in Medicine, 8, 153.')
    st.markdown(body = '* [[3](https://latamt.ieeer9.org/index.php/transactions/article/download/2813/349)] Evangelista, L. G. C., & Guedes, '
                       'E. B. (2019). Ensembles of convolutional neural networks on computer-aided pulmonary tuberculosis detection. IEEE '
                       'Latin America Transactions, 17(12), 1954-1963.')
    st.markdown(body = '* [[4](https://biblioteca.sbrt.org.br/articlefile/2915.pdf)] Machado, A., Araújo, L., & Veloso, L. Classificação de '
                       'Distúrbios Pulmonares em Radiografias de Tórax Usando Redes Convolucionais.')

    question_text = st.text_area(label = 'Alguma dúvida? Entre em contanto conosco:', help = f'Entraremos em contanto pelo e-mail {email} '
                                                                                              'brevemente')
    
    st.balloons()

    return question_text

