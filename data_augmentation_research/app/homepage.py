import streamlit as st

def app(name, university, email, skill, radiology_familiarity, pdi_familiarity): 

    
    familiarity_helper = ['Selecione uma opção', 'Alto grau de familiaridade', 
                          'Médio grau de familiaridade', 'Baixo grau de familiaridade']

    st.image(image = 'data/homepage-fig.jpg', use_column_width = True)
    st.header(body = 'Pesquisa sobre o uso de processamento de imagens para otimização da análise de distúrbios '
                     'pulmonares em radiografias torácicas')
    st.markdown(body = '**Objetivo da Pesquisa**: analisar quais estratégias de aumentos de dados são '
                       'eficazes em radiografias de distúrbios pulmonares. A partir dessas estratégias, '
                       'por exemplo, é possível gerar mais imagens úteis para treinamento de algoritmos '
                       'inteligentes fundamentados no estado da arte do *Deep Learning*. Desse modo, '
                       'convidamos você a nos ajudar na análise da qualidade das estratégias propostas '
                       'ao longo da pesquisa.')
    st.markdown(body = '**Metodologia**: será analisada diversas imagens médicas de radiografias torácicas, '
                       'e para cada uma delas, haverá uma estratégia de aumento de dados associada '
                       '[[1](https://www.ncbi.nlm.nih.gov/pmc/articles/pmc5977656/), [2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7956964/)]. '
                       'A sua contribuição será dada através da votação da qualidade de análise das imagens, na qual ela pode ser qualificada '
                       'como eficaz, neutra ou deteriorativa através da inspeção visual da imagem como forma de diagnóstico. '
                       'A base de dados utilizada possui 45 radiografias de tórax associada a seis distúrbios pulmonares e '
                       'imagens de pulmões saudáveis. Fique a vontade para gerar quantas imagens achar necessário durante a '
                       'análise de cada uma das estratégias de aumento de dados.')
    st.markdown(body = '**Contextualização**: as técnicas de aumento de dados geram diferentes versões de um conjunto de '
                       'dados real de uma forma artificial para aumentar seu tamanho. Os modelos de visão computacional '
                       'usam estratégia de aumento de dados para lidar com a escassez e falta de diversidade dos dados. '
                       'Entretanto, deve haver um cuidado nas estratégias utilizadas, para não descaracterizar os padrões '
                       'contidos na imagem original (principalmente imagens médicas) '
                       '[[3](https://latamt.ieeer9.org/index.php/transactions/article/download/2813/349)].')
    st.markdown(body = 'Essa pesquisa faz parte de um projeto científico (PIBIC) continuado '
                       '[[4](https://biblioteca.sbrt.org.br/articlefile/2915.pdf)] desenvolvido por alunos de **Engenharia '
                       'Elétrica**, em parceria com alunos do curso de **Medicina**, ambos vinculados à **Universidade '
                       'Federal de Campina Grande (UFCG)**.')

    with st.expander(label = 'Clique aqui para saber mais sobre os membros do projeto'):
        st.markdown(body = '***')

        col1, col2 = st.columns(spec = [1,8])

        with col1:
            st.image(image = 'https://user-images.githubusercontent.com/58775072/145095066-08ec2fd0-0e30-44b4-998d-c3e78d4409e0.jpg', 
                     width = 120)
        with col2:
            st.markdown(body = '**Nome**: [Luciana Ribeiro Veloso](http://lattes.cnpq.br/2498050002491677)')
            st.markdown(body = '**Formação Acadêmica**: Doutora em Engenharia Elétrica (UFCG)')
            st.markdown(body = '**Biografia**: Possui graduação em Engenharia Elétrica pela Universidade Federal da '
                               'Paraíba (1995), mestrado em Engenharia Elétrica pela Universidade Federal da Paraíba '
                               '(1998) e Doutorado em Engenharia Elétrica pela Universidade Federal de Campina Grande '
                               '(2009). Atualmente é professora da Universidade Federal de Campina Grande. Tem '
                               'experiência na área de Engenharia Elétrica, com ênfase em Processamento de Imagens, ' 
                               'atuando principalmente nos seguintes temas: Reconhecimento de Palavras Manuscritas, '
                               'Processamento de Imagens e Reconhecimento de Padrões.')

        st.markdown(body = '***')
        
        col1, col2 = st.columns(spec = [1,8])
        
        with col1:
            st.image(image = 'https://user-images.githubusercontent.com/58775072/145095291-61c00da5-211a-4b86-9728-77594fbf60e9.gif', 
                     width = 120)
        with col2:
            st.markdown(body = '**Nome**: [Leo de Lima Araújo](http://lattes.cnpq.br/2093156188518982)')
            st.markdown(body = '**Formação Acadêmica**: Mestrando em Engenharia Elétrica (UFCG)')
            st.markdown(body = '**Biografia**: Pós-graduando em Engenharia Elétrica pela Universidade Federal de Campina '
                               'Grande. Trabalha sobretudo com Aprendizado Profundo (*Deep Learning*) e Visão Computacional '
                               'e atua no desenvolvimento pesquisas na área de saúde, envolvendo o diagnóstico automatizado '
                               'de Distúrbios Pulmonares e *COVID-19*, além de aplicações industriais, identificando defeitos '
                               'em televisores/monitores na linha de produção.')
        
        st.markdown(body = '***')
        
        col1, col2 = st.columns(spec = [1,8])
        
        with col1:
            st.image(image = 'https://user-images.githubusercontent.com/58775072/145095535-fb303146-1c30-40bf-85d2-6b90e4a68ab3.png', 
                     width = 120)
        with col2:
            st.markdown(body = '**Nome**: [Alysson Machado de Oliveira Barbosa](http://lattes.cnpq.br/0536933202710340)')
            st.markdown(body = '**Formação Acadêmica**: Graduando em Engenharia Elétrica (UFCG)')
            st.markdown(body = '**Biografia**: Possui segundo grau completo na instituição Colégio Alfredo Dantas (CAD) '
                               'e atualmente faz graduação em Engenharia Elétrica pela Universidade Federal de Campina '
                               'Grande (UFCG). Desenvolveu um projeto intitulado "Álgebra Linear com Python" , visando aplicar '
                               'computacionalmente conceitos de Álgebra Linear. Atualmente é ativo no desenvolvimento '
                               'de trabalhos em Inteligência Artificial (IA) com conhecimentos em *Machine Learning* ' 
                               'e *Deep Learning*. Tem experiência no desenvolvimento de páginas web, programação em C, '
                               'C++, Python e versionamento de projetos com Git. Desenvolveu atividades relevantes '
                               'para a comunidade acadêmica como o evento *Academic Talk* em parceria com o capítulo '
                               'PELS/IAS da UFCG associado ao Instituto de Engenheiros Elétricos e Eletrônicos (IEEE).')

        st.markdown(body = '***')
       
        col1, col2 = st.columns(spec = [1,8])
        
        with col1:
            st.image(image = 'https://user-images.githubusercontent.com/58775072/145095714-6f2f2464-918b-45be-bc62-c266fa0eaedb.gif', 
                    width = 120)
        with col2:
            st.markdown(body = '**Nome**: [Sarah Laís Silva de Freitas](http://lattes.cnpq.br/2609835503203134)')
            st.markdown(body = '**Formação Acadêmica**: Graduanda em Medicina (UFCG)')
            st.markdown(body = '**Biografia**: Acadêmica de medicina da Universidade Federal de Campina Grande; '
                               'bolsista do Programa de Educação pelo Trabalho para a Saúde - Interprofissionalidades; '
                               'diretora de publicação, pesquisa e extensão da Liga Acadêmica de Semiologia e '
                               'Propedêutica Médica da Universidade Federal de Campina Grande; membro da Federação '
                               'Internacional de Associações de Estudantes de Medicina.')
        
        st.markdown(body = '***')
        
        col1, col2 = st.columns(spec = [1,8])
        
        with col1:
            st.image(image = 'https://user-images.githubusercontent.com/58775072/145095881-a2101b2b-b28f-4c01-ac2a-398887fdc04f.gif', 
                     width = 120)
        with col2:
            st.markdown(body = '**Nome**: [Marcelo Victor Ferreira Gurgel](http://lattes.cnpq.br/4915896131680959)')
            st.markdown(body = '**Formação Acadêmica**: Graduanda em Medicina (UFCG)')
            st.markdown(body = '**Biografia**: Acadêmico de medicina na Universidade Federal de Campina Grande. Diretor '
                               'de marketing da Liga Acadêmica de oftalmologia. Participante ativo da Liga Médico-Acadêmica '
                               'de Pneumologia. Extensionista do Projeto Hanseníase sem estigmas.')
     
    st.markdown(body = '***')
    st.markdown(body = '### Informações básicas sobre o respondente')
    name = st.text_input(label = 'Informe seu nome:', value = name)
    university = st.text_input(label = 'Informe a sua universidade:', value = university)
    email = st.text_input(label = 'Informe o seu e-mail para contato:', value = email)
    skill = st.text_input(label = 'Informe o seu curso/formação ou área de estudo:', value = skill)
    radiology_familiarity = st.selectbox(label = 'Qual o seu grau de experiência com a análise de radiografias?', 
                                         options = familiarity_helper, index = familiarity_helper.index(radiology_familiarity))
    pdi_familiarity = st.selectbox(label = 'Qual o seu grau de experiência com o processamento de imagens digitais?', 
                                   options = familiarity_helper, index = familiarity_helper.index(pdi_familiarity))
    
    st.subheader(body = 'Revise suas informações')
    st.markdown(body = '**Nome**: ' + name)
    st.markdown(body = '**Universidade de formação**: ' + university)
    st.markdown(body = '**E-mail**: ' + email)
    st.markdown(body = '**Curso ou área de atuação**: ' + skill)
    if radiology_familiarity == 'Selecione uma opção':
        st.markdown(body = '**Grau de experiência com a Radiologia**: ')
    else:
        st.markdown(body = '**Grau de experiência com a Radiologia**: ' + radiology_familiarity)
    if pdi_familiarity == 'Selecione uma opção':
        st.markdown(body = '**Grau de experiência com Processamento de Imagens**: ')
    else:
        st.markdown(body = '**Grau de experiência com Processamento de Imagens**: ' + pdi_familiarity)
    st.markdown(body = '***')

    col1, col2 = st.columns(spec = [8, 2])

    with col1:
        state = st.checkbox(label = 'Autorizo que as as respostas coletadas sejam utilizadas em projetos e artigos acadêmicos de '
                                    'iniciação científica.')

    state_research = False

    if name.strip() != '' and university.strip() != '' and skill.strip() != '' and email.strip() != '' and radiology_familiarity != 'Selecione uma opção' and pdi_familiarity != 'Selecione uma opção' and state == True:
        with col2:
            state_research = st.button(label = 'Iniciar Pesquisa')
    elif name.strip() != '' and university.strip() != '' and skill.strip() != '' and email.strip() != ''  and radiology_familiarity != 'Selecione uma opção' and pdi_familiarity != 'Selecione uma opção' and state == False:
        st.warning(body = 'Por favor, autorize o uso de sua avaliação na pesquisa para iniciá-la.')
    elif name.strip() == '' and university.strip() == '' and skill.strip() == '' and email.strip() == '' and radiology_familiarity == 'Selecione uma opção' and pdi_familiarity == 'Selecione uma opção' and state == True:
        st.warning(body = 'Por favor, preencha as suas informações básicas antes de iniciar a pesquisa.')
    elif (name.strip() == '' or university.strip() == '' or skill.strip() == '' or email.strip() == '' or radiology_familiarity == 'Selecione uma opção' or pdi_familiarity == 'Selecione uma opção') and state == True:
        st.warning(body = 'Certifique-se de ter preenchido todas as suas informações básicas.')

    get_info_homepage = dict()
    get_info_homepage['name'] = name
    get_info_homepage['university'] = university
    get_info_homepage['email'] = email
    get_info_homepage['skill'] = skill
    get_info_homepage['radiology_familiarity'] = radiology_familiarity
    get_info_homepage['pdi_familiarity'] = pdi_familiarity
    get_info_homepage['state'] = state
    get_info_homepage['state_research'] = state_research

    if not state_research:
        st.markdown(body = '''<a href='#inicio'>
                                <img src = 'https://user-images.githubusercontent.com/58775072/148143380-98ed5a88-4480-4850-af4f-1aee3e2b829c.png' 
                                width = 50px alt = 'Voltar para o topo da página' style = "float: right;">
                              </a>''', unsafe_allow_html = True)
            
    return get_info_homepage
    
