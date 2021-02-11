# baixar dados a partir de urls 
import urllib.request
# organização de dataframes 
import pandas as pd
# particionando os dados em treinamento, validação e teste  
from sklearn.model_selection import train_test_split

def data_download(link, file_name):
    '''baixar dados a partir de urls'''
    
    import urllib.request
    # download de file_name
    urllib.request.urlretrieve(link, file_name) 
    
    return None 

def organize_csv(csv_file):
    '''organizando dataframe para treinamento em rede neural convolucional'''
    
    # carregando o dataframe
    dataframe = pd.read_csv(csv_file)
    
    # renomeando alguns colunas 
    dataframe = dataframe.rename(columns = {'Finding Labels': 'finding_labels', 'Patient ID': 'patient_id'})
    
    # removendo colunas desnecessárias na etapa de treinamento 
    dataframe = dataframe.drop(columns = ['Follow-up #', 'Patient Age', 'Patient Gender', 'View Position',
                                          'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]']) 
    
    # mantendo os dados consistentes com cada um dos rótulos  
    dataframe = dataframe[(dataframe.finding_labels == 'No Finding') |
                          (dataframe.finding_labels == 'Atelectasis') |
                          (dataframe.finding_labels == 'Cardiomegaly') |
                          (dataframe.finding_labels == 'Consolidation') |
                          (dataframe.finding_labels == 'Edema') |
                          (dataframe.finding_labels == 'Effusion') |
                          (dataframe.finding_labels == 'Emphysema') |
                          (dataframe.finding_labels == 'Fibrosis') |
                          (dataframe.finding_labels == 'Hernia') |
                          (dataframe.finding_labels == 'Infiltration') |
                          (dataframe.finding_labels == 'Mass') |
                          (dataframe.finding_labels == 'Nodule') |
                          (dataframe.finding_labels == 'Pleural_Thickening') |
                          (dataframe.finding_labels == 'Pneumonia') |
                          (dataframe.finding_labels == 'Pneumothorax')]
    dataframe = pd.get_dummies(data = dataframe, prefix = '', columns = ['finding_labels'])
    
    return dataframe

def download_images():
    '''baixando as imagens do servidor'''
    
    # links com cada um dos arquivos compactados 
    links = [
            'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
            'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
            'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
            'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
            'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
            'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
            'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
            'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
            'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
            'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
            'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
            'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz']
    
    # iterando entre os links e baixando os dados 
    for idx, link in enumerate(links):
        fn = 'images_%02d.tar.gz' % (idx+1)
        print('downloading'+fn+'...')
        urllib.request.urlretrieve(link, fn)  

    print("Download complete. Please check the checksums")
    
    return None 
 
def train_validation_test_split(dataframe):
    '''particionando os dados em treinamento, validação e teste'''
    
    # remoção de quantidades exagerageradas com rótulos normais (desbalanceamento de dados) 
    rowns = dataframe.loc[(dataframe['_No Finding'] == 1)].index
    dataframe = dataframe.drop(rowns[5000:], axis = 0)
    
    # particionando os dados em treinamento, validação e teste  
    train_df, test_df = train_test_split(dataframe, test_size = 0.1, random_state = 42)
    train_df, validation_df = train_test_split(train_df, test_size = 0.1, random_state = 42)
    
    return train_df, validation_df, test_df


