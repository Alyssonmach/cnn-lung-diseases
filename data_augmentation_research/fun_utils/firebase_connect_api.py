import random, pickle

def get_firebase_config():
    '''get access to firebase configuration dictionary'''

    firebaseConfig = {'apiKey': 'AIzaSyAdGA677WSlWvCYri8XMuM0SvSB0Jn4-z0',
                    'authDomain': 'data-aug-research.firebaseapp.com',
                    'databaseURL': 'https://data-aug-research-default-rtdb.firebaseio.com',
                    'projectId': 'data-aug-research',
                    'storageBucket': 'data-aug-research.appspot.com',
                    'messagingSenderId': '661305969997',
                    'appId': '1:661305969997:web:d4ccfde7a9d7d501bcbb04',
                    'measurementId': 'G-9LH3F3ZB5K'}
    
    return firebaseConfig

def get_random_string(k = 8):
    '''
    creates a string with k random characters and no symbols

    Args:
        k (int) --> maximum number of characters in the string
    Returns:
        random_string (str) --> randomly generated string
    '''
    
    alphabet_lower = 'abcdefghijklmnopqrstuvwxyz'
    alphabet_upper = alphabet_lower.upper()
    numbers = '123456789'

    join_strings = alphabet_lower + alphabet_upper + numbers
    random_string = ''.join(random.sample(population = join_strings, k = 8))

    return random_string

def save_dict_in_json_format(data, json_file = 'data.json'):
    '''
    save a pythonic dictionary to a json file
    
    Args:
        data (dict) --> pythonic dictionary
        json_file (str) --> json file name
    '''

    try:
        with open(json_file, 'wb') as file_open:
            pickle.dump(data, file_open).encode('utf8')
        return True
    except:
        return False

def read_json_in_dict_format(json_file = 'data.json'):
    '''
    open a json file in a pythonic dictionary
    
    Args:
        json_file (str) --> json file name
    Returns:
        data (dict) --> pythonic dictionary
    '''

    try:
        with open(json_file, 'rb') as file_open:
            json_to_dict = pickle.load(file_open)
        return json_to_dict
    except:
        return dict()