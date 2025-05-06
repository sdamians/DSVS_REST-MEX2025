import re
import unicodedata
from datasets import Dataset


def df_to_dataset(df, columns : list, text_column: str):
    dataset_dict = { column: df[column].to_numpy() for column in columns if column != text_column }

    df['review'] = df[text_column].apply(clean_characters)
    df['review'] = df['review'].apply(clean_text)
    df['review'] = df['review'].apply(smart_capitalize)

    dataset_dict['reviews'] = df['review'].to_list()

    return Dataset.from_dict(dataset_dict)

def smart_capitalize(text):
    # Busca la primera letra alfabética (incluyendo letras acentuadas y ñ)
    match = re.search(r'([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ])', text)
    if not match:
        return text  # No hay letra alfabética

    idx = match.start()
    return text[:idx] + text[idx].upper() + text[idx+1:]

def clean_text(text):
    text = re.sub(r"\[[^()]*\]|\([^()]*\)|{[^()]*}", "", text)
    
    text = re.sub(r"' (.+) '", r"'\1'", text)
    text = re.sub(r'" (.+) "', r'"\1"', text)
    text = re.sub(r"(\d+) : (\d+)", r"\1:\2", text)
    text = re.sub(r"\s+([:,!\?%\.])", r"\1", text)
    text = re.sub(r"([¿¡$])\s+", r"\1", text)
    
    text = re.sub(r"^,+ ", "", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r"[\(\)\[\]{}]", "", text)

    text = re.sub(r"\s+", " ", text)
    #print(f"clean_words: {text}")
    return text.strip()

def clean_characters(text):
    text = re.sub(r"`|‘|’|´", "'", text) # Remove quotes with other formats
    text = re.sub(r"''", "", text) # Remove double quotes
    text = re.sub(r"\xad|\x81|…|_|\u200b", " ", text) # Remove weird empty characters
    text = re.sub("[-—–]+", "-", text) # Normalize dashes with other formats
    text = re.sub(r"\.+", ".", text) # Periods
    
    # Normalize letters with accents, and ñ
    text = re.sub(r"б|Ã¡|Ã¡|à", "á", text)
    text = re.sub(r"Ã©|è|й", "é", text)
    text = re.sub(r"у|Ã³|í³|ò", "ó", text)
    text = re.sub(r"ъ|Ãº|ù", "ú", text)

    text = re.sub(r"Ã|À", "Á", text)
    text = re.sub(r"Ã‰|È", "É", text)
    text = re.sub(r"Ã|Ì", "Í", text)
    text = re.sub(r"Ã“|Ò", "Ó", text)
    text = re.sub(r"Ãš|Ù", "Ú", text)
    
    text = re.sub(r"Ã‘", "Ñ", text)
    text = re.sub(r"с|Ã±|a±|í±", "ñ", text)
    text = re.sub(r"е", "e", text)
    text = re.sub(r"Â¿|Ї", "¿", text)
    text = re.sub(r"éÂ¼", "üe", text)  
    
    if re.search(r"н|Ã ­|Ã|­�|ì", text) is not None: 
        text = re.sub(r"н|Ã ­|Ã|­�|ì", "í", text)
        text = re.sub(r"í ", "í", text)

    symbols_pattern = r"([\.,¡!¿\?\[\]\(\)%\$:'_\"-])"
    # Set an empty space to special symbols
    text = re.sub(symbols_pattern, r' \1 ', text)
    # Remove all other special symbols
    text = re.sub(r"[^A-Za-z0-9ÁÉÍÓÚáéíóúüÑñ \.,¡!¿\?\[\]\(\)%\$:'_\"-]", '', text)
    # Remove "Más" at the end of the sentence
    text = re.sub(r'\b[Mm][áa]s\b[\.\!\?]?$', '', text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return unicodedata.normalize('NFC', text.strip())