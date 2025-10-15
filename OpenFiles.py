import os
import pandas as pd
import nltk 
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords')
from collections import Counter
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import re

# negative polarity
# the r in front of the string indicates a raw string literal, which treats backslashes as literal characters
negative_polarity = r"data/op_spam_v1.4/negative_polarity" # fill in your own path

# listdir function: list all files and folders in a directory
# isfile function: check if a path is a file   
# join function: join one or more path components

# create a class to store the content of each file
class FileContent:
        def __init__(self, folder, subfolder, filename, content):
                self.folder = folder              # main folder
                self.subfolder = subfolder        # subfolder   
                self.filename = filename          # name of the file
                self.content = content            # content of the file
                # each argument is stored as an instance variable
                # each object will remember its own folder, subfolder, filename and content

# __repr__ function: tells how to object is printed when you use print()
        def __repr__(self):
                return f"FileContent(folder={self.folder}, subfolder={self.subfolder}, filename={self.filename}, content={self.content})"

# defnies a function that takes a starting folder path and returns a list of all subfolder paths
def load_files_recursive(base_folder):
    # content_list will store all FileContent objects
    content_list = []
    # iterate over everything (entry) inside the base_folder 
    for entry in os.listdir(base_folder):
        # create the full path by joining the base_folder path with the entry name
        path = os.path.join(base_folder, entry)
        # check if the path is a directory (another folder) or a file
        # if it's a directory (another folder), we need to go deeper and call the load_files_recursive function again
        if os.path.isdir(path):
            # call the function again with the new path
            # use extend to add all the files foudn inside that subfolder into the current list
            content_list.extend(load_files_recursive(path))
        # if the path is a file, we can open it and read its content
        elif os.path.isfile(path):
            # open the file and read its content
            # encoding="utf-8" ensures that special characters are read correctly. utf-8 is the most widely used 
            # encoding today. It can represent any character in the Unicode standard (text characters from any language).
            with open(path, "r", encoding="utf-8") as file:
                content = file.read() # reads its entire content into a string 
                folder = os.path.basename(os.path.dirname(base_folder)) # parent folder name
                subfolder = os.path.basename(base_folder) # current folder name
                filename = entry # file name itself
                content_list.append(FileContent(folder, subfolder, filename, content))
    return content_list

# this block runs only if the file is executed directly, not if it's imported

class FileLoader:
    @staticmethod
    # defnies a function that takes a starting folder path and returns a list of all subfolder paths
    def load_files_recursive(base_folder):
        # content_list will store all FileContent objects
        content_list = []
        # iterate over everything (entry) inside the base_folder 
        for entry in os.listdir(base_folder):
            # create the full path by joining the base_folder path with the entry name
            path = os.path.join(base_folder, entry)
            # check if the path is a directory (another folder) or a file
            # if it's a directory (another folder), we need to go deeper and call the load_files_recursive function again
            if os.path.isdir(path):
                # call the function again with the new path
                # use extend to add all the files foudn inside that subfolder into the current list
                content_list.extend(FileLoader.load_files_recursive(path))
            # if the path is a file, we can open it and read its content
            elif os.path.isfile(path):
                # open the file and read its content
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        content = file.read()
                except UnicodeDecodeError:
                    with open(path, "r", encoding="latin-1") as file:
                        content = file.read()

                folder = os.path.basename(os.path.dirname(base_folder))  # parent folder
                subfolder = os.path.basename(base_folder)                # current folder
                filename = entry                                         # file name
                content_list.append(FileContent(folder, subfolder, filename, content))
        return content_list
    
class Word_preprocessing:
    @staticmethod
    
    def stem_text(text):
        ps = PorterStemmer()
        words = text.split()
        words = [ps.stem(word) for word in words]
        text = ' '.join(words)
        return text
    
    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words]
        text = ' '.join(words)
        return text
    
    def apply_preprocessing(df, stem=False, lemmatize=False):
        processed_contents = []
        for content in df['content']:
            if stem:
                content = Word_preprocessing.stem_text(content)
            if lemmatize:
                content = Word_preprocessing.lemmatize_text(content)
            processed_contents.append(content)
        df['content'] = processed_contents
        return df
    
    def stop_words(df):
        stopw = set(stopwords.words("english"))
        tokenizer = RegexpTokenizer(r'\w+')

        def clean(text):
            text = tokenizer.tokenize(text)
            tokens = [t.lower() for t in text]
            clean_words = [word for word in tokens if word not in stopw]
            new_text = ' '.join(clean_words)
            return new_text
        
        df = df.copy()
        df.loc[:, 'content'] = df['content'].apply(clean)
        
        return df
    
    def create_custom_stopwords(df, threshold=0.0045):
        # Count word frequencies across all texts
        tokenizer = RegexpTokenizer(r'\w+')

        all_words = []
        for text in df["content"]:
            words = word_tokenize(text.lower())
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        total_words = len(all_words)
        
        # Words appearing more than threshold become stopwords
        custom_stops = {word for word, freq in word_freq.items() 
                    if freq / total_words > threshold}
        
        def clean(text):
            text = tokenizer.tokenize(text)
            tokens = [t.lower() for t in text]
            clean_words = [word for word in tokens if word not in custom_stops]
            new_text = ' '.join(clean_words)
            return new_text

        df = df.copy()
        df.loc[:, 'content'] = df['content'].apply(clean)

        return df
    
class Split_data:
    @staticmethod
    def split_data(file_list=negative_polarity, train=(1,2,3,4), test=(5,)):
        
        files = FileLoader.load_files_recursive(file_list)
        
        df = pd.DataFrame([{
            "folder": file.folder,
            "subfolder": file.subfolder,
            "filename": file.filename,
            "content": file.content
         } for file in files])
        
        # df = Word_preprocessing.apply_preprocessing(df, stem=True, lemmatize=True)

        df["label"] = (df["folder"].str.contains("deceptive", case=False)).astype(int)
        # 1 = deceptive, 0 = truthful
        
        train_data = {f"fold{i}" for i in train}
        test_data = {f"fold{i}" for i in test}

        train_df = df[df['subfolder'].isin(train_data).reset_index(drop=True)]
        test_df = df[df['subfolder'].isin(test_data).reset_index(drop=True)]

        train_df = Word_preprocessing.stop_words(train_df)
        test_df = Word_preprocessing.stop_words(test_df)

        train_df = Word_preprocessing.create_custom_stopwords(train_df)
        test_df = Word_preprocessing.create_custom_stopwords(test_df)

        return train_df, test_df
    

if __name__ == "__main__":
        # calls the function with the negative_polarity path and stores the result in files
        # files = FileLoader.load_files_recursive(negative_polarity) 

        # print(f"Total folders loaded: {len(set(file.folder for file in files))}")
        # print(f"Subfolders in folder: {len(set(file.subfolder for file in files))}")
        # print(f"Total subfolders loaded: {len(set(file.folder for file in files)) * len(set(file.subfolder for file in files))}")
        # print(f"Total files loaded: {len(files)}")
        # print(files[0])

    train_df, test_df = Split_data.split_data(train=(1,2,3,4), test=(5,))

    print("Train:", train_df["subfolder"].value_counts().to_dict())
    print("Test :", test_df["subfolder"].value_counts().to_dict())
    print("Voorbeeld train-rij:\n", train_df.head(1))
