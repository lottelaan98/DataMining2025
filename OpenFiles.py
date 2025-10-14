import os
import pandas as pd
import nltk

# negative polarity
# the r in front of the string indicates a raw string literal, which treats backslashes as literal characters
negative_polarity = r"C:\Users\lotte\Documents\Artificial Intelligence\Data mining\Assignment\op_spam_v1.4\op_spam_v1.4\negative_polarity" # fill in your own path

# listdir function: list all files and folders in a directory
# isfile function: check if a path is a file   
# join function: join one or more path components

# create a class to store the content of each file
# each file will be represented as an object of this class
# self. creates an instance variable that has the properties of the FileContent class 
# in this case, each instance/file will have the properties: folder, subfolder, filename and content
# if you use FileContent.folder, you access the folder property of that specific instance/file
# if you have multiple instances/files, then by using FileContent.folder[0], you get the folder of the first instance/file
class FileContent:
        def __init__(self, folder, subfolder, filename, content): # list all the properties of the class
                self.folder = folder              # main folder
                self.subfolder = subfolder        # subfolder   
                self.filename = filename          # name of the file
                self.content = content            # content of the file
                # each argument is stored as an instance variable
                # each object will remember its own folder, subfolder, filename and content

# __repr__ function: tells how to object is printed when you use print()
# it returns a string representation of the object
# So if you use FileContent.__repr__, it will return a string that shows the folder, subfolder, filename and content of that specific instance/file.
# That is the same as using print(FileContent)
        def __repr__(self):
                return f"FileContent(folder={self.folder}, subfolder={self.subfolder}, filename={self.filename}, content={self.content})"

# Different kind of methods for classes:
# 1. instance method (): self is the first argument, you call it on an instance (object) of the class
#       Always has self as the first argument
#       self = the instance (object) calling the method
#       You call it like this: instance.method()

#       For example: 
#           class Example:
#               def hello(self, name):
#                   return f"Hello {name}, I am {self}!"

#       You call it like this: print(Example().hello("Alice"))

#       Use when the method needs to access or modify the object's own data

# 2. class method (@classmethod): cls is the first argument, you call it on the class itself
#      Always has cls as the first argument

#      For example:
#          class Example:
#              @classmethod
#              count = 0
#              def increment_count(cls):
#                  cls.count += 1
#                  return cls.count

#      You call it like this: print(Example.increment_count())

#      Use when the method needs to access or modify class-level data (shared across all instances), not an individual instance

# 3. static method (@staticmethod): no self, no cls because it's a static method
#       No self, no cls
#       Works like a normal function, but it's inside the class for organization

#       For example:
#           class Example:
#               @staticmethod
#               def add(a, b):
#                   return a + b

#      You call it like this: Example.add(3, 5)

#      Use when the method doesn't need to access or modify instance or class data


class FileLoader:
    # staticmethod: No self, no cls because it's a static method
    # Works like a normal function, but it's inside the class for organization
    # You call it like this: FildeLoader.load_files_recursive(base_folder)
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
                # encoding="utf-8" ensures that special characters are read correctly. utf-8 is the most widely used 
                # encoding today. It can represent any character in the Unicode standard (text characters from any language).
                with open(path, "r", encoding="utf-8") as file:
                    content = file.read() # reads its entire content into a string 
                    folder = os.path.basename(os.path.dirname(base_folder)) # parent folder name
                    subfolder = os.path.basename(base_folder) # current folder name
                    filename = entry # file name itself
                    content_list.append(FileContent(folder, subfolder, filename, content))
        return content_list
    
# class Word_preprocessing:
#     @staticmethod
    
#     def stem_text(text):
#         from nltk.stem import PorterStemmer
#         ps = PorterStemmer()
#         words = text.split()
#         words = [ps.stem(word) for word in words]
#         text = ' '.join(words)
#         return text
    
#     def lemmatize_text(text):
#         from nltk.stem import WordNetLemmatizer
#         lemmatizer = WordNetLemmatizer()
#         words = text.split()
#         words = [lemmatizer.lemmatize(word) for word in words]
#         text = ' '.join(words)
#         return text
    
#     def apply_preprocessing(df, stopwords=None, stem=False, lemmatize=False):
#         processed_contents = []
#         for content in df['content']:
#             if stem:
#                 content = Word_preprocessing.stem_text(content)
#             if lemmatize:
#                 content = Word_preprocessing.lemmatize_text(content)
#             processed_contents.append(content)
#         df['content'] = processed_contents
#         return df
    
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
        
        df["label"] = (df["folder"].str.contains("deceptive", case=False)).astype(int)
        # 1 = deceptive, 0 = truthful
        
        train_data = {f"fold{i}" for i in train}
        test_data = {f"fold{i}" for i in test}

        train_df = df[df['subfolder'].isin(train_data).reset_index(drop=True)]
        test_df = df[df['subfolder'].isin(test_data).reset_index(drop=True)]

        return train_df, test_df
    
# This is how it was written if it was a @classmethod
# class FileLoader:
#     content_list = []

#     @classmethod
#     def load_files_recursive(cls, base_folder):
#         for entry in os.listdir(base_folder):
#             path = os.path.join(base_folder, entry)
#             if os.path.isdir(path):
#                 cls.load_files_recursive(path)   # recursion on the class
#             elif os.path.isfile(path):
#                 with open(path, "r", encoding="utf-8") as file:
#                     content = file.read()
#                     folder = os.path.basename(os.path.dirname(base_folder))
#                     subfolder = os.path.basename(base_folder)
#                     filename = entry
#                     cls.content_list.append(FileContent(folder, subfolder, filename, content))
#         return cls.content_list

# this block runs only if the file is executed directly, not if it's imported
# when you use __name_ == "_main_", you can include code that should only run when the script is executed directly
# so when I import OpenFiles in another script, the code inside this block won't run automatically
# So it won't print(files[0]) when I import OpenFiles in another script
# It will only run when I execute OpenFiles.py directly
# But by importing OenFiles, I can still use the load_files_recursive function in another script
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
