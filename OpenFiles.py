import os

# negative polarity
# the r in front of the string indicates a raw string literal, which treats backslashes as literal characters
negative_polarity = r"C:\Users\....\negative_polarity" # fill in your own path

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
if __name__ == "__main__":
        # calls the function with the negative_polarity path and stores the result in files
        files = load_files_recursive(negative_polarity) 
        # print(f"Total folders loaded: {len(set(file.folder for file in files))}")
        # print(f"Subfolders in folder: {len(set(file.subfolder for file in files))}")
        # print(f"Total subfolders loaded: {len(set(file.folder for file in files)) * len(set(file.subfolder for file in files))}")
        # print(f"Total files loaded: {len(files)}")
        print(files[0])
