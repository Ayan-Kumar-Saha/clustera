import os

def is_valid_path(path):

    if os.path.exists(path):
        return True

    return False


def makedir(path):
    
    os.mkdir(path)


def make_data_home(path):
    
    if not is_valid_path(path):
        makedir(path)


def make_file_path(dir, filename):

    return os.path.join(dir, filename)

