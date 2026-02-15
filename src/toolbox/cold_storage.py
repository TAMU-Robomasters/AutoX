import json
import os


class ColdStorage(dict):
    """Summary:
        behaves like a python dictionary, but every time 
        a key is set, it will write the output to a json file
        (e.g. values are saved in cold storage)
    Note1:
        THIS WILL FAIL:
            thing = ColdStorage(path="blah")
            thing["hi"] = [1,2,3] # fine
            thing["hi"].push(4)   # NOT FINE: 4 will not be saved to cold storage
            # must do
            thing["hi"] = [1,2,3,4] # will save
    Note2:
        Assumes it is the only thing writing to the file
        (both one instance of itself, e.g. no multithreading, and no random thing editing the JSON at runtime)
    """
    def __init__(self, path):
        self.path = path
        self.prev_string_content = "{}"
        try:
            with open(self.path,'r') as file:
                self.prev_string_content = file.read()
        except:
            # if directory doesnt exist, create it
            directory = os.path.dirname(self.path)
            if directory == '': directory = '.'
            os.makedirs(directory, exist_ok=True)
            # if file doesnt exist, create it
            with open(self.path, 'w') as the_file:
                the_file.write('{}')
        # read json
        try:
            self._dict = json.loads(self.prev_string_content)
        except Exception:
            print('''\n\nWarning json file is corrupted, resetting to empty dict\n\n''')
            with open(self.path+".corrupted", 'w') as the_file:
                the_file.write(str(self.prev_string_content))
            with open(self.path, 'w') as the_file:
                the_file.write('{}')
            self.prev_string_content = '{}'
            self._dict = {}
    
    def save_if_needed(self):
        output_as_string = json.dumps(self._dict)
        if output_as_string != self.prev_string_content:
            self.prev_string_content = output_as_string
            # write to cold storage
            with open(self.path, 'w') as outfile:
                outfile.write(output_as_string)
    
    def __getitem__(self, key):
        return self._dict[key]
    
    def __setitem__(self, key, value):
        self._dict[key] = value
        self.save_if_needed()
    
    def __delitem__(self, key):
        try:
            del self._dict[key]
            self.save_if_needed()
        except Exception:
            pass
    
    def keys(self):
        return self._dict.keys()
    def values(self):
        return self._dict.values()
    def items(self):
        return self._dict.items()
    def update(self,*args,**kwargs):
        output = self._dict.update(*args,**kwargs)
        self.save_if_needed()
        return output
    def setdefault(self,*args,**kwargs):
        output = self._dict.setdefault(*args,**kwargs)
        self.save_if_needed()
        return output
    def __iter__(self):
        return self._dict.__iter__()
    def __reversed__(self):
        return self._dict.__reversed__()
    def __contains__(self):
        return self._dict.__contains__()
    def __len__(self):
        return self._dict.__len__()
    def __repr__(self):
        return self._dict.__repr__()
    def __str__(self):
        return self._dict.__str__()
    def __eq__(self, other):
        return self._dict.__eq__(other)
