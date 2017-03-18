import numpy as np

#reads data and converts numerical data to floats
def read_file(path):
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    
    #get data from lines
    data = []
    for line in lines:
        temp_row = []
        temp_element = ""
        for char in line:
            if(char != ',' and char != '\n'):
                temp_element += char
            else:
                try:
                    temp_row.append(float(temp_element))
                except:
                    temp_row.append(temp_element)
                temp_element = ""
                
        data.append(temp_row)
        
    return data

#normalizes data, excluding last column which is a string value of the class
def normalize_data(data):
    #transpose data to easilly access columns
    columns = transpose(data)

    #count max sig figs in each column, for rounding purposes
    sig_figs = []
    for x in columns[:-1]:
        sig_figs.append(len(str(d))-1 for d in x)
    
    #normalize data
    for (i,x) in enumerate(columns[:-1]):
        temp_min = min(x)
        temp_max = max(x)
        for (j,y) in enumerate(x):
            columns[i][j] = ((y - temp_min) / (temp_max - temp_min))
            
    #transpose data back to original form
    return transpose(columns)

#takes string value classifiers and converts them to integer values
def discretize_classes(data):
    
    #transpose data to easilly access columns
    columns = transpose(data) 
    
    names = []
    #loop through last column (string value classes)
    for x in columns[-1]:
        if(names.count(x) == 0):
            names.append(x)
            
    for (i,x) in enumerate(columns[-1]):
        columns[-1][i] = names.index(x)
    
    #transpose back to original form
    return transpose(columns)
    
def transpose(data):
    return [[row[i] for row in data] for i in range(len(data[0]))] 

#takes numerical data, converts it to string values and creates a
#csv .txt file from it.  Then writes it to the specified path
def write_file(path, data):
    
    string_data = np.array(data, str)
    lines = []
    
    for i in string_data:
        temp_line = ""
        for j in i:
            temp_line += j + ","
        lines.append(temp_line[:-1] + "\n") #appends the line except for the last comma
        
    f = open(path, "w")
    f.writelines(lines)

def preprocess_data():
    readpath = "data/iris_data.txt"
    writepath = "data/iris_data_preprocessed.txt"
    
    data = read_file(readpath)
    normalized_data = normalize_data(data)
    discretized_normalized_data = discretize_classes(normalized_data)
    
    write_file(writepath, discretized_normalized_data)
    
preprocess_data()