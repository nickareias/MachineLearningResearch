from sklearn import preprocessing
import numpy as np

def transpose(data):
    return [[row[i] for row in data] for i in range(len(data[0]))] 

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
            if(char != ',' and char != '\n' and char != ' '):
                temp_element += char
            elif(char != ' '):
                try:    #try to convert to float
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

    #normalize data
    for (i,x) in enumerate(columns):
        temp_min = min(x)
        temp_max = max(x)
        
        #if temp_min == temp_max it will cause div by 0
        if(temp_min !=  temp_max):
            for (j,y) in enumerate(x):
                columns[i][j] = ((y - temp_min) / (temp_max - temp_min))
            
    #transpose data back to original form
    return transpose(columns)

#takes categorical attributes and converts them to integer values
def discretize_data(data):
    
    #transpose data to easilly access columns
    columns = transpose(data) 
    
    for (a,column) in enumerate(columns):    
        names = []
        #loop through last column (string value classes)
        for x in column:
            if(names.count(x) == 0 and x != '?' and x != np.NaN):
                names.append(x)
                
        for (i,x) in enumerate(column):
            if(columns[a][i] == '?'):
                columns[a][i] = np.NaN
            else:   
                columns[a][i] = names.index(x)
        
    #transpose back to original form
    return transpose(columns)

#searches data and converts '?' to np.NaN
def question_to_nan(data):
    
    #transpose data to easilly access columns
    columns = transpose(data) 
    
    for (a,column) in enumerate(columns):        
        for (i,x) in enumerate(column):
            if(columns[a][i] == '?'):
                columns[a][i] = np.NaN

    #transpose back to original form
    return transpose(columns)


#splits data into 2 arrays, continuous variables and categorical variables
def split_data(data):
    
    #transpose data to easilly access columns
    columns = transpose(data) 
    
    continuous = []
    categorical = []
    
    for column in columns:
        if(isinstance(column[0], str)):    
            categorical.append(column)
        else:
            continuous.append(column)
    
    return (transpose(continuous), transpose(categorical))
    
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
    
def find_missing_data(data):
    
    num_missing = 0
    rows_missing = 0
    for d in data:
        temp_missing = 0
        for e in d:
            if(e == '?' or e == np.NaN):
                num_missing += 1
                temp_missing += 1
    
        if(temp_missing != 0): 
            rows_missing += 1
          
    return num_missing, rows_missing
    
def us_foreign(data):
    #transpose data to easilly access columns
    columns = transpose(data) 
    
    #loop through country column
    #0 for US
    #1 for foreign
    for (i,c) in enumerate(columns[-2]):
        if(c == "United-States"):
            columns[-2][i] = 0
        else:
            columns[-2][i] = 1
        
    #transpose back to original form
    return transpose(columns)

def remove_edu_num(data):
    
    columns = transpose(data)
    
    #delete 3rd column, education number
    del columns[2]
    
    return transpose(columns)

def remove_fnlwgt(data):
    
    columns = transpose(data)
    
    #delete 3rd column, education number
    del columns[1]
    
    return transpose(columns)

def pre_process(readpath, writepath):
    
    data = read_file(readpath)
    
    continuous, categorical = split_data(data)
    
    #remove numerical value of education
    continuous = remove_edu_num(continuous)
    #remove fnlwgt
    continuous = remove_fnlwgt(continuous)

    #convert to correct format, and convert all '?' to np.NaN for impution
    int_categorical = discretize_data(categorical)
    
    #convert country column to 0 or 1 for US or foriegn
    int_categorical = us_foreign(int_categorical)
    
    float_continuous = question_to_nan(continuous)

    #initialize imputer replacing all NaN with the most frequent of the column
    #using most frequent for categorical, because categorical int representatons
    #can't contain decimals from averaging
    imp = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    #fit imputer
    imp.fit(int_categorical)
    #impute categorical data
    imputed_categorical = imp.transform(int_categorical)
    
    #initialize imputer using mean to impute missing values since the data is 
    #continuous
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    #impute continuous data
    imp.fit(float_continuous)
    imputed_continuous = imp.transform(float_continuous)


    #encode all but the last column which is class
    #class is left as integer to be interpreted by learning script
    categorical_enc = transpose(transpose(imputed_categorical)[:-1])
    classifier = transpose(imputed_categorical)[-1]
    classifier = np.reshape(classifier, (len(classifier),1))
    enc = preprocessing.OneHotEncoder()
    enc.fit(categorical_enc)
    encoded_categorical = enc.transform(categorical_enc).toarray()
    
    normal_continuous = normalize_data(imputed_continuous)
    
    #final data is in the form [continuous, one-hot categorical, classifier]
    #classifier is an int value that will be turned into a vector during learning
    
    final_data = transpose(transpose(normal_continuous) + transpose(encoded_categorical) + transpose(classifier))
    #final_data = transpose(transpose(normal_continuous) + transpose(imputed_categorical)[:-1] + transpose(classifier))

    write_file(writepath, final_data)
 
#process both training and testing files and save them as separate files
pre_process("adult.txt","adult_onehot_2_countries_no_edu_fnlwgt.txt")



"""
data = read_file("adult.txt")

columns = transpose(data)
   
names = {}
#loop through last column (string value classes)
for x in columns[-2]:
    if(names.get(x) == None):
        names.update({x:1})
    else:
        names[x] += 1

data2 = transpose(columns)
"""