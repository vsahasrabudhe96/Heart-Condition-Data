from data_clean import DataSet

if __name__ == "__main__":
    data = DataSet()
    data.load_csv()
    int_col,str_col = data.continuous_categorical()
    data.encoding()
    X,y = data.data_target()

    
    