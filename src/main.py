from data_clean import DataSet

if __name__ == "__main__":
    data = DataSet()
    data.load_csv()
    
    print("----Before Encoding the categorical values-----")
    data.print_df()
    
    int_col,str_col = data.continuous_categorical()
    
    data.encoding()
    
    print("-----After encoding the categorical values-----")
    
    data.print_df()
    
    