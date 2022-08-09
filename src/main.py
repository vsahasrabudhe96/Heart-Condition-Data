from data_clean import DataSet

if __name__ == "__main__":
    data = DataSet()
    data.load_csv()
    data.print_df()