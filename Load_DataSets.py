# Load FER2013 dataset
data = pd.read_csv("/kaggle/input/fer2013/fer2013.csv")
#check data shape
data.shape

#preview first 5 row of data
data.head(5)

#check usage values
#80% training, 10% validation and 10% test
data.Usage.value_counts()
