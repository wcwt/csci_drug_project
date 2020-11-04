import function as f
import numpy as np
test_path = "../source_file/csci_data/SR-ARE-test/"

a,b,c = f.dataloader(test_path)
a,b,c,d = f.seperate_sample(a,c)
print(b)

a = [[1,2,3],[0,3,2]]
print(np.argmax(a))
