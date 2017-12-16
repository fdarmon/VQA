from keras.models import Model, load_model
import numpy as np
from dataset import Dataset
from norm_layer import Norm_layer
import h5py
import argparse
from data_decode import show_results

def main(args):
    if args.load is None:
        assert(False)
    else:
        model=load_model(args.load)

    f=h5py.File(args.input)
    dataset=Dataset(f,args.bs)
    res=model.predict_generator(dataset,use_multiprocessing=True,verbose=2)
    f.close()

    precision_all,precision_yes,precision_number,precision_other=show_results(np.argmax(res,axis=1)+1)
    print("Precision all : {} \n Precision yes : {} \n Precision numbers : {} \
     \n Precision other : {} ".format(precision_all,precision_yes,precision_number,precision_other))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description = "Training script")
    parser.add_argument("--input",default="./dataset_test.h5",help="Path to training data")
    parser.add_argument("--load", default=None, help='File from which to load model')
    parser.add_argument("--bs",default=500,help='Batch size',type=int)

    args=parser.parse_args()
    main(args)
