import os
import numpy as np
import scipy.io as io
import argparse
import hues

parser = argparse.ArgumentParser()
parser.add_argument("--path",default='./',type=str)
opt, _ = parser.parse_known_args()

if __name__ == "__main__":

    for root, dirs, files in os.walk(opt.path, topdown=False):
        for name in files:
            if os.path.splitext(name)[-1] == '.npy':
                hues.info(os.path.join(root,name))
                temp = np.load(os.path.join(root,name))
                if temp.ndim == 3:
                    temp = temp.transpose(1,2,0)
                save_name = os.path.splitext(name)[0] + '.mat'
                io.savemat(os.path.join(root,save_name),{'img':temp})
