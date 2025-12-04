from   datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot
from   processing import *
import save
from   spectrum import *
import tools

def make_test_files(nb_files):
    folder = Path().absolute().parent.joinpath("Data/Test")

    for f in folder.glob("*"):
        if f.is_file():
            f.unlink()

    rng = np.random.default_rng()
    data = rng.integers(0,100,(20,3,nb_files))
    dataframes = [pd.DataFrame(data[:,:,i]) for i in range(nb_files)]
    tmz_mz = np.arange(100)
    frames_rt = np.arange(100)
    paths_to_feather = [folder.joinpath("{}.feather".format(i)) for i in range(nb_files)]
    paths_to_npz = [folder.joinpath("{}.npz".format(i)) for i in range(nb_files)]

    for df,path_feather,path_npz in zip(dataframes,paths_to_feather,paths_to_npz):
        df.to_feather(path_feather,compression="zstd",compression_level=10)
        np.savez(path_npz,tmz_mz=tmz_mz,frames_rt=frames_rt,date_time=dt.now())



def main():

    params_algo = {
        "skip_digest":True, # load ds
    }

    if params_algo["skip_digest"] is False: make_test_files(5)


    params_data = {
        "analyser":"tof",
        "folder":"Data/Test",
        "format":"feather",
        "name_specification":"order",
        "name_tweak":False,
        "pattern":"",
    }

    digest = [
        (save,"spectrum",{}),
        (tools,"window",{"axis":0,"min":10,"max":30,"in_coord":None}),
        (tools,"window",{"axis":1,"min":10,"max":70,"in_coord":None}),
        (tools,"interpolation",{"axis":0,"out_coord":np.linspace(15,40,100),"in_coord":None}),
        (save,"ds",{}),
    ]
    intermission = [
        (load,"ds",{})
    ]
    visualize = [
        (plot,"rt_EDTA",{})
    ]
    align = []

    # TODO: check pipeline consistency


    los = List_of_Spectrum(params_data)
    los.sort()

    pipeline_parallel = intermission if params_algo["skip_digest"] else digest
    for s in los: # TODO: parallel
        for obj,attr,args in pipeline_parallel:
            getattr(obj,attr)(s,**args)

        plt.figure()
        plt.imshow(s.ds.todense())
    
    for obj,attr,args in visualize:
        getattr(obj,attr)(los,**args)

    plt.show()


if __name__ == "__main__":
    main()
