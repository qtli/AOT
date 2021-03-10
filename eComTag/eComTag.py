import pickle
from Lang import Lang


if __name__ == '__main__':
    print("LOADING eComTag DATASET ...")
    with open('ecomtag_dataset_preproc_w_domain.p', "rb") as f:
        [data_tra, data_val, data_tst, vocab] = pickle.load(f)

    print("train length: ", len(data_tra['reviews']))
    print("validation length: ", len(data_val['reviews']))
    print("test length: ", len(data_tst['reviews']))

    for i in range(20,22):
        print('[domain]:', data_tra['domain'][i])
        print('[reviews]:', [' '.join(u) for u in data_tra['reviews'][i]])
        print('[labels]:', data_tra['labels'][i])
        print('[tags]:', ' '.join(data_tra['tags'][i]))
        # print('[tag_positions]:', data_tra['tag_aln'][i])
        print(" ")


