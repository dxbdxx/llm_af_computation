import pickle

from tqdm import tqdm

from src import config

if __name__ == '__main__':
    data_set_dir = config.dataset_dir
    af_count = 0
    extensions_count = {"com": 0, "grd": 0, "prf": 0, "stb": 0}
    arguments_count = {"com": 0, "grd": 0, "prf": 0, "stb": 0}
    scep = {"com": 0, "grd": 0, "prf": 0, "stb": 0}  #all
    cred = {"com": 0, "grd": 0, "prf": 0, "stb": 0}

    # test 25
    for i in range(6, 26):
        data_dir = data_set_dir / f"test-{i}" / "problems" / "enumeration"
        print(data_dir)
        af_pkl_paths = list(data_dir.glob("*.pkl"))[0:100]
        print(af_pkl_paths)
        for af_pkl_path in tqdm(af_pkl_paths, desc=f"test-{i}_af_pkl"):
            with open(af_pkl_path, 'rb') as file:
                loaded_data = pickle.load(file)
                print(loaded_data)
                af = loaded_data['af']
                af_count += 1
                extensions_count["grd"] += len(af["extensions"]["grd"])
                extensions_count["com"] += len(af["extensions"]["com"])
                extensions_count["prf"] += len(af["extensions"]["prf"])
                extensions_count["stb"] += len(af["extensions"]["stb"])

                arguments_count["grd"] += sum(len(e) for e in af["extensions"]["grd"])
                arguments_count["com"] += sum(len(e) for e in af["extensions"]["com"])
                arguments_count["prf"] += sum(len(e) for e in af["extensions"]["prf"])
                arguments_count["stb"] += sum(len(e) for e in af["extensions"]["stb"])

                scep["grd"] += sum([all(arg in e for e in af["extensions"]["grd"]) for arg in range(0, i)])
                scep["com"] += sum([all(arg in e for e in af["extensions"]["com"]) for arg in range(0, i)])
                scep["prf"] += sum([all(arg in e for e in af["extensions"]["prf"]) for arg in range(0, i)])
                scep["stb"] += sum([all(arg in e for e in af["extensions"]["stb"]) for arg in range(0, i)])

                cred["grd"] += sum([any(arg in e for e in af["extensions"]["grd"]) for arg in range(0, i)])
                cred["com"] += sum([any(arg in e for e in af["extensions"]["com"]) for arg in range(0, i)])
                cred["prf"] += sum([any(arg in e for e in af["extensions"]["prf"]) for arg in range(0, i)])
                cred["stb"] += sum([any(arg in e for e in af["extensions"]["stb"]) for arg in range(0, i)])

    print(f"af_count: {af_count}")
    extensions_per_AF = {key: value / af_count for key, value in extensions_count.items()}
    print(f"extensions_per_AF: {extensions_per_AF}")
    Arguments_per_extension = {key: arguments_count[key] / value for key, value in extensions_count.items()}
    print(f"Arguments_per_extension: {Arguments_per_extension}")
    Scep_per_AF = {key: value / af_count for key, value in scep.items()}
    print(f"Scep_per_AF: {Scep_per_AF}")
    Cred_per_AF = {key: value / af_count for key, value in cred.items()}
    print(f"Cred_per_AF: {Cred_per_AF}")
