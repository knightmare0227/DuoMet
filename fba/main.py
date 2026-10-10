from validate import *
import logging

logging.disable()


def main(gem_name):
    #
    # parallel = Parallel(n_jobs=-1)
    # parallel(delayed(validate)((gem_name, k)) for k in ['50', '100', '150', '200', '250', '300'])

    for k in ['200']:
        validate((gem_name, k))


if __name__ == "__main__":
    root_path = '../data/gems/'
    data_list = sorted(os.listdir(root_path))
    for gem_name in data_list:
        main(gem_name[:-4])

