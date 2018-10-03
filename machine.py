
import json
import re

import numpy as np
import qml.fchl as qml_fchl
import qml.math as qml_math

global __ATOM_LIST__
__ATOM_LIST__ = [ x.strip() for x in ['h ','he', \
      'li','be','b ','c ','n ','o ','f ','ne', \
      'na','mg','al','si','p ','s ','cl','ar', \
      'k ','ca','sc','ti','v ','cr','mn','fe','co','ni','cu', \
      'zn','ga','ge','as','se','br','kr', \
      'rb','sr','y ','zr','nb','mo','tc','ru','rh','pd','ag', \
      'cd','in','sn','sb','te','i ','xe', \
      'cs','ba','la','ce','pr','nd','pm','sm','eu','gd','tb','dy', \
      'ho','er','tm','yb','lu','hf','ta','w ','re','os','ir','pt', \
      'au','hg','tl','pb','bi','po','at','rn', \
      'fr','ra','ac','th','pa','u ','np','pu'] ]


def get_atom(atom):
    global __ATOM_LIST__
    atom = atom.lower()
    return __ATOM_LIST__.index(atom) + 1


def get_representations(charge_list, coordinates_list, parameters):

    nmax = parameters['nmax']
    cut_distance = parameters['cut_distance']

    rep_list = []

    for atoms, coordinates in zip(charge_list, coordinates_list):
        rep = qml_fchl.generate_representation(
            coordinates,
            atoms,
            max_size=nmax,
            neighbors=nmax,
            cut_distance=cut_distance)
        rep_list.append(rep)

    rep_list = np.array(rep_list)

    return rep_list


def make_kernel(representations_x, representations_y, parameters):

    # TODO if id(representations_x) == id(representations_y)

    kernel_args = {
        "kernel": parameters['kernel'],
        "kernel_args": parameters['kernel_args'],
        "cut_distance": parameters['cut_distance'],
        "alchemy": parameters['alchemy']
    }

    # TODO if key in parameters: kernel_args[key] = parameters[key]

    kernel = qml_fchl.get_local_kernels(representations_x, representations_y, **kernel_args)
    kernel = kernel[0]

    return kernel


def get_alphas(kernel, properties):
    alpha = qml_math.cho_solve(kernel, properties)
    return alpha


def predict(predict_representations, trained_representations, alpha, parameters):
    kernel = make_kernel(predict_representations, trained_representations, parameters)
    predictions = np.dot(kernel, alpha)
    return predictions


def get_coordinates_xyz(filename):
    """
    Get coordinates from filename and return a vectorset with all the
    coordinates, in XYZ format.
    """

    f = open(filename, 'r')
    coordinates_list = list()
    atoms_list = list()
    charges_list = list()

    for line in f:

        # Read the first line to obtain the number of atoms to read
        try:
            n_atoms = int(line)
        except ValueError:
            exit("Could not obtain the number of atoms in the .xyz file.")

        # Skip the title line
        next(f)

        coordinates = []
        atoms = []
        charges = []

        for _ in range(n_atoms):
            line = next(f)

            atom = re.findall(r'[a-zA-Z]+', line)[0]
            atom = atom.upper()

            charge = get_atom(atom)

            numbers = re.findall(r'[-]?\d+\.\d*(?:[Ee][-\+]\d+)?', line)
            numbers = [float(number) for number in numbers]

            # The numbers are not valid unless we obtain exacly three
            if len(numbers) != 3:
                exit("Reading the .xyz file failed in line {0}. Please check the format.".format(lines_read + 2))

            coordinates.append(np.array(numbers))
            atoms.append(atom)
            charges.append(charge)

        coordinates_list.append(np.array(coordinates, dtype=float))
        atoms_list.append(np.array(atoms))
        charges_list.append(np.array(charges))

    coordinates_list = np.array(coordinates_list)
    atoms_list = np.array(atoms_list)
    charges_list = np.array(charges_list)

    return charges_list, coordinates_list


def get_coordinates_sdf(filename):
    """
    """
    # TODO
    # NOTE just use rdkit?

    return


def main():

    description = """
Stand-alone predictor of conformational energies using QML

"""

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--xyz', action='store', help='Molecule file containing the conformational structures. Either xyz or sdf format.')

    # Prediction parameters
    parser.add_argument('--training', '-t', action='store', metavar='FILE.npy', help='Training set in representation form')
    parser.add_argument('--alpha', '-a', action='store', metavar='FILE.npy', help='The alpha values')
    parser.add_argument('--model', '-m', action='store', metavar='FILE.json', help='The model, e.i. the settings, kernel, hyperparameters')

    # Train parameters


    args = parser.parse_args()


    # check format
    fmt = args.xyz.split('.')[-1].lower()

    if fmt == "xyz":
        charges_list, coordinates_list = get_coordinates_xyz(args.xyz)

    elif fmt == "sdf":
        quit("error: missing sdf feature")

    else:
        print("error: Don't recognize the extension. Either XYZ or SDF format.")
        quit()


    # Load model
    with open(args.model, 'r') as f:
        MODEL = json.load(f)


    # Check input size
    NMAX = MODEL['nmax']
    this_n = max([len(atoms) for atoms in charges_list])

    if this_n > NMAX:
        print("error: The model is trained for {:}, but input has {:} atoms".format(NMAX, this_n))
        quit()

    # load training and alphas
    alpha = np.load(args.alpha)
    training = np.load(args.training)

    # generate predict representations
    representations = get_representations(charges_list, coordinates_list, MODEL)
    predictions = predict(representations, training, alpha, MODEL)

    for prediction in predictions:
        print(prediction)


    return


if __name__ == "__main__":
    main()
