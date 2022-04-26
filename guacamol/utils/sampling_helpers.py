from typing import List, Set, Iterable, Union

from guacamol.utils.chemistry import is_valid, canonicalize, canonicalize_list, pass_through_filters


def sample_valid_molecules(model, number_molecules: int, prior_gen=[],
                           use_filters=False, max_tries=10) -> List[str]:
    """
    Sample from the given generator until the desired number of valid molecules
    has been sampled (i.e., ignore invalid molecules).

    Args:
        model: model to sample from
        number_molecules: number of valid molecules to generate
        max_tries: determines the maximum number N of samples to draw, N = number_molecules * max_tries

    Returns:
        A list of number_molecules valid molecules. If this was not possible with the given max_tries, the list may be shorter.
    """

    max_samples = max_tries * number_molecules
    number_already_sampled = len(prior_gen)

    valid_molecules = prior_gen

    while len(valid_molecules) < number_molecules and number_already_sampled < max_samples:
        remaining_to_sample = number_molecules - len(valid_molecules)

        samples = model.generate(remaining_to_sample)
        number_already_sampled += remaining_to_sample

        valid_molecules += [m for m in samples if is_valid(m)]
        if use_filters:
            valid_molecules = pass_through_filters(valid_molecules)

    return valid_molecules


def sample_unique_molecules(model, number_molecules: int, prior_gen=[],
                            use_filters=False, max_tries=10) -> List[str]:
    """
    Sample from the given generator until the desired number of unique (distinct) molecules
    has been sampled (i.e., ignore duplicate molecules).

    Args:
        model: model to sample from
        number_molecules: number of unique (distinct) molecules to generate
        max_tries: determines the maximum number N of samples to draw, N = number_molecules * max_tries

    Returns:
        A list of number_molecules unique molecules, in canonalized form.
        If this was not possible with the given max_tries, the list may be shorter.
        The generation order is kept.
    """

    max_samples = max_tries * number_molecules
    number_already_sampled = 0

    unique_list = list(set(prior_gen))
    unique_list = [m for m in unique_list if is_valid(m)]
    if use_filters:
        unique_list = pass_through_filters(unique_list)
    unique_set = set(unique_list)

    while len(unique_list) < number_molecules and number_already_sampled < max_samples:
        remaining_to_sample = number_molecules - len(unique_list)

        samples = model.generate(remaining_to_sample)
        number_already_sampled += remaining_to_sample

        for smiles in samples:
            if is_valid(smiles):
                canonical_smiles = canonicalize(smiles)
                if canonical_smiles is not None and canonical_smiles not in unique_set:
                    unique_list.append(canonical_smiles)
                    unique_set.add(canonical_smiles)

        if use_filters:
            unique_list = pass_through_filters(unique_list)
        unique_set = set(unique_list)

    return unique_list

def sample_novel_molecules(model, number_molecules: int, train_mols: Union[str, List[str]],
                           prior_gen=[], use_filters=False, max_tries=10) -> List[str]:
    """
    Sample from the given generator until the desired number of novel (distinct from
    training set) molecules have been sampled

    Args:
        model: model to sample from
        number_molecules: number of novel molecules to generate
        train_molecules: molecules used to train the generator
        max_tries: determines the maximum number N of samples to draw, N = number_molecules * max_tries

    Returns:
        A list of number_molecules novel molecules, in canonilized form.
        If this was not possible with the given max_tries, the list may be shorter.
        The generation order is kept
    """
    if isinstance(train_mols, str):
        train_molecules = [s.strip() for s in open(train_mols).readlines()]
        train_molecules = set(canonicalize_list(train_molecules, include_stereocenters=False))
    else:
        train_molecules = set(train_mols)

    max_samples = max_tries * number_molecules
    number_already_sampled = 0

    unique_list = list(set(prior_gen))
    unique_list = [m for m in unique_list if is_valid(m)]
    if use_filters:
        unique_list = pass_through_filters(unique_list)
    unique_set = set(unique_list)

    novel_set = unique_set.difference(train_molecules)
    novel_list = list(novel_set)

    while len(unique_list) < number_molecules and number_already_sampled < max_samples:
        remaining_to_sample = number_molecules - len(novel_list)

        samples = model.generate(remaining_to_sample)
        number_already_sampled += remaining_to_sample

        for smile in samples:
            if is_valid(smile):
                canonical_smiles = canonicalize(smile)
                if canonical_smiles is not None and canonical_smiles not in unique_set:
                    unique_list.append(canonical_smiles)

        if use_filters:
            unique_list = pass_through_filters(unique_list)
        unique_set = set(unique_list)

        novel_set = unique_set.difference(train_molecules)
        novel_list = list(novel_set)

    return novel_list
