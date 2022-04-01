import logging
import time
from abc import abstractmethod
from typing import Dict, Any, Iterable, List
import numpy as np

from guacamol.utils.math import cos_similarity
from guacamol.utils.chemistry import canonicalize_list, is_valid, calculate_pc_descriptors, continuous_kldiv, \
    discrete_kldiv, calculate_internal_pairwise_similarities, tokenizer, fragment_list, scaffold_list, \
    get_fingerprints_from_smileslist, average_agg_tanimoto, pass_through_filters
from guacamol.utils.data import get_random_subset
from guacamol.utils.sampling_helpers import sample_valid_molecules, sample_unique_molecules

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DistributionLearningBenchmarkResult:
    """
    Contains the results of a distribution learning benchmark.

    NB: timing does not make sense since training happens outside of DistributionLearningBenchmark.
    """

    def __init__(self, benchmark_name: str, score: float, sampling_time: float, metadata: Dict[str, Any]) -> None:
        """
        Args:
            benchmark_name: name of the distribution-learning benchmark
            score: benchmark score
            sampling_time: time for sampling the molecules in seconds
            metadata: benchmark-specific information
        """
        self.benchmark_name = benchmark_name
        self.score = score
        self.sampling_time = sampling_time
        self.metadata = metadata


class DistributionLearningBenchmark:
    """
    Base class for assessing how well a model is able to generate molecules matching a molecule distribution.

    Derived class should implement the assess_molecules function
    """

    def __init__(self, name: str, number_samples: int) -> None:
        self.name = name
        self.number_samples = number_samples

    @abstractmethod
    def assess_model(self, model) -> DistributionLearningBenchmarkResult:
        """
        Assess a distribution-matching generator model.

        Args:
            model: model to assess
        """


class ValidityBenchmark(DistributionLearningBenchmark):
    """
    Assesses what percentage of molecules generated by a model are valid molecules (i.e. have a valid SMILES string)
    """

    def __init__(self, number_samples) -> None:
        super().__init__(name='Validity', number_samples=number_samples)

    def assess_model(self, model, return_valid=True, use_filters=False):
        start_time = time.time()
        molecules = model.generate(number_samples=self.number_samples)
        end_time = time.time()

        if len(molecules) != self.number_samples:
            raise Exception('The model did not generate the correct number of molecules')

        valid = []
        for smi in molecules:
            if is_valid(smi):
                valid.append(smi)
        if use_filters:
            valid = pass_through_filters(valid)
            self.name = self.name + '_filters'
        number_valid = len(valid)
        validity_ratio = number_valid / self.number_samples
        metadata = {
            'number_samples': self.number_samples,
            'number_valid': number_valid,
            'use_filters': use_filters
        }

        result = DistributionLearningBenchmarkResult(benchmark_name=self.name,
                                                     score=validity_ratio,
                                                     sampling_time=end_time - start_time,
                                                     metadata=metadata)
        if return_valid:
            return result, valid
        else:
            return result


class UniquenessBenchmark(DistributionLearningBenchmark):
    """
    Assesses what percentage of molecules generated by a model are unique.
    """

    def __init__(self, number_samples) -> None:
        super().__init__(name='Uniqueness', number_samples=number_samples)

    def assess_model(self, model) -> DistributionLearningBenchmarkResult:
        start_time = time.time()
        molecules = sample_valid_molecules(model=model, number_molecules=self.number_samples)
        end_time = time.time()

        if len(molecules) != self.number_samples:
            logger.warning('The model could not generate enough valid molecules. The score will be penalized.')

        # canonicalize_list removes duplicates (and invalid molecules, but there shouldn't be any)
        unique_molecules = canonicalize_list(molecules, include_stereocenters=False)

        unique_ratio = len(unique_molecules) / self.number_samples
        metadata = {
            'number_samples': self.number_samples,
            'number_unique': len(unique_molecules)
        }

        return DistributionLearningBenchmarkResult(benchmark_name=self.name,
                                                   score=unique_ratio,
                                                   sampling_time=end_time - start_time,
                                                   metadata=metadata)


class NoveltyBenchmark(DistributionLearningBenchmark):
    def __init__(self, number_samples: int, training_set: Iterable[str]) -> None:
        """
        Args:
            number_samples: number of samples to generate from the model
            training_set: molecules from the training set
        """
        super().__init__(name='Novelty', number_samples=number_samples)
        self.training_set_molecules = set(canonicalize_list(training_set, include_stereocenters=False))

    def assess_model(self, model) -> DistributionLearningBenchmarkResult:
        """
        Assess a distribution-matching generator model.

        Args:
            model: model to assess
        """
        start_time = time.time()
        molecules = sample_unique_molecules(model=model, number_molecules=self.number_samples, max_tries=2)
        end_time = time.time()

        if len(molecules) != self.number_samples:
            logger.warning('The model could not generate enough unique molecules. The score will be penalized.')

        # canonicalize_list in order to remove stereo information (also removes duplicates and invalid molecules, but there shouldn't be any)
        unique_molecules = set(canonicalize_list(molecules, include_stereocenters=False))

        novel_molecules = unique_molecules.difference(self.training_set_molecules)

        novel_ratio = len(novel_molecules) / self.number_samples

        metadata = {
            'number_samples': self.number_samples,
            'number_novel': len(novel_molecules)
        }

        return DistributionLearningBenchmarkResult(benchmark_name=self.name,
                                                   score=novel_ratio,
                                                   sampling_time=end_time - start_time,
                                                   metadata=metadata)


class KLDivBenchmark(DistributionLearningBenchmark):
    """
    Computes the KL divergence between a number of samples and the training set for physchem descriptors
    """

    def __init__(self, number_samples: int, training_set: List[str]) -> None:
        """
        Args:
            number_samples: number of samples to generate from the model
            training_set: molecules from the training set
        """
        super().__init__(name='KL divergence', number_samples=number_samples)
        self.training_set_molecules = canonicalize_list(get_random_subset(training_set, self.number_samples, seed=42),
                                                        include_stereocenters=False)
        self.pc_descriptor_subset = [
            'BertzCT',
            'MolLogP',
            'MolWt',
            'TPSA',
            'NumHAcceptors',
            'NumHDonors',
            'NumRotatableBonds',
            'NumAliphaticRings',
            'NumAromaticRings'
        ]

    def assess_model(self, model) -> DistributionLearningBenchmarkResult:
        """
        Assess a distribution-matching generator model.

        Args:
            model: model to assess
        """
        start_time = time.time()
        molecules = sample_unique_molecules(model=model, number_molecules=self.number_samples, max_tries=2)
        end_time = time.time()

        if len(molecules) != self.number_samples:
            logger.warning('The model could not generate enough unique molecules. The score will be penalized.')

        # canonicalize_list in order to remove stereo information (also removes duplicates and invalid molecules, but there shouldn't be any)
        unique_molecules = set(canonicalize_list(molecules, include_stereocenters=False))

        # first we calculate the descriptors, which are np.arrays of size n_samples x n_descriptors
        d_sampled = calculate_pc_descriptors(unique_molecules, self.pc_descriptor_subset)
        d_chembl = calculate_pc_descriptors(self.training_set_molecules, self.pc_descriptor_subset)

        kldivs = {}

        # now we calculate the kl divergence for the float valued descriptors ...
        for i in range(4):
            kldiv = continuous_kldiv(X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i])
            kldivs[self.pc_descriptor_subset[i]] = kldiv

        # ... and for the int valued ones.
        for i in range(4, 9):
            kldiv = discrete_kldiv(X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i])
            kldivs[self.pc_descriptor_subset[i]] = kldiv

        # pairwise similarity

        chembl_sim = calculate_internal_pairwise_similarities(self.training_set_molecules)
        chembl_sim = chembl_sim.max(axis=1)

        sampled_sim = calculate_internal_pairwise_similarities(unique_molecules)
        sampled_sim = sampled_sim.max(axis=1)

        kldiv_int_int = continuous_kldiv(X_baseline=chembl_sim, X_sampled=sampled_sim)
        kldivs['internal_similarity'] = kldiv_int_int

        # for some reason, this runs into problems when both sets are identical.
        # cross_set_sim = calculate_pairwise_similarities(self.training_set_molecules, unique_molecules)
        # cross_set_sim = cross_set_sim.max(axis=1)
        #
        # kldiv_ext = discrete_kldiv(chembl_sim, cross_set_sim)
        # kldivs['external_similarity'] = kldiv_ext
        # kldiv_sum += kldiv_ext

        metadata = {
            'number_samples': self.number_samples,
            'kl_divs': kldivs
        }

        # Each KL divergence value is transformed to be in [0, 1].
        # Then their average delivers the final score.
        partial_scores = [np.exp(-score) for score in kldivs.values()]
        score = sum(partial_scores) / len(partial_scores)

        return DistributionLearningBenchmarkResult(benchmark_name=self.name,
                                                   score=score,
                                                   sampling_time=end_time - start_time,
                                                   metadata=metadata)

class ReconstructionBenchmark(DistributionLearningBenchmark):
    """
    Computes the reconstruction accuracy for a set of holdout molecules
    """
    def __init__(self, test_set: List[str], sample_size: int) -> None:
        """
        Args:
            test_set: list of smiles to reconstruct
            sample_size: number of smiles to reconstruct
        """
        super().__init__(name='Reconstruction', number_samples=sample_size)
        self.test_set_molecules = canonicalize_list(get_random_subset(test_set, self.number_samples, seed=42),
                                                    include_stereocenters=False)
        self.test_set_tokenized = [tokenizer(smi) for smi in self.test_set_molecules]

    def assess_model(self, model) -> DistributionLearningBenchmarkResult:
        """
        Assess a distribution-matching generator model

        Args:
            model: model to assess
        """
        start_time = time.time()
        reconstructed_smiles = model.reconstruct(self.test_set_molecules)
        end_time = time.time()
        reconstructed_tokenized = [tokenizer(smi) for smi in reconstructed_smiles]
        smile_accs = []
        hits = 0
        misses = 0
        position_accs = np.zeros((2, model.args.max_length+1))

        for in_smi, out_smi in zip(self.test_set_tokenized, reconstructed_tokenized):
            if in_smi == out_smi:
                smile_accs.append(1)
            else:
                smile_accs.append(0)

            misses += abs(len(in_smi) - len(out_smi))
            for j, (token_in, token_out) in enumerate(zip(in_smi, out_smi)):
                if token_in == token_out:
                    hits += 1
                    position_accs[0,j] += 1
                else:
                    misses += 1
                position_accs[1,j] += 1

        smile_acc = np.mean(smile_accs)
        token_acc = hits / (hits + misses)
        position_acc = []
        for i in range(model.args.max_length+1):
            position_acc.append(position_accs[0,i] / position_accs[1,i])
        score = (smile_acc, token_acc, position_acc)

        metadata = {
            'number_samples': self.number_samples,
            'accuracy_types': ['smiles', 'tokens', 'positional']
        }

        return DistributionLearningBenchmarkResult(benchmark_name=self.name,
                                                   score=score,
                                                   sampling_time=end_time-start_time,
                                                   metadata=metadata)

class FragBenchmark(DistributionLearningBenchmark):
    """
    Computes the cosine similarity between generated fragments and holdout set fragments
    """
    def __init__(self, test_set: List[str], sample_size: int, type: str) -> None:
        """
        Args:
            test_set: list of smiles to use for fragment comparison
            sample_size: number of smiles to sample
        """
        super().__init__(name='Frag', number_samples=sample_size)
        self.type = type
        self.ref_frags = fragment_list(test_set)

    def assess_model(self, model) -> DistributionLearningBenchmarkResult:
        """
        Assess a distribution-matching generator model

        Args:
            model: model to assess
        """
        start_time = time.time()
        molecules = sample_valid_molecules(model=model, number_molecules=self.number_samples)
        mol_frags = fragment_list(molecules)
        end_time = time.time()

        metadata = {
            'number_samples': self.number_samples,
            'number_valid': len(molecules)
        }

        score = cos_similarity(self.ref_frags, mol_frags)

        return DistributionLearningBenchmarkResult(benchmark_name=self.name+'_{}'.format(self.type),
                                                   score=score,
                                                   sampling_time=end_time-start_time,
                                                   metadata=metadata)

class ScafBenchmark(DistributionLearningBenchmark):
    """
    Computes the cosine similarity between generated scaffolds and holdout set scaffolds
    """
    def __init__(self, test_set: List[str], sample_size: int, type: str) -> None:
        """
        Args:
            test_set: list of smiles to use for scaffold comparison
            sample_size: number of smiles to sample
        """
        super().__init__(name='Scaf', number_samples=sample_size)
        self.type = type
        self.ref_scafs = scaffold_list(test_set)

    def assess_model(self, model) -> DistributionLearningBenchmarkResult:
        """
        Assess a distribution-matching generator model

        Args:
            model: model to assess
        """
        start_time = time.time()
        molecules = sample_valid_molecules(model=model, number_molecules=self.number_samples)
        mol_scafs = scaffold_list(molecules)
        end_time = time.time()

        metadata = {
            'number_samples': self.number_samples,
            'number_valid': len(molecules)
        }

        score = cos_similarity(self.ref_scafs, mol_scafs)

        return DistributionLearningBenchmarkResult(benchmark_name=self.name+'_{}'.format(self.type),
                                                   score=score,
                                                   sampling_time=end_time-start_time,
                                                   metadata=metadata)

class SNNBenchmark(DistributionLearningBenchmark):
    """
    Computes the average max similarities of generated smiles to holdout smiles
    """
    def __init__(self, test_set: List[str], sample_size: int) -> None:
        """
        Args:
            test_set: list of smiles to compare to generated set
            sample_size: number of smiles to sample
        """
        super().__init__(name='SNN', number_samples=sample_size)
        self.test_fps = np.vstack(get_fingerprints_from_smileslist(test_set))

    def assess_model(self, model) -> DistributionLearningBenchmarkResult:
        """
        Assess a distribution-matching generator model

        Args:
            model: model to assess
        """
        start_time = time.time()
        molecules = sample_valid_molecules(model=model, number_molecules=self.number_samples)
        gen_fps = np.vstack(get_fingerprints_from_smileslist(molecules))
        end_time = time.time()

        metadata = {
            'number_samples': self.number_samples,
            'number_valid': len(molecules)
        }

        score = average_agg_tanimoto(self.test_fps, gen_fps)

        return DistributionLearningBenchmarkResult(benchmark_name=self.name,
                                                   score=score,
                                                   sampling_time=end_time-start_time,
                                                   metadata=metadata)

class IntDivBenchmark(DistributionLearningBenchmark):
    """
    Computes the average max similarities of generated smiles to themselves
    """
    def __init__(self, number_samples: int, p: int) -> None:
        """
        Args:
            sample_size: number of smiles to sample
        """
        super().__init__(name='IntDiv', number_samples=number_samples)
        self.p = p

    def assess_model(self, model) -> DistributionLearningBenchmarkResult:
        """
        Assess a distribution-matching generator model

        Args:
            model: model to assess
        """
        start_time = time.time()
        molecules = sample_valid_molecules(model=model, number_molecules=self.number_samples)
        gen_fps = np.vstack(get_fingerprints_from_smileslist(molecules))
        end_time = time.time()

        metadata = {
            'number_samples': self.number_samples,
            'number_valid': len(molecules),
            'p': self.p
        }

        score = average_agg_tanimoto(gen_fps, gen_fps, p=self.p)

        return DistributionLearningBenchmarkResult(benchmark_name=self.name+'{}'.format(self.p),
                                                   score=score,
                                                   sampling_time=end_time-start_time,
                                                   metadata=metadata)
