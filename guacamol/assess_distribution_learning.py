import datetime
import json
import logging
from collections import OrderedDict
from typing import List, Dict, Any, Union, Set

import guacamol
from guacamol.distribution_learning_benchmark import DistributionLearningBenchmark, DistributionLearningBenchmarkResult, \
    ValidityBenchmark, UniquenessBenchmark
from guacamol.standard_benchmarks import novelty_benchmark
from guacamol.utils.sampling_helpers import sample_novel_molecules
from guacamol.utils.chemistry import canonicalize_list
from guacamol.benchmark_suites import distribution_learning_benchmark_suite
from guacamol.utils.data import get_time_string

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def assess_distribution_learning(model,
                                 train_mols: Union[str, Set[str]],
                                 test_mols: Union[str, Set[str]],
                                 test_scaffold_file=None,
                                 reconstruct=False,
                                 use_filters=False,
                                 json_output_file='output_distribution_learning.json',
                                 benchmark_version='v2',
                                 number_samples=10000) -> None:
    """
    Assesses a distribution-matching model for de novo molecule design.

    Args:
        model: Model to evaluate
        chembl_training_file: path to ChEMBL training set, necessary for some benchmarks
        json_output_file: Name of the file where to save the results in JSON format
        benchmark_version: which benchmark suite to execute
    """
    _assess_distribution_learning(model=model,
                                  train_mols=train_mols,
                                  test_mols=test_mols,
                                  test_scaffold_file=test_scaffold_file,
                                  reconstruct=reconstruct,
                                  use_filters=use_filters,
                                  json_output_file=json_output_file,
                                  benchmark_version=benchmark_version,
                                  number_samples=number_samples)


def _assess_distribution_learning(model,
                                  train_mols: Union[str, Set[str]],
                                  test_mols: Union[str, Set[str]],
                                  test_scaffold_file: str,
                                  reconstruct: bool,
                                  use_filters: bool,
                                  json_output_file: str,
                                  benchmark_version: str,
                                  number_samples: int) -> None:
    """
    Internal equivalent to assess_distribution_learning, but allows for a flexible number of samples.
    To call directly only for testing.
    """
    logger.info(f'Benchmarking distribution learning, version {benchmark_version}')
    if isinstance(train_mols, str):
        train_mols = [s.strip() for s in open(train_mols).readlines()]
        train_mols = set(canonicalize_list(train_mols))
    if isinstance(test_mols, str):
        test_mols = [s.strip() for s in open(test_mols).readlines()]
        test_mols = set(test_mols)
    if test_scaffold_file is not None:
        test_scaffold_mols = [s.strip() for s in open(test_scaffold_mols).readlines()]
        test_scaffold_mols = set(test_scaffold_mols)
    else:
        test_scaffold_mols = None
    benchmarks = distribution_learning_benchmark_suite(train_mols=train_mols,
                                                       test_mols=test_mols,
                                                       test_scaffold_mols=test_scaffold_mols,
                                                       reconstruct=reconstruct,
                                                       version_name=benchmark_version,
                                                       number_samples=number_samples)

    if benchmark_version == 'v1':
        results = _evaluate_distribution_learning_benchmarks(model=model, benchmarks=benchmarks)
    elif benchmark_version == 'v2':
        results = []
        validity = ValidityBenchmark(number_samples=number_samples,
                                     use_filters=use_filters)
        uniqueness = UniquenessBenchmark(number_samples=number_samples,
                                         use_filters=use_filters)
        novelty = novelty_benchmark(train_mols, number_samples, use_filters=use_filters,
                                    return_train_mols=True)

        print(f'Running benchmark: validity')
        result, valid = validity.assess_model(model)
        results.append(result)
        print(f'Results for the benchmark validity:')
        print(f'  Score: {result.score:.6f}')
        print(f'  Sampling time: {str(datetime.timedelta(seconds=int(result.sampling_time)))}')
        print(f'  Metadata: {result.metadata}')

        print(f'Running benchmark: uniqueness')
        result, unique = uniqueness.assess_model(model, valid)
        results.append(result)
        print(f'Results for the benchmark uniqueness:')
        print(f'  Score: {result.score:.6f}')
        print(f'  Sampling time: {str(datetime.timedelta(seconds=int(result.sampling_time)))}')
        print(f'  Metadata: {result.metadata}')

        print(f'Running benchmark: novelty')
        result, novel, train_mols = novelty.assess_model(model, unique)
        results.append(result)
        print(f'Results for the benchmark novelty:')
        print(f'  Score: {result.score:.6f}')
        print(f'  Sampling time: {str(datetime.timedelta(seconds=int(result.sampling_time)))}')
        print(f'  Metadata: {result.metadata}')
        if len(novel) < number_samples:
            print(f'Upsampling to specified number of novel molecules')
            novel = sample_novel_molecules(model, number_molecules=number_samples,
                                           train_mols=train_mols, prior_gen=novel,
                                           use_filters=use_filters)
        results += _evaluate_distribution_learning_benchmarks(model=model, benchmarks=benchmarks,
                                                              prior_gen=novel)


    benchmark_results: Dict[str, Any] = OrderedDict()
    benchmark_results['guacamol_version'] = guacamol.__version__
    benchmark_results['benchmark_suite_version'] = benchmark_version
    benchmark_results['timestamp'] = get_time_string()
    benchmark_results['samples'] = model.generate(100)
    benchmark_results['results'] = [vars(result) for result in results]

    logger.info(f'Save results to file {json_output_file}')
    with open(json_output_file, 'wt') as f:
        f.write(json.dumps(benchmark_results, indent=4))


def _evaluate_distribution_learning_benchmarks(model,
                                               benchmarks: List[DistributionLearningBenchmark],
                                               prior_gen=None) -> List[DistributionLearningBenchmarkResult]:
    """
    Evaluate a model with the given benchmarks.
    Should not be called directly except for testing purposes.

    Args:
        model: model to assess
        benchmarks: list of benchmarks to evaluate
        json_output_file: Name of the file where to save the results in JSON format
    """

    print(f'Number of benchmarks: {len(benchmarks)}')

    results = []
    for i, benchmark in enumerate(benchmarks, 1):
        print(f'Running benchmark {i}/{len(benchmarks)}: {benchmark.name}')
        result = benchmark.assess_model(model, prior_gen)
        print(f'Results for the benchmark "{result.benchmark_name}":')
        if result.benchmark_name == 'Reconstruction':
            print(f'  Score: {result.score[0]:.6f}')
        else:
            print(f'  Score: {result.score:6f}')
        print(f'  Sampling time: {str(datetime.timedelta(seconds=int(result.sampling_time)))}')
        print(f'  Metadata: {result.metadata}')
        results.append(result)

    logger.info('Finished execution of the benchmarks')

    return results
