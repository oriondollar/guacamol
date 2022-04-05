import datetime
import json
import logging
from collections import OrderedDict
from typing import List, Dict, Any

import guacamol
from guacamol.distribution_learning_benchmark import DistributionLearningBenchmark, DistributionLearningBenchmarkResult, \
    ValidityBenchmark, UniquenessBenchmark
from guacamol.standard_benchmarks import novelty_benchmark
from guacamol.sampling_helpers import sample_novel_molecules
from guacamol.benchmark_suites import distribution_learning_benchmark_suite
from guacamol.utils.data import get_time_string

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def assess_distribution_learning(model,
                                 train_file: str,
                                 test_file=None,
                                 test_scaffold_file=None,
                                 use_filters=False,
                                 json_output_file='output_distribution_learning.json',
                                 benchmark_version='v2') -> None:
    """
    Assesses a distribution-matching model for de novo molecule design.

    Args:
        model: Model to evaluate
        chembl_training_file: path to ChEMBL training set, necessary for some benchmarks
        json_output_file: Name of the file where to save the results in JSON format
        benchmark_version: which benchmark suite to execute
    """
    _assess_distribution_learning(model=model,
                                  train_file=train_file,
                                  test_file=test_file,
                                  test_scaffold_file=test_scaffold_file,
                                  use_filters=use_filters,
                                  json_output_file=json_output_file,
                                  benchmark_version=benchmark_version,
                                  number_samples=10000)


def _assess_distribution_learning(model,
                                  train_file: str,
                                  test_file: str,
                                  test_scaffold_file: str,
                                  use_filters: bool,
                                  json_output_file: str,
                                  benchmark_version: str,
                                  number_samples: int) -> None:
    """
    Internal equivalent to assess_distribution_learning, but allows for a flexible number of samples.
    To call directly only for testing.
    """
    logger.info(f'Benchmarking distribution learning, version {benchmark_version}')
    benchmarks = distribution_learning_benchmark_suite(train_file_path=train_file,
                                                       test_file_path=test_file,
                                                       test_scaffold_file_path=test_scaffold_file,
                                                       version_name=benchmark_version,
                                                       number_samples=number_samples)

    if version_name == 'v1':
        results = _evaluate_distribution_learning_benchmarks(model=model, benchmarks=benchmarks)
    elif version_name == 'v2':
        results = []
        validity = ValidityBenchmark(number_samples=number_samples,
                                     use_filters=use_filters)
        uniqueness = UniquenessBenchmark(number_samples=number_samples,
                                         use_filters=use_filters)
        novelty = novelty_benchmark(train_file_path, number_samples, use_filters=use_filters)

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
        result, novel = novelty.assess_model(model, unique)
        results.append(result)
        print(f'Results for the benchmark novelty:')
        print(f'  Score: {result.score:.6f}')
        print(f'  Sampling time: {str(datetime.timedelta(seconds=int(result.sampling_time)))}')
        print(f'  Metadata: {result.metadata}')
        if len(novel) < number_samples:
            print(f'Upsampling to specified number of novel molecules')
            novel = sample_novel_molecules(model, number_molecules=number_samples,
                                           train_file=train_file_path, prior_gen=novel,
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
                                               prior_gen=None) \
                                               -> List[DistributionLearningBenchmarkResult]:
    """
    Evaluate a model with the given benchmarks.
    Should not be called directly except for testing purposes.

    Args:
        model: model to assess
        benchmarks: list of benchmarks to evaluate
        json_output_file: Name of the file where to save the results in JSON format
    """

    print(f'Number of benchmarks: {len(benchmarks)}')

    for i, benchmark in enumerate(benchmarks, 1):
        print(f'Running benchmark {i}/{len(benchmarks)}: {benchmark.name}')
        result = benchmark.assess_model(model, prior_gen)
        print(f'Results for the benchmark "{result.benchmark_name}":')
        print(f'  Score: {result.score:.6f}')
        print(f'  Sampling time: {str(datetime.timedelta(seconds=int(result.sampling_time)))}')
        print(f'  Metadata: {result.metadata}')
        results.append(result)

    logger.info('Finished execution of the benchmarks')

    return results
