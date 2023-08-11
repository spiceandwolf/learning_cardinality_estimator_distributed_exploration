import argparse
import logging
import os
import shutil
import time

import numpy as np
from data_preparation.join_data_preparation import prepare_sample_hdf
from data_preparation.prepare_single_tables import prepare_all_tables
from ensemble_compilation.spn_ensemble import read_ensemble
from ensemble_creation.rdc_based import candidate_evaluation
from rspn.code_generation.generate_code import generate_ensemble_code
from schemas.imdb.schema import gen_power_schema


np.random.seed(1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='power', help='Which dataset to be used')  # imdb-light

    # generate hdf
    parser.add_argument('--generate_hdf', help='Prepare hdf5 files for single tables', action='store_true')
    parser.add_argument('--generate_sampled_hdfs', help='Prepare hdf5 files for single tables', action='store_true')
    parser.add_argument('--csv_seperator', default=',')
    parser.add_argument('--csv_path',
                        default='../../../train-test-data/forest_power-data-sql/no_head')  # No header has been modified
    parser.add_argument('--version', default='cols_4_distinct_1000_corr_5_skew_5')  # No header has been modified
    parser.add_argument('--hdf_path', default='../imdb-benchmark/gen_single_light')
    parser.add_argument('--max_rows_per_hdf_file', type=int, default=100000000)
    parser.add_argument('--hdf_sample_size', type=int, default=10000)
    parser.add_argument('--partition_num', type=int, default=1)

    # generate ensembles
    parser.add_argument('--generate_ensemble', help='Trains SPNs on schema', action='store_true')
    parser.add_argument('--ensemble_strategy', default='rdc_based')
    parser.add_argument('--ensemble_path', default='../imdb-benchmark/spn_ensembles')
    parser.add_argument('--pairwise_rdc_path', default='../imdb-benchmark/spn_ensembles/pairwise_rdc.pkl')
    parser.add_argument('--samples_rdc_ensemble_tests', type=int, default=10000)
    parser.add_argument('--samples_per_spn', help="How many samples to use for joins with n tables",
                        nargs='+', type=int, default=[10000000, 10000000, 1000000, 1000000, 1000000])
    parser.add_argument('--post_sampling_factor', nargs='+', type=int, default=[10, 10, 5, 1, 1])
    parser.add_argument('--rdc_threshold', help='If RDC value is smaller independence is assumed', type=float,
                        default=0.3)
    parser.add_argument('--bloom_filters', help='Generates Bloom filters for grouping', action='store_true')
    parser.add_argument('--ensemble_budget_factor', type=int, default=5)
    parser.add_argument('--ensemble_max_no_joins', type=int, default=3)
    parser.add_argument('--incremental_learning_rate', type=int, default=0)
    parser.add_argument('--incremental_condition', type=str, default=None)

    # generate code
    parser.add_argument('--code_generation', help='Generates code for trained SPNs for faster Inference',
                        action='store_true')
    parser.add_argument('--use_generated_code', action='store_true')

    # ground truth
    parser.add_argument('--aqp_ground_truth', help='Computes ground truth for AQP', action='store_true')
    parser.add_argument('--cardinalities_ground_truth', help='Computes ground truth for Cardinalities',
                        action='store_true')

    # evaluation
    parser.add_argument('--evaluate_cardinalities', help='Evaluates SPN ensemble to compute cardinalities',
                        action='store_true')
    parser.add_argument('--rdc_spn_selection', help='Uses pairwise rdc values to for the SPN compilation',
                        action='store_true')
    parser.add_argument('--evaluate_cardinalities_scale', help='Evaluates SPN ensemble to compute cardinalities',
                        action='store_true')
    parser.add_argument('--evaluate_aqp_queries', help='Evaluates SPN ensemble for AQP', action='store_true')
    parser.add_argument('--against_ground_truth', help='Computes ground truth for AQP', action='store_true')
    parser.add_argument('--evaluate_confidence_intervals',
                        help='Evaluates SPN ensemble and compares stds with true stds', action='store_true')
    parser.add_argument('--confidence_upsampling_factor', type=int, default=300)
    parser.add_argument('--confidence_sample_size', type=int, default=10000000)
    parser.add_argument('--ensemble_location', nargs='+',
                        default=['../ssb-benchmark/spn_ensembles/ensemble_single_ssb-500gb_10000000.pkl',
                                 '../ssb-benchmark/spn_ensembles/ensemble_relationships_ssb-500gb_10000000.pkl'])
    parser.add_argument('--query_file_location',
                        default='./benchmarks/ssb/sql/cardinality_queries.sql')  # External write
    parser.add_argument('--ground_truth_file_location',
                        default='./benchmarks/ssb/sql/cardinality_true_cardinalities_100GB.csv')  # External write
    parser.add_argument('--database_name', default='ai4db')  # modified
    parser.add_argument('--target_path', default='../ssb-benchmark/results')  # External write
    parser.add_argument('--raw_folder', default='../ssb-benchmark/results')  # External write
    parser.add_argument('--confidence_intervals', help='Compute confidence intervals', action='store_true')
    parser.add_argument('--max_variants', help='How many spn compilations should be computed for the cardinality '
                                               'estimation. Seeting this parameter to 1 means greedy strategy.',
                        type=int, default=1)  # External write
    parser.add_argument('--no_exploit_overlapping', action='store_true')
    parser.add_argument('--no_merge_indicator_exp', action='store_true')

    # evaluation of spn ensembles in folder
    parser.add_argument('--hdf_build_path', default='')

    # log level
    parser.add_argument('--log_level', type=int, default=logging.DEBUG)

    args = parser.parse_args()

    args.exploit_overlapping = not args.no_exploit_overlapping
    args.merge_indicator_exp = not args.no_merge_indicator_exp

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=args.log_level,
        # [%(threadName)-12.12s]
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("logs/{}_{}.log".format(args.dataset, time.strftime("%Y%m%d-%H%M%S"))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    # Generate schema
    
    table_csv_path = args.csv_path + '/{}.csv'
    partition = args.partition_num
    version = args.version 
    schema = gen_power_schema(table_csv_path, version, partition)
    

    # Generate HDF files for simpler sampling 
    if args.generate_hdf:
        logger.info(f"Generating HDF files for tables in {args.csv_path} and store to path {args.hdf_path}")

        if os.path.exists(args.hdf_path):
            logger.info(f"Removing target path {args.hdf_path}")
            shutil.rmtree(args.hdf_path)

        logger.info(f"Making target path {args.hdf_path}")
        os.makedirs(args.hdf_path)

        prepare_all_tables(schema, args.hdf_path, csv_seperator=args.csv_seperator,
                           max_table_data=args.max_rows_per_hdf_file)
        logger.info(f"Files successfully created")

    # Generate sampled HDF files for fast join calculations
    if args.generate_sampled_hdfs:
        logger.info(f"Generating sampled HDF files for tables in {args.csv_path} and store to path {args.hdf_path}")
        prepare_sample_hdf(schema, args.hdf_path, args.max_rows_per_hdf_file, args.hdf_sample_size,
                           version)  # Add a parameter
        logger.info(f"Files successfully created")
        # Pretime

    # Generate ensemble for cardinality schemas
    if args.generate_ensemble:
        time1 = time.time()

        logging.info(
                f"maqp(generate_ensemble: ensemble_strategy={args.ensemble_strategy}, incremental_learning_rate={args.incremental_learning_rate}, incremental_condition={args.incremental_condition}, ensemble_path={args.ensemble_path})")
        candidate_evaluation(version, partition, schema, args.hdf_path, args.samples_rdc_ensemble_tests, args.samples_per_spn,
                                 args.max_rows_per_hdf_file, args.ensemble_path, args.database_name,
                                 args.post_sampling_factor, args.ensemble_budget_factor, args.ensemble_max_no_joins,
                                 args.rdc_threshold, args.pairwise_rdc_path,
                                 incremental_learning_rate=args.incremental_learning_rate,
                                 incremental_condition=args.incremental_condition)  # Add a parameter
        # Traintime
        timetrain = time.time()
        # fmetric.write('Traintime: '+ str(timetrain-time1) + '\n')
        print('Traintime: \n', timetrain - time1)
        # fmetric.close()

    # Read pre-trained ensemble and evaluate cardinality queries
    if args.evaluate_cardinalities:
        from evaluation.cardinality_evaluation_distributed import evaluate_cardinalities

        logging.info(
            f"maqp(evaluate_cardinalities: database_name={args.database_name}, target_path={args.target_path})")
        evaluate_cardinalities(version, args.ensemble_location, args.database_name, args.query_file_location,
                               args.target_path,
                               schema, args.rdc_spn_selection, args.pairwise_rdc_path,
                               use_generated_code=args.use_generated_code,
                               merge_indicator_exp=args.merge_indicator_exp,
                               exploit_overlapping=args.exploit_overlapping, max_variants=args.max_variants,
                               true_cardinalities_path=args.ground_truth_file_location, min_sample_ratio=0)

