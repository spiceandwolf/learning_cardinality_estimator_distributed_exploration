import numpy as np
import pandas as pd

from myutils.graph_representation import QueryType, Table

def read_table_csv(table, csv_seperator=','):
    """
    从csv文件中读取数据还原table
    """
    df_rows = pd.read_csv(table.csv_file_location, header=None, escapechar='\\', 
                          encoding='utf-8', quotechar='"', sep=csv_seperator, low_memory= False)
    df_rows.columns = [table.table_name + '.' + attr for attr in table.attributes]
    
    for attribute in table.irrelevant_attributes:
        df_rows = df_rows.drop(table.table_name + '.' + attribute, axis=1)

    return df_rows.apply(pd.to_numeric, errors="ignore")

def write_table_csv(table_path, df_rows, csv_seperator=','):
    """
    根据table生成csv
    """
    df_rows.to_csv(table_path, index=False, sep=csv_seperator)

def add_header(table_path, names, csv_seperator=','):
    df_rows = pd.read_csv(table_path, header=None, names=names, escapechar='\\', 
                          encoding='utf-8', quotechar='"', sep=csv_seperator, low_memory= False)
    df_rows.to_csv(table_path, index=False, sep=csv_seperator)

def Discretize(col, data=None):
    """Transforms data values into integers using a Column's vocab.

    Args:
        col: the Column.
        data: list-like data to be discretized.  If None, defaults to col.data.

    Returns:
        col_data: discretized version; an np.ndarray of type np.int32.
    """
    # pd.Categorical() does not allow categories be passed in an array
    # containing np.nan.  It makes it a special case to return code -1
    # for NaN values.

    if data is None:
        data = col.data

    # pd.isnull returns true for both np.nan and np.datetime64('NaT').
    isnan = pd.isnull(col.all_distinct_values)
    if isnan.any():
        # We always add nan or nat to the beginning.
        assert isnan.sum() == 1, isnan
        assert isnan[0], isnan

        dvs = col.all_distinct_values[1:]
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data)

        # Since nan/nat bin_id is supposed to be 0 but pandas returns -1, just
        # add 1 to everybody
        bin_ids = bin_ids + 1
    else:
        # This column has no nan or nat values.
        dvs = col.all_distinct_values
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data), (len(bin_ids), len(data))

    bin_ids = bin_ids.astype(np.int32, copy=False)
    assert (bin_ids >= 0).all(), (col, data, bin_ids)
    return bin_ids
    
csv_path_to_be_handel = '~/oblab/CardinalityEstimationTestbed/Overall/train-test-data/imdbdata-num' + '/{}.csv'

table_infos = {
    # 'title':['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
    #             'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
    #             'series_years', 'md5sum'], 
    'movie_info_idx':['id', 'movie_id', 'info_type_id', 'info', 'note'],
    'movie_info':['id', 'movie_id', 'info_type_id', 'info', 'note'],
    'info_type':['id', 'info'],
    'cast_info':['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order','role_id'],
    'char_name':['id', 'name', 'imdb_index', 'imdb_id', 'name_pcode_nf','surname_pcode', 'md5sum'],
    'role_type':['id', 'role'],
    'complete_cast':['id', 'movie_id', 'subject_id', 'status_id'],
    'comp_cast_type':['id', 'kind'],
    'name':['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf','name_pcode_nf', 'surname_pcode', 'md5sum'],
    'aka_name':['id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf',
                'name_pcode_nf', 'surname_pcode', 'md5sum'],
    'movie_keyword':['id', 'movie_id', 'keyword_id'],
    'keyword':['id', 'keyword', 'phonetic_code'],
    'person_info':['id', 'person_id', 'info_type_id', 'info', 'note'],
    'movie_companies':['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
    'company_name':['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf',
                                                       'name_pcode_sf', 'md5sum'],
    'company_type':['id', 'kind'],
    'aka_title':['id', 'movie_id', 'title', 'imdb_index', 'kind_id',
                     'production_year', 'phonetic_code', 'episode_of_id', 'season_nr',
                     'episode_nr', 'note', 'md5sum'],
    'kind_type':['id', 'kind'],
}

for table_info in table_infos.items():
    table_path = csv_path_to_be_handel.format(table_info[0])
    add_header(table_path, table_info[1])
