import json
from myutils.physical_db import DBConnection

db_connection = DBConnection()
# 不用COUNT(*)可以获得join size
# sql = 'SELECT COUNT(*) FROM cast_info ci, title t WHERE t.id=ci.movie_id AND t.production_year<2011 AND t.phonetic_code<11942 AND t.series_years<296 AND ci.role_id>2;'
sql = 'SELECT * FROM cast_info ci, title t WHERE t.id=ci.movie_id AND t.phonetic_code<11364 AND t.series_years<428 AND t.production_year=2009 AND ci.role_id<4;'
res = db_connection.get_query_explain(sql)

with open('./metric_result/explain_result_2.json', 'w') as f:
    json.dump(res, f)