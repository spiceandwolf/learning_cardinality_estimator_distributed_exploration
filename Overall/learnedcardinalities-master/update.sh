python update_train.py --min-max-file ./data/col4_min_max_vals.csv --queries 10000 --epochs 101 --batch 1024 --hid 256 --train-query-file ../train-test-data/imdb-cols-sql/4/train-4-num.sql --test-query-file ../train-test-data/imdb-cols-sql/4/test-only4-num.sql --train --version imdb
python update_train.py --min-max-file ./data/col4_min_max_vals.csv --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/imdb-cols-sql/4/train-4-num.sql --test-query-file ../train-test-data/imdb-cols-sql/4/test-only4-num.sql --version imdb