SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years>596 AND t.production_year>2001 AND t.kind_id<7 AND t.phonetic_code<14575 AND ci.role_id>2 AND mi.info_type_id<17;,1347319
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.phonetic_code<11442 AND t.production_year<1980 AND t.series_years>393 AND t.kind_id<7 AND ci.role_id=1 AND mi.info_type_id=4;,24250
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year>1996 AND t.phonetic_code>7319 AND t.series_years>588 AND t.kind_id<7 AND ci.role_id>1 AND mi.info_type_id=4;,256653
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years<1166 AND t.production_year>1974 AND t.kind_id<7 AND t.phonetic_code<15017 AND ci.role_id>1 AND mi.info_type_id>105;,155
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years<745 AND t.production_year<2001 AND t.kind_id=7 AND t.phonetic_code<16253 AND ci.role_id>1 AND mi.info_type_id>3;,938
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id=7 AND t.series_years<1186 AND t.production_year<2011 AND t.phonetic_code>7404 AND mi.info_type_id<3 AND ci.role_id=1;,232
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.series_years>708 AND t.production_year<2005 AND t.phonetic_code=5947 AND mi.info_type_id<16 AND ci.role_id>1;,30
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.series_years<342 AND t.production_year<2011 AND t.phonetic_code<1746 AND mi.info_type_id=7 AND ci.role_id<2;,715
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years>752 AND t.kind_id<7 AND t.production_year>2005 AND t.phonetic_code>4388 AND mi.info_type_id>2 AND ci.role_id=2;,193238
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years>270 AND t.kind_id<7 AND t.production_year>2012 AND t.phonetic_code>10608 AND ci.role_id=1 AND mi.info_type_id=8;,1784
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.phonetic_code>323 AND t.production_year=2001 AND t.series_years>1039 AND t.kind_id<7 AND mi.info_type_id<3 AND ci.role_id>1;,40954
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.phonetic_code<14648 AND t.production_year<2011 AND t.kind_id<7 AND t.series_years=625 AND mi.info_type_id<3 AND ci.role_id=1;,14
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year<2010 AND t.series_years<917 AND t.phonetic_code>15859 AND t.kind_id=7 AND ci.role_id>2 AND mi.info_type_id=4;,35
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.production_year<2011 AND t.phonetic_code>5321 AND t.series_years>477 AND ci.role_id=2 AND mi.info_type_id>107;,80
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.phonetic_code<16746 AND t.kind_id<7 AND t.series_years>139 AND t.production_year>2004 AND ci.role_id=1 AND mi.info_type_id>16;,88926
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.series_years<935 AND t.phonetic_code>1302 AND t.production_year<2011 AND mi.info_type_id>1 AND ci.role_id<4;,1441779
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.phonetic_code>1137 AND t.series_years>1077 AND t.production_year=2005 AND t.kind_id<7 AND mi.info_type_id=3 AND ci.role_id>1;,42589
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years<810 AND t.kind_id=7 AND t.phonetic_code<1599 AND t.production_year>1969 AND ci.role_id=1 AND mi.info_type_id=8;,17
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.phonetic_code>1975 AND t.series_years>124 AND t.production_year>2010 AND mi.info_type_id>16 AND ci.role_id>1;,68970
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.phonetic_code<2739 AND t.production_year<2005 AND t.kind_id<7 AND t.series_years>786 AND mi.info_type_id>16 AND ci.role_id>2;,216617
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years>107 AND t.production_year>2010 AND t.phonetic_code<1840 AND t.kind_id<7 AND ci.role_id=1 AND mi.info_type_id<3;,3704
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.series_years>405 AND t.production_year<2008 AND t.phonetic_code>19832 AND ci.role_id=1 AND mi.info_type_id>16;,40030
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year>2010 AND t.series_years>140 AND t.phonetic_code>19992 AND t.kind_id<7 AND ci.role_id=1 AND mi.info_type_id=3;,1590
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years>536 AND t.production_year>1998 AND t.kind_id<7 AND t.phonetic_code>20796 AND mi.info_type_id=7 AND ci.role_id=1;,969
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.production_year<1985 AND t.phonetic_code<17825 AND t.series_years=844 AND ci.role_id=1 AND mi.info_type_id<7;,229
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years>1048 AND t.phonetic_code>17185 AND t.kind_id<7 AND t.production_year>2006 AND mi.info_type_id<3 AND ci.role_id=1;,12641
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id=7 AND t.production_year<2010 AND t.phonetic_code>17007 AND t.series_years>414 AND ci.role_id=1 AND mi.info_type_id>8;,67
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years<1048 AND t.kind_id=7 AND t.phonetic_code>1152 AND t.production_year<2009 AND mi.info_type_id<2 AND ci.role_id=2;,37
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.phonetic_code<20113 AND t.production_year<1996 AND t.kind_id<7 AND t.series_years>599 AND ci.role_id=1 AND mi.info_type_id=8;,110362
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years=435 AND t.phonetic_code<20392 AND t.kind_id<7 AND t.production_year<2004 AND mi.info_type_id=8 AND ci.role_id>1;,10
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years<392 AND t.phonetic_code<21003 AND t.production_year<2011 AND t.kind_id<7 AND ci.role_id>1 AND mi.info_type_id>3;,135759
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year<1984 AND t.phonetic_code<5930 AND t.series_years<662 AND t.kind_id=7 AND ci.role_id=2 AND mi.info_type_id>3;,5
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.production_year>1992 AND t.kind_id<7 AND t.phonetic_code>12332 AND t.series_years>409 AND mi.info_type_id=7 AND ci.role_id=1;,14132
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.phonetic_code>18261 AND t.kind_id=7 AND t.series_years<1119 AND t.production_year<2003 AND ci.role_id=1 AND mi.info_type_id>4;,31
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.phonetic_code<16709 AND t.series_years>761 AND t.production_year<1985 AND t.kind_id<7 AND mi.info_type_id>7 AND ci.role_id<2;,91607
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.series_years>590 AND t.production_year=2009 AND t.phonetic_code<13945 AND mi.info_type_id=3 AND ci.role_id=1;,7354
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years>552 AND t.phonetic_code=3895 AND t.production_year<1997 AND t.kind_id<7 AND ci.role_id>1 AND mi.info_type_id<7;,37
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year=2012 AND t.kind_id<7 AND t.series_years>1268 AND t.phonetic_code>11397 AND ci.role_id=1 AND mi.info_type_id>16;,5155
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id=7 AND t.phonetic_code>525 AND t.production_year>1953 AND t.series_years>711 AND ci.role_id=1 AND mi.info_type_id<4;,26
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years>235 AND t.production_year<2007 AND t.phonetic_code>1477 AND t.kind_id<7 AND mi.info_type_id>13 AND ci.role_id>1;,2718442
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.phonetic_code<14409 AND t.production_year>1951 AND t.series_years<978 AND ci.role_id=1 AND mi.info_type_id=81;,80
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.production_year<2010 AND t.series_years>396 AND t.phonetic_code=1853 AND t.kind_id<7 AND mi.info_type_id<6 AND ci.role_id=4;,233
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year>1959 AND t.series_years<977 AND t.kind_id<7 AND t.phonetic_code=270 AND ci.role_id=4 AND mi.info_type_id=16;,1
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.series_years>193 AND t.phonetic_code>10515 AND t.production_year>1995 AND ci.role_id<2 AND mi.info_type_id>2;,575882
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years>173 AND t.kind_id<7 AND t.production_year>1998 AND t.phonetic_code>16853 AND ci.role_id=1 AND mi.info_type_id>5;,120138
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.series_years>319 AND t.phonetic_code<2962 AND t.production_year<1963 AND mi.info_type_id>4 AND ci.role_id>1;,21770
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years>538 AND t.kind_id<7 AND t.production_year<1986 AND t.phonetic_code<16131 AND ci.role_id<2 AND mi.info_type_id=3;,51348
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years>668 AND t.production_year>2001 AND t.kind_id<7 AND t.phonetic_code<5237 AND ci.role_id>1 AND mi.info_type_id<2;,51700
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years=1188 AND t.kind_id<7 AND t.production_year<2006 AND t.phonetic_code<20811 AND ci.role_id=2 AND mi.info_type_id>7;,279
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years>105 AND t.production_year=2009 AND t.kind_id<7 AND t.phonetic_code<6645 AND ci.role_id=1 AND mi.info_type_id>2;,17985
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.phonetic_code>2696 AND t.production_year<2010 AND t.series_years>1073 AND mi.info_type_id=1 AND ci.role_id>2;,165090
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year<2007 AND t.phonetic_code>11767 AND t.kind_id<7 AND t.series_years<363 AND ci.role_id>1 AND mi.info_type_id<6;,17700
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year<2010 AND t.phonetic_code>11617 AND t.kind_id=7 AND t.series_years<165 AND ci.role_id>1 AND mi.info_type_id<3;,176
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years=144 AND t.phonetic_code<17705 AND t.kind_id<7 AND t.production_year<1978 AND ci.role_id>2 AND mi.info_type_id<2;,3
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.production_year=1991 AND t.phonetic_code<1718 AND t.series_years>139 AND mi.info_type_id=18 AND ci.role_id>1;,492
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.phonetic_code>2035 AND t.kind_id=7 AND t.series_years<561 AND t.production_year<2010 AND mi.info_type_id>6 AND ci.role_id>1;,632
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.phonetic_code<9925 AND t.production_year<1985 AND t.kind_id<7 AND t.series_years>181 AND ci.role_id=1 AND mi.info_type_id=3;,39891
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year<2010 AND t.kind_id<7 AND t.phonetic_code=16804 AND t.series_years<840 AND ci.role_id<2 AND mi.info_type_id<15;,202
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year>1990 AND t.series_years>421 AND t.kind_id<7 AND t.phonetic_code<16015 AND ci.role_id>1 AND mi.info_type_id=13;,27221
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.production_year<2007 AND t.kind_id<7 AND t.series_years=1194 AND t.phonetic_code<3917 AND mi.info_type_id=8 AND ci.role_id>1;,20
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id=7 AND t.phonetic_code<1469 AND t.series_years<1255 AND t.production_year<2011 AND ci.role_id>1 AND mi.info_type_id<15;,353
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.phonetic_code<18978 AND t.series_years=759 AND t.kind_id<7 AND t.production_year>1959 AND ci.role_id=1 AND mi.info_type_id=7;,109
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.phonetic_code<6855 AND t.series_years<115 AND t.kind_id=7 AND t.production_year<2003 AND ci.role_id=1 AND mi.info_type_id>3;,487
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years<565 AND t.phonetic_code<16035 AND t.kind_id<7 AND t.production_year<2008 AND mi.info_type_id>2 AND ci.role_id<2;,461684
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.phonetic_code=16804 AND t.kind_id<7 AND t.series_years>711 AND t.production_year<1996 AND ci.role_id=1 AND mi.info_type_id>17;,41
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years<813 AND t.production_year<2005 AND t.kind_id<7 AND t.phonetic_code>14687 AND ci.role_id=1 AND mi.info_type_id=8;,14443
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years<634 AND t.phonetic_code>2007 AND t.production_year>2007 AND t.kind_id=7 AND mi.info_type_id>15 AND ci.role_id=1;,207
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.production_year<2012 AND t.kind_id<7 AND t.phonetic_code<21698 AND t.series_years>1214 AND mi.info_type_id=4 AND ci.role_id<2;,156469
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id=7 AND t.phonetic_code<2692 AND t.production_year<1998 AND t.series_years<604 AND mi.info_type_id>6 AND ci.role_id>4;,129
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.phonetic_code>13768 AND t.kind_id=7 AND t.series_years<787 AND t.production_year<2010 AND mi.info_type_id=16 AND ci.role_id=1;,159
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.phonetic_code>8509 AND t.kind_id<7 AND t.production_year<2004 AND t.series_years=1160 AND mi.info_type_id>1 AND ci.role_id>1;,320
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years>651 AND t.phonetic_code=10439 AND t.kind_id<7 AND t.production_year<2013 AND mi.info_type_id=16 AND ci.role_id=1;,23
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year<1993 AND t.phonetic_code<669 AND t.series_years>811 AND t.kind_id<7 AND ci.role_id>1 AND mi.info_type_id=7;,1382
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years<913 AND t.kind_id<7 AND t.production_year<1967 AND t.phonetic_code<3063 AND mi.info_type_id<3 AND ci.role_id>2;,7339
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.phonetic_code<6512 AND t.series_years>174 AND t.production_year<1968 AND t.kind_id<7 AND ci.role_id<2 AND mi.info_type_id=15;,262434
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year>2002 AND t.series_years<631 AND t.kind_id=7 AND t.phonetic_code<6653 AND ci.role_id>1 AND mi.info_type_id>4;,22
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id=7 AND t.series_years<475 AND t.phonetic_code<6727 AND t.production_year>1960 AND ci.role_id=2 AND mi.info_type_id=2;,9
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id=7 AND t.phonetic_code<12994 AND t.series_years<406 AND t.production_year<2010 AND mi.info_type_id<6 AND ci.role_id=1;,249
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.phonetic_code>765 AND t.series_years<883 AND t.production_year<2011 AND mi.info_type_id<2 AND ci.role_id=1;,44601
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.series_years>891 AND t.phonetic_code<15054 AND t.production_year<1996 AND mi.info_type_id=2 AND ci.role_id>2;,50318
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year>1956 AND t.series_years>1095 AND t.kind_id<7 AND t.phonetic_code<21686 AND ci.role_id<2 AND mi.info_type_id=3;,263674
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years<1229 AND t.phonetic_code<2968 AND t.production_year<2010 AND t.kind_id=7 AND mi.info_type_id<7 AND ci.role_id=1;,125
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years<262 AND t.phonetic_code<12015 AND t.kind_id=7 AND t.production_year<1965 AND ci.role_id<2 AND mi.info_type_id>8;,4
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years>333 AND t.production_year>2007 AND t.phonetic_code>15740 AND t.kind_id<7 AND mi.info_type_id<4 AND ci.role_id>1;,97102
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id=7 AND t.production_year<2008 AND t.phonetic_code<6279 AND t.series_years<916 AND mi.info_type_id>2 AND ci.role_id>1;,497
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.series_years<940 AND t.production_year<2000 AND t.phonetic_code>14216 AND mi.info_type_id=8 AND ci.role_id>1;,30168
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.production_year<2006 AND t.series_years<337 AND t.phonetic_code>6454 AND mi.info_type_id=16 AND ci.role_id>2;,2738
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.phonetic_code<16025 AND t.production_year<2007 AND t.series_years<538 AND t.kind_id<7 AND mi.info_type_id>1 AND ci.role_id<2;,469994
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.series_years=1215 AND t.phonetic_code>5467 AND t.production_year<2002 AND mi.info_type_id=8 AND ci.role_id<2;,3
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year>2006 AND t.kind_id<7 AND t.phonetic_code>17182 AND t.series_years>979 AND ci.role_id>2 AND mi.info_type_id=7;,6586
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.series_years<1282 AND t.phonetic_code>19023 AND t.production_year<2011 AND mi.info_type_id>17 AND ci.role_id>2;,100485
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.series_years<1009 AND t.phonetic_code<14700 AND t.production_year<2011 AND mi.info_type_id=5 AND ci.role_id<2;,12781
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years>1109 AND t.production_year<2008 AND t.phonetic_code<13316 AND t.kind_id<7 AND ci.role_id>1 AND mi.info_type_id=16;,293979
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.series_years>1113 AND t.production_year<2012 AND t.phonetic_code>12286 AND mi.info_type_id=8 AND ci.role_id<2;,72008
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.production_year>1959 AND t.kind_id=7 AND t.phonetic_code<3956 AND t.series_years<353 AND mi.info_type_id>3 AND ci.role_id<2;,512
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.production_year>1995 AND t.phonetic_code>2565 AND t.kind_id<7 AND t.series_years>1000 AND mi.info_type_id>7 AND ci.role_id>2;,1716211
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years>407 AND t.production_year>2010 AND t.phonetic_code>5342 AND t.kind_id<7 AND ci.role_id<2 AND mi.info_type_id<2;,4971
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.production_year<2005 AND t.series_years>1233 AND t.phonetic_code>2522 AND mi.info_type_id=3 AND ci.role_id<2;,93462
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.phonetic_code>17708 AND t.production_year<2013 AND t.kind_id=7 AND t.series_years>649 AND mi.info_type_id<104 AND ci.role_id<2;,67
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.production_year>2009 AND t.phonetic_code>4432 AND t.series_years>933 AND mi.info_type_id=4 AND ci.role_id=2;,14562
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years>714 AND t.production_year=2012 AND t.kind_id<7 AND t.phonetic_code<16559 AND mi.info_type_id<16 AND ci.role_id>1;,107302
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.phonetic_code>10359 AND t.production_year<2008 AND t.series_years<807 AND ci.role_id=1 AND mi.info_type_id<3;,38923
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.phonetic_code>9754 AND t.series_years>617 AND t.production_year=1979 AND t.kind_id<7 AND mi.info_type_id<16 AND ci.role_id=2;,8967
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year<2008 AND t.kind_id<7 AND t.phonetic_code<11315 AND t.series_years=601 AND ci.role_id>2 AND mi.info_type_id<16;,782
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.phonetic_code<17978 AND t.production_year<2004 AND t.kind_id<7 AND t.series_years>256 AND mi.info_type_id=16 AND ci.role_id=1;,162738
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.series_years=141 AND t.phonetic_code<6653 AND t.production_year<2011 AND mi.info_type_id=2 AND ci.role_id>2;,15
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year<1983 AND t.phonetic_code<5060 AND t.series_years<387 AND t.kind_id=7 AND ci.role_id<2 AND mi.info_type_id>7;,2
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years<1297 AND t.production_year<2009 AND t.kind_id=7 AND t.phonetic_code>11515 AND mi.info_type_id>7 AND ci.role_id<2;,341
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.phonetic_code>2340 AND t.production_year<1995 AND t.series_years<396 AND t.kind_id<7 AND ci.role_id=2 AND mi.info_type_id=1;,3769
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year<2012 AND t.phonetic_code<19998 AND t.series_years=484 AND t.kind_id<7 AND ci.role_id=2 AND mi.info_type_id<16;,18
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.phonetic_code>4642 AND t.kind_id<7 AND t.series_years>960 AND t.production_year>2011 AND mi.info_type_id<16 AND ci.role_id<2;,47020
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.production_year<1997 AND t.phonetic_code>5143 AND t.series_years>876 AND ci.role_id>1 AND mi.info_type_id=16;,98920
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year<2008 AND t.kind_id=7 AND t.series_years>220 AND t.phonetic_code<7511 AND ci.role_id>1 AND mi.info_type_id<8;,26
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year<1959 AND t.phonetic_code>6582 AND t.kind_id<7 AND t.series_years=215 AND ci.role_id=2 AND mi.info_type_id<8;,35
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.phonetic_code>20144 AND t.kind_id<7 AND t.production_year=2011 AND t.series_years>499 AND ci.role_id=1 AND mi.info_type_id>2;,3311
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.production_year<2011 AND t.series_years>1016 AND t.kind_id<7 AND t.phonetic_code>17631 AND ci.role_id>2 AND mi.info_type_id=8;,46796
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.phonetic_code<15858 AND t.kind_id=7 AND t.production_year>1996 AND t.series_years<1074 AND ci.role_id=1 AND mi.info_type_id>1;,362
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.production_year<2009 AND t.phonetic_code>4421 AND t.series_years<1142 AND t.kind_id<7 AND mi.info_type_id=1 AND ci.role_id=2;,32735
SELECT COUNT(*) FROM cast_info ci, movie_info mi, title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.series_years>149 AND t.kind_id<7 AND t.production_year<2011 AND t.phonetic_code>1452 AND ci.role_id=1 AND mi.info_type_id<8;,1181342
SELECT COUNT(*) FROM movie_info mi, cast_info ci, title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.series_years>505 AND t.production_year<2009 AND t.kind_id=7 AND t.phonetic_code>9007 AND mi.info_type_id=16 AND ci.role_id=1;,105
