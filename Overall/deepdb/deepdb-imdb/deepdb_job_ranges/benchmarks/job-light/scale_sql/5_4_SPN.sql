SELECT COUNT(*) from title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx where t.id = mc.movie_id and t.id = ci.movie_id and t.id = mi.movie_id and t.id = mi_idx.movie_id and t.kind_id < 7 and t.production_year < 1999 and mc.company_type_id = 2 and mi_idx.info_type_id = 100;
SELECT COUNT(*) from title t,movie_companies mc,cast_info ci,movie_info mi,movie_keyword mk where t.id = mc.movie_id and t.id = ci.movie_id and t.id = mi.movie_id and t.id = mk.movie_id and ci.person_id < 926305 and ci.role_id < 10 and mi.info_type_id < 7 and mk.keyword_id > 1074;
SELECT COUNT(*) from title t,movie_companies mc,cast_info ci,movie_info mi,movie_keyword mk where t.id = mc.movie_id and t.id = ci.movie_id and t.id = mi.movie_id and t.id = mk.movie_id and t.kind_id = 7 and t.production_year = 2004 and ci.role_id < 2 and mk.keyword_id < 2909;
SELECT COUNT(*) from title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk where t.id = ci.movie_id and t.id = mi.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and t.kind_id = 1 and t.production_year < 2008 and mi.info_type_id < 2 and mk.keyword_id < 3291;
SELECT COUNT(*) from title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk where t.id = mc.movie_id and t.id = ci.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and t.kind_id < 7 and mc.company_id < 428 and ci.person_id < 525577 and ci.role_id = 9;
SELECT COUNT(*) from title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk where t.id = mc.movie_id and t.id = ci.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and t.kind_id < 2 and mc.company_id > 80011 and ci.person_id = 613664 and ci.role_id = 1;
SELECT COUNT(*) from title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk where t.id = mc.movie_id and t.id = mi.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and t.kind_id = 7 and t.production_year > 2004 and mi_idx.info_type_id < 101 and mk.keyword_id < 44523;
SELECT COUNT(*) from title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx where t.id = mc.movie_id and t.id = ci.movie_id and t.id = mi.movie_id and t.id = mi_idx.movie_id and t.kind_id = 1 and ci.person_id > 813072 and ci.role_id = 1 and mi_idx.info_type_id = 99;
SELECT COUNT(*) from title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk where t.id = mc.movie_id and t.id = mi.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and mc.company_id > 34 and mc.company_type_id < 2 and mi_idx.info_type_id > 99 and mk.keyword_id < 335;
SELECT COUNT(*) from title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk where t.id = mc.movie_id and t.id = ci.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and t.kind_id < 7 and mc.company_id < 52087 and ci.person_id < 158834 and ci.role_id > 1;
SELECT COUNT(*) from title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk where t.id = mc.movie_id and t.id = mi.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and mc.company_id > 33285 and mc.company_type_id > 1 and mi.info_type_id < 13 and mk.keyword_id > 3921;
SELECT COUNT(*) from title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk where t.id = ci.movie_id and t.id = mi.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and ci.person_id < 463537 and mi.info_type_id > 16 and mi_idx.info_type_id < 101 and mk.keyword_id < 16037;
SELECT COUNT(*) from title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk where t.id = ci.movie_id and t.id = mi.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and t.production_year < 1973 and ci.role_id < 9 and mi.info_type_id = 6 and mk.keyword_id < 3736;
SELECT COUNT(*) from title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk where t.id = mc.movie_id and t.id = mi.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and t.kind_id = 7 and t.production_year = 2011 and mi.info_type_id = 16 and mk.keyword_id > 870;
SELECT COUNT(*) from title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk where t.id = mc.movie_id and t.id = mi.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and mc.company_id > 11323 and mc.company_type_id = 2 and mi.info_type_id > 3 and mk.keyword_id > 4962;
SELECT COUNT(*) from title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk where t.id = mc.movie_id and t.id = ci.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and mc.company_id > 525 and mc.company_type_id < 2 and ci.role_id = 1 and mi_idx.info_type_id > 100;
SELECT COUNT(*) from title t,cast_info ci,movie_info mi,movie_info_idx mi_idx,movie_keyword mk where t.id = ci.movie_id and t.id = mi.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and t.kind_id = 7 and mi.info_type_id = 9 and mi_idx.info_type_id < 100 and mk.keyword_id < 71574;
SELECT COUNT(*) from title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx where t.id = mc.movie_id and t.id = ci.movie_id and t.id = mi.movie_id and t.id = mi_idx.movie_id and t.production_year = 2007 and ci.role_id > 1 and mi.info_type_id > 8 and mi_idx.info_type_id < 100;
SELECT COUNT(*) from title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx,movie_keyword mk where t.id = mc.movie_id and t.id = mi.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and mc.company_id = 19 and mc.company_type_id < 2 and mi.info_type_id < 6 and mk.keyword_id < 6663;
SELECT COUNT(*) from title t,movie_companies mc,cast_info ci,movie_info mi,movie_info_idx mi_idx where t.id = mc.movie_id and t.id = ci.movie_id and t.id = mi.movie_id and t.id = mi_idx.movie_id and mc.company_id > 19661 and mc.company_type_id < 2 and mi.info_type_id < 18 and mi_idx.info_type_id > 99;
SELECT COUNT(*) from title t,movie_companies mc,cast_info ci,movie_info_idx mi_idx,movie_keyword mk where t.id = mc.movie_id and t.id = ci.movie_id and t.id = mi_idx.movie_id and t.id = mk.movie_id and mc.company_id < 145 and mc.company_type_id < 2 and ci.person_id > 1174178 and mi_idx.info_type_id < 101;