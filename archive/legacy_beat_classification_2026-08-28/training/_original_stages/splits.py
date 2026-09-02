"""Single source of truth for every record-ID split used in training/eval.

MITDB_DS1 / MITDB_DS2 / SVDB_RECORDS were previously hardcoded inside
train_classifiers.py; moved here so eval_classifier.py (and any future
script) can reference the same lists without a circular import, and so
there is exactly one place that defines "what is DS1" (patient-level
splits are meaningless if two files can silently drift out of sync).

DS1_TRAIN / DS1_VAL: patient-level validation carve-out of DS1, added so
tuning decisions (feature engineering, class-imbalance handling) have an
honest held-out signal instead of being tuned directly against DS2 (DS2
must stay report-only — see RESEARCH_AUDIT.md).

Record selection for DS1_VAL was NOT random: MITDB record 208 alone
contains 373 of DS1's ~415 real F-class beats (F is already critically
scarce). Randomly assigning 208 to validation would gut training's only
real F signal. Records 118/201/207/223 were chosen instead because,
together, they carry meaningful S (404 of DS1's 944 S beats) and V (897)
support for validation while leaving 208 and the bulk of F-bearing
records (108/109/114/124/203/205/215) in training. Verified by direct
annotation count (AAMI-mapped) before finalizing:

  DS1_VAL beats  : S=404  V=897  F=16    (118:S96, 201:S128/V198/F2,
                    207:S107/V210, 223:V473/S73/F14)
  DS1_TRAIN beats: S=540  V~=rest F=399  (keeps 208's 373 F beats)

This is a one-time, documented, non-random choice — re-run the same
annotation-count check before changing it.
"""
from __future__ import annotations

MITDB_DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
             201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
MITDB_DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
             213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

SVDB_RECORDS = [800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812,
                820, 821, 822, 823, 824, 825, 826, 827, 828, 829,
                840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850,
                851, 852, 853, 854, 855, 856, 857, 858, 859, 860,
                861, 862, 863, 864, 865, 866, 867, 868, 869, 870,
                871, 872, 873, 874, 875, 876, 877, 878, 879, 880,
                881, 882, 883, 884, 885, 886, 887, 888, 889, 890,
                891, 892, 893, 894]

# INCART: all 75 records, already fully present at data/raw/public/incartdb
# (downloaded in an earlier session, per download_datasets.py's budget
# plan). Train-only enrichment, same as SVDB -- MITDB DS2 stays the only
# held-out test set. Filenames are "I01".."I75", not bare integers, so
# these are strings (build_dataset/_load_record_beats take str(rid) and
# glob for f"{rid}.hea", which works for either type).
INCART_RECORDS = [f"I{i:02d}" for i in range(1, 76)]

# LTAFDB (Long Term AF Database): 84 records, 24h+ Holter recordings, real
# per-beat N/V/A(->S) annotations (confirmed via a sample record before
# downloading the rest -- see ABLATION_REPORT.md / SESSION_LOG). Train-only.
# Fetched verbatim via wfdb.get_record_list("ltafdb") -- do not hand-edit.
LTAFDB_RECORDS = ["00", "01", "03", "05", "06", "07", "08", "10", "100", "101",
                   "102", "103", "104", "105", "11", "110", "111", "112", "113", "114",
                   "115", "116", "117", "118", "119", "12", "120", "121", "122", "13",
                   "15", "16", "17", "18", "19", "20", "200", "201", "202", "203",
                   "204", "205", "206", "207", "208", "21", "22", "23", "24", "25",
                   "26", "28", "30", "32", "33", "34", "35", "37", "38", "39",
                   "42", "43", "44", "45", "47", "48", "49", "51", "53", "54",
                   "55", "56", "58", "60", "62", "64", "65", "68", "69", "70",
                   "71", "72", "74", "75"]

# SDDB (Sudden Cardiac Death Holter Database): 23 records. Smaller than
# LTAFDB but notable for actually containing real F-class beats (a sample
# record had 75 F beats -- F has essentially no other real-data source in
# this pipeline besides MITDB/SVDB's ~410 examples). Train-only.
# Fetched verbatim via wfdb.get_record_list("sddb") -- do not hand-edit.
SDDB_RECORDS = ["30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
                 "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
                 "50", "51", "52"]

_DS1_VAL_RECORDS = [118, 201, 207, 223]

DS1_VAL = sorted(_DS1_VAL_RECORDS)
DS1_TRAIN = sorted(set(MITDB_DS1) - set(_DS1_VAL_RECORDS))

assert set(DS1_TRAIN) | set(DS1_VAL) == set(MITDB_DS1)
assert set(DS1_TRAIN) & set(DS1_VAL) == set()
assert set(DS1_VAL) & set(MITDB_DS2) == set()
assert set(DS1_TRAIN) & set(MITDB_DS2) == set()
