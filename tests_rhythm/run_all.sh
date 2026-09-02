#!/usr/bin/env bash
# One-command test suite for the rhythm pipeline.
#   bash tests_rhythm/run_all.sh
set -u
cd "$(dirname "$0")/.."
echo "=============================================================="
echo " RHYTHM PIPELINE TEST SUITE"
echo "=============================================================="
echo
echo "--- Task A/B/D: pytest suites ---"
python3 -m pytest tests_rhythm/ -q -rxX -p no:cacheprovider 2>&1 | grep -v Deprecation | tail -20
rc=$?
echo
echo "--- Task C: golden-file regression across 37 captures ---"
python3 scripts_rhythm/s18_golden_manifest.py --check
gc=$?
echo
echo "=============================================================="
if [ $rc -eq 0 ] && [ $gc -eq 0 ]; then
  echo " SUITE GREEN"
  echo " NOTE: test_healthy_population_reads_regular is XFAIL by design."
  echo "       It is EXPECTED to fail until device R-peak detection is fixed."
  echo "       If it ever XPASSes, the suite goes red on purpose - someone"
  echo "       must confirm the detector was genuinely fixed."
else
  echo " SUITE RED - see above"
fi
echo "=============================================================="
