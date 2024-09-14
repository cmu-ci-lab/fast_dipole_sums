scans="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"

for scan in $scans; do
    python3 exp_runner.py --conf confs/dtu.conf --case dtu/dtu_scan$scan
done
