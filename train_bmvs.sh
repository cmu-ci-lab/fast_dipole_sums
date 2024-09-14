scans="basketball bear bread camera clock cow dog doll dragon durian fountain gundam house jade man monster sculpture stone"

for scan in $scans; do
    python3 exp_runner.py --conf confs/bmvs.conf --case bmvs/$scan
done