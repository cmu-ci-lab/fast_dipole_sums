scans="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
base_dir="exp/dtu"

for scan in $scans; do
    python3 misc/clean_mesh.py --base_dir $base_dir --mesh_name 00020000.ply --scan $scan;
    python3 misc/dtu_eval.py --base_dir $base_dir --scan $scan --mesh_name new_$scan.ply;
done
