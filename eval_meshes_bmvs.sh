scans="basketball bear bread camera clock cow dog doll dragon durian fountain gundam house jade man monster sculpture stone"
base_dir="exp/bmvs"

for scan in $scans; do
    python3 misc/clean_mesh_bmvs.py --base_dir $base_dir --mesh_name 00020000.ply --scan $scan;
    python3 misc/transform_mesh.py --base_dir $base_dir --mesh_name new_$scan.ply --scan $scan;
    python3 misc/bmvs_eval.py --base_dir $base_dir --scan $scan --mesh_name new_${scan}_scaled.ply;
done
