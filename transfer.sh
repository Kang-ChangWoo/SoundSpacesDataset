for scene_dir in /home/rvi-lab/workspace/sound-spaces/dataset/*/; do
    scene=$(basename "$scene_dir")
    echo "=== Transferring $scene ==="
    while ! rsync -avz --partial --progress \
        -e "ssh -p 55555 -o ServerAliveInterval=15" \
        "$scene_dir" \
        rvi@n3.unist.info:/data/changwoo/matterport3d/"$scene"/; do
        echo "Retrying $scene in 3s..."; sleep 3
    done
    echo "=== $scene done ==="
done
echo "ALL DONE!"