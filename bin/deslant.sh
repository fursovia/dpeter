

DATA_DIR="data/rotated_images"
OUT_DIR="data/deslant_images"

mkdir -p ${OUT_DIR}


for img in ${DATA_DIR}/*.jpg; do
	img_name=$(basename ${img})
	echo "deslanting ${img_name}"
	/home/DeslantImg/DeslantImg ${img} ${OUT_DIR}/${img_name}
done
