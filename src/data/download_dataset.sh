DATASET=$1

if [ $DATASET == "celeba" ]; then
    # CelebA images
    URL=https://www.dropbox.com/s/ftcx1gf6tobtw08/celeba.zip?dl=0
    ZIP=./celeba.zip
    mkdir -p ./celeba
    wget -N $URL -O $ZIP
    unzip $ZIP -d ./celeba
    rm $ZIP
