# script used to setup on AWS EC2 p2.*

# PRECONDITION:
# - image based on ubuntu 16.04 AMI
# - docker already installed

# install nvidia
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo apt-get update && sudo apt-get install -y --no-install-recommends linux-headers-generic dkms cuda-drivers
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
sudo nvidia-docker run --rm nvidia/cuda nvidia-smi

# setup cress-classify
git clone https://github.com/mfa/cress-classify.git
cd cress-classify/gpu/
sudo ./nvidia.sh
sudo nvidia-docker build -f Dockerfile.gpu --tag cress_classify .

cd notebook/data/

# download csv of photos and sensor data
for x in $(seq 54 112); do wget https://cress.space/csv/photo_cycle_${x}_enriched.csv; done

mkdir cache
cd cache/
# download hdf5 of images
for x in $(seq 54 112); do
    mkdir $x
    wget https://cress.space/hdf5/cycle${x}_hdf5_cache.zip
    cd $x
    unzip ../cycle${x}_hdf5_cache.zip
    cd ..
    rm cycle${x}_hdf5_cache.zip
done
