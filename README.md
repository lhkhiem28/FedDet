This repository supports training YOLOv3 on the VOC2007 dataset in a federated paradigm. 

## Getting Started

### Dataset
You can download the prepared VOC2007 dataset [here](https://drive.google.com/file/d/1xeLIiurhhhrUFQwrjfU5Pkadiavo_VpU/view), which was split into 64 partitions (each corresponding to one client) in an IID manner. 

### Setup
This repository supports training on real IoT devices such as Raspberry Pi or Jetson series. I recommend using the Docker images I prepared to avoid painful errors:
* Pull Docker images for Raspberry Pi 4: `docker pull lhkhiem28/rpi:2.1`
* Pull Docker images for Jetson Nano: `docker pull lhkhiem28/jetson:2.2.1`

### Running
On the server:</br>
`python server.py`</br>
On client `i`:</br>
`python3 client.py --cid=i`</br>