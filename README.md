# Transferable Class Statistics and Multi-scale Feature Approximation for 3D Object Detection

# Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 20.04)
* Python 3.10
* PyTorch 2.4.1+cu118
* [`spconv v2.x`](https://github.com/traveller59/spconv)


### Install

a. Clone this repository.
```shell
git clone https://github.com/open-mmlab/OpenPCDet.git
```

b. Copy our documents to the openpcdet related folder。

c. Install the dependent libraries as follows:

* Install the dependent python libraries: 

```shell
pip install -r requirements.txt

```

* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 
   
d. Install this `pcdet` library and its dependent libraries by running the following command:
```shell
python setup.py develop
```
