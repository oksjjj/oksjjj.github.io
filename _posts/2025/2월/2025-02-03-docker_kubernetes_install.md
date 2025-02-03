---
title: 도커/쿠버네티스 실습 환경 구축
author: oksjjj
date: 2025-02-03 13:02:00 +0900
categories: [도커/쿠버네티스]
tags: [docker, container, kubernetes, 쿠버네티스, k8s, 컨테이너, container]
render_with_liquid: false
---

## **개요**

리눅스 또는 윈도우 PC, 그리고 맥북에 도커/쿠버네티스를 설치하고,  
이후 pod, replicaset, deployment, service 등을 실습할 수 있는 매뉴얼을 작성해 보고자 합니다.  

도커/쿠버네티스 실습 환경을 구축하기 위해서, 책도 보고, 구글링도 했는데,  
항상 무언가 하나는 제대로 동작하지 않았습니다다.  

**그래서 숱하게 구글링을 해서, 현재까지는 100% 구동 가능한 버전을 정리한 것이 이 내용입니다.**  
**자신만의 도커/쿠버네티스 실습 환경을 구축해 보고 싶으신 분들은 아래 내용을 참고해서 고생을 덜 하기를 바랍니다^^**  

## **최소 사양**

예전에는 램 8GB로 구성이 가능했는데, 쿠버네티스 사양이 올라가면서 8GB로는 설치가 되지 않습니다.  
16GB의 PC에서 설치하는 것을 권장합니다.  

## **하이퍼바이저 설치**

설치하고자 하는 쿠버네티스 클러스터는 마스터 노드 1개, 워커 노드 2개로 구성되어 있습니다.  
VM을 3개 생성하기 위해 하이퍼바이저 설치가 필요합니다.  

각 OS별 추천 환경은 다음과 같습니다.  

| 구분 | 하이퍼바이저 | 비고 |
|-------|-------|-------|
| 윈도우 | VMWare Workstation Pro | 자체 네트워크 구성을 위해 PRo 버전 필요 |
| Mac OS | VMWare Fusion Pro | 자체 네트워크 구성을 위해 PRo 버전 필요 |
| 리눅스 | virt-manager | 무료 |

VMWare Pro 버전은 유료이기 때문에, 리눅스에서 virt-manager를 통해 설치하는 것을 추천합니다.  

## **가상 네트워크 생성**

VMWare 또는 virt-manager에서 쿠버네티스 노드 간의 통신에 쓰이는 가상 네트워크를 생성합니다.  

| 네트워크 타입 | 네트워크 이름 | 네트워크 대역 |
|-------|-------|-------|
| NAT | kube_network | 172.31.0.0/24 |

## **VM 생성**

### VM 생성을 위한 리눅스 이미지를 다운 받습니다.  
* ubuntu 24.04 (Server용 LTS)  
  (맥북 애플 실리콘 버전은 arm 아키텍쳐로 다운로드 받습니다)  

### 아래와 같이 VM을 3개 생성하고 ubuntu를 설치합니다.  

| VM 명칭 | OS 이미지 | CPU | RAM | Disk | NIC | IP |
|-------|-------|-------|-------|-------|-------|-------|
| kube-master | ubuntu 24.04 | 2 | 4096 | 40G | kube_network | 172.31.0.100 |
| kube-worker1 | ubuntu 24.04 | 2 | 2048 | 30G | kube_network | 172.31.0.101 |
| kube-worker2 | ubuntu 24.04 | 2 | 2048 | 30G | kube_network | 172.31.0.102 |


* ubuntu 설치 시 참고 사항
  * subnet : 172.31.0.0/24
  * dns, default gateway (VMWare) : 172.31.0.2
  * dns, default gateway (virt-manager) : 172.31.0.1
  * Install OpenSSH Server 체크

### VM 설치가 완료되면 모두 poweroff 하고, 1_os_install 이라는 명칭으로 스냅샷을 저장합니다.  
* 스냅샷은 최소 1GB 이상의 용량을 소모하므로, 필수적인 사항은 아닙니다  
* 이후 단계에서 쉽게 rollback 할 수 있으므로 스냅샷을 저장할 것을 권장합니다. 


## **도커 설치**

### 아래 사항들은 3개 VM에 대해 공통적으로 적용합니다.

### VM을 power on 하고 나서, root 패스워드를 설정 후 root 계정으로 전환합니다.

```
sudo passwd root
```

```
su -
```

### ubuntu 리눅스의 pkg를 업데이트 합니다.
```
apt update
apt upgrade
```

### timezone을 설정합니다.
```
sudo dpkg-reconfigure tzdata
```
* Asia - Seoul로 설정

### swap 기능 off
```
vi /etc/fstab
```
* swap 기능이 켜져 있으면 쿠버네티스 설치 단계에서 실패합니다.  
* 마지막 행인 swap 라인 앞에 #을 추가하여 주석 처리합니다.

### NTP 설정

[NTP 설정 관련 참고 (설정 명령어는 아래에 모두 기술되어 있음)](https://docs.openstack.org/ko_KR/install-guide/environment-ntp-controller.html)

#### master, worker 노드 모두 chrony를 설치하고 설정 파일을 엽니다.  
```
apt install chrony
vi /etc/chrony/chrony.conf
```

#### chrony.conf 파일 설정 방법

* master 노드 : 임의의 곳에 아래 문구를 추가하고, 파일 저장하고 닫습니다. 
```
allow 172.31.0.0/24
```

* worker 노드  
  * 아래 4개 라인을 삭제합니다
```
pool ntp.ubuntu.com        iburst maxsources 4
pool 0.ubuntu.pool.ntp.org iburst maxsources 1
pool 1.ubuntu.pool.ntp.org iburst maxsources 1
pool 2.ubuntu.pool.ntp.org iburst maxsources 2
```
  * 아래 라인을 추가하고, 파일 저장하고 닫습니다.
```
server 172.31.0.100 iburst
```

* master/worker 노드 모두 아래 명령어를 실행합니다.
```
service chrony restart
chronyc sources
```
  * master 노드는 외부 노드들을 바라 보는지 확인합니다.
  * worker 노드들은 모두 172.31.0.100을 바라 보는지 확인합니다.

### 서버 network 관련 커널 파라미터 수정

* 아래 명령어를 일괄 실행합니다.

```
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

sudo modprobe overlay
sudo modprobe br_netfilter

# 필요한 sysctl 파라미터를 설정하면, 재부팅 후에도 값이 유지된다.
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

# 재부팅하지 않고 sysctl 파라미터 적용하기
sudo sysctl --system
```

### 도커 pkg 설치

[도커 pkg 설치 관련 참고 (설정 명령어는 아래에 모두 기술되어 있음)](https://docs.docker.com/engine/install/ubuntu/)

* 아래 명령어를 일괄 실행합니다.  

```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

* 설치가 완료되면 아래 명령어를 실행하여 아래 라인이 출력되는지 확인합니다.  

```
docker ps -a
```

```
root@kube-master:~# docker ps -a
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
```

### 모두 poweroff 하고, 2_docker_install  이라는 명칭으로 스냅샷을 저장합니다. 


## **쿠버네티스 설치**

### 아래 사항들은 3개 VM에 대해 공통적으로 적용합니다.

### VM을 power on 하고 나서, root 계정으로 전환합니다.

```
su -
```

### go 언어 설치
* 현재 쿠버네티스에서는 공식적으로는 컨테이너 런타임으로써 도커를 지원하지 않습니다.
* 도커를 사용하기 위해서는 cri-docker를 설치해야 하는데, 이를 위해서 go 언어가 필요합니다.
* cri-docker 버전에 따라 설치해야 하는 go 언어의 버전이 달라질 수 있습니다.

#### x86 아키텍쳐용 go 언어 설치 방법 (주의: 애플 실리콘은 아래 명령어 실행 불가)

[x86 go 언어 설치 관련 참고 (설정 명령어는 아래에 모두 기술되어 있음)](https://tecadmin.net/how-to-install-go-on-ubuntu-20-04/)

* 아래 명령어 실행 후 go version이 출력되는지 확인합니다.  
```
wget https://go.dev/dl/go1.21.6.linux-amd64.tar.gz
sudo tar -xvf go1.21.6.linux-amd64.tar.gz -C /usr/local
echo "export PATH=$PATH:/usr/local/go/bin" >> ~/.profile
source ~/.profile
go version
```

#### arm 아키텍쳐용 go 언어 설치 방법 (애플 실리콘은 아래 명령어로 설치)

* 아래 명령어 실행 후 go version이 출력되는지 확인합니다.  
```
wget  https://go.dev/dl/go1.22.2.linux-arm64.tar.gz
sudo tar -xvf go1.22.2.linux-arm64.tar.gz -C /usr/local
echo "export PATH=$PATH:/usr/local/go/bin" >> ~/.profile
source ~/.profile
go version
```

#### x86, arm 모두 아래 명령어를 통해서 go 언어 설치 완료

```
apt install make
apt install gcc -y
```

### cri-docker 설치

※ 쿠버네티스의 컨테이너 런타임으로 도커를 사용하기 위해서 필요

[cri-docker 설치 관련 참고 (설정 명령어는 아래에 모두 기술되어 있음)](https://github.com/Mirantis/cri-dockerd)

* 아래 명령어를 일괄 실행합니다  

```
git clone https://github.com/Mirantis/cri-dockerd.git

cd cri-dockerd
make cri-dockerd

mkdir -p /usr/local/bin
install -o root -g root -m 0755 cri-dockerd /usr/local/bin/cri-dockerd
install packaging/systemd/* /etc/systemd/system
sed -i -e 's,/usr/bin/cri-dockerd,/usr/local/bin/cri-dockerd,' /etc/systemd/system/cri-docker.service
systemctl daemon-reload
systemctl enable --now cri-docker.socket
cd ..
```

### 쿠버네티스 pkg 설치

[쿠버네티스 pkg 설치 관련 참고 (설정 명령어는 아래에 모두 기술되어 있음)](https://kubernetes.io/ko/docs/setup/production-environment/tools/kubeadm/install-kubeadm/)

* 아래 명령어 실행를 일괄 실행합니다.  

```
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl

curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.28/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.28/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt update

sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl
```

### 쿠버네티스 셋업

※ (주의!) master와 worker 노드의 실행 명령어가 상이합니다  

#### kube-master 노드에서 아래 명령어를 실행합니다

```
kubeadm init --apiserver-advertise-address 172.31.0.100 --pod-network-cidr=192.168.0.0/16 --cri-socket=unix:///var/run/cri-dockerd.sock
```

* 위 명령어를 수행하면 맨 마지막에 아래와 같이 출력되는데

```
kubeadm join 172.31.0.100:6443 --token fqlhoe.274v1o931hvz1ydf \
	--discovery-token-ca-cert-hash sha256:a4aca0211449df54eccf0cbdca72c9216f87b41bd49ec2114a573ef3a3f14d89
```

#### 출력된 부분 뒤에 다음 내용을 붙여 넣어서 kube-worker1, kube-worker2에서 실행합니다  
```
 --cri-socket=unix:///var/run/cri-dockerd.sock
```

* 예시  
```
kubeadm join 172.31.0.100:6443 --token fqlhoe.274v1o931hvz1ydf \
--discovery-token-ca-cert-hash sha256:a4aca0211449df54eccf0cbdca72c9216f87b41bd49ec2114a573ef3a3f14d89 \
--cri-socket=unix:///var/run/cri-dockerd.sock
```
※ (주의!) 토큰의 만료 기한이 있기 때문에 master 명령어 수행 후 바로 worker 노드에서 실행해야 합니다  

#### kube-master에서 아래 명령어를 실행합니다

※ 아래 명령어를 실행해야 kubectl get nodes 명령어가 정상 동작합니다  

```
mkdir -p $HOME/.kube
cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
chown $(id -u):$(id -g) $HOME/.kube/config
```

#### kube-master에서 아래 명령어를 실행해서, kube-master 및 kube-worker1,2가 NotReady 상태로 출력되는지 확인합니다  

```
kubectl get nodes
```


#### Calico 설치
※ 쿠버네티스의 pod에 IP를 할당하기 위해서는 CNI(Container Network Interface) 설치가 필요합니다.  

[Calico 설치 관련 참고 (설정 명령어는 아래에 모두 기술되어 있음)](httpshttps://docs.tigera.io/calico/latest/getting-started/kubernetes/quickstart)

#### kube-master에서 아래 명령어를 실행하고, 모든 pod가 running 및 ready 상태가 될 때까지 기다랍니다  
  * 완료되면 ctrl+c 로 빠져 나옵니다  

```
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.27.3/manifests/tigera-operator.yaml
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.27.3/manifests/custom-resources.yaml
watch kubectl get pods -o wide -A
```

#### kube-master에서 아래 명령어를 실행해서 모든 노드가 Ready 상태로 출력되는지 확인합니다  

```
kubectl get nodes
```


## **오류 수정**

※ /var/log/syslog에 Deactivated successfully 가 지속적으로 출력되는 문제가 있는데 아래 방법으로 해결합니다  

### kube-master에서 아래 명령어를 실행합니다다  

```
vim /etc/systemd/journald.conf
```

### 아래 2개 단어가 있는 라인 2개를 주석 해제하고 (# 제거), 다음과 같이 수정합니다 (debug -> notice)  

* 수정 전  

```
#MaxLevelStore=debug
#MaxLevelSyslog=debug
```

* 수정 후  

```
MaxLevelStore=notice
MaxLevelSyslog=notice
```

### kube-master에서 아래 명령어를 실행합니다  

```
systemctl restart systemd-journald
```

### 모두 poweroff 하고, 3_kube_setup  이라는 명칭으로 스냅샷을 저장합니다다. 


## 실습 환경 구축 완료

여기까지 하면 도커/쿠너네티스 실습 환경은 완료되었습니다.  
원활한 실습을 위해서는 아래 내용을 꼭 참고하기 바랍니다.  

## **Troubleshooting**

* 실습을 위해 kube-master 및 모든 worker 노드들을 기동한 이후에는,  
  kube-master 노드에서 아래 명령어를 실행한 이후,  
  모든 노드의 pod들이 running 및 ready 상태 (또는 completed)가 된 것을 확인하고 실습을 진행해야 합니다.  

```
watch kubectl get pods -A -o wide
```

  * 저의 경우, 우분투 노트북의 virt-manager 에서는 조금만 기다리면 상태가 정상이 되는데,  
    맥북의 VMWare Fusion에서는 아무리 기다려도 상태가 정상이 되지 않았습니다.  
    이때는 아래와 같은 명령어를 통해 복구를 시도합니다.  

### 쿠버네티스 pod가 정상 기동되지 않을 경우  

**방법 1. 비정상 pod가 많은 노드(예: kube-master, kube-worker1 등)에서 root 계정으로 아래 명령어 수행 후 상태를 관찰합니다**  

```
docker stop $(docker ps -aq) && docker rm $(docker ps -aq)
```
**방법 2. 비정상 pod가 많은 노드(예: kube-master, kube-worker1 등)를 재기동(reboot)합니다**  

**방법 3.특정 pod만 삭제하여 재기동합니다 (kube-master에서 시행)**  

(예시)  
```
kubectl delete pod -n calico-system calico-typha-cf8ddc9ff-2kj5f
```

**인내심을 가지고 방법 1, 방법 2, 방법 3 등을 시도하다 보면, 모두 ready 상태가 됩니다**  

### kube-master에서 kubectl get nodes 수행 시 아래와 같은 에러 메시지가 출력될 경우  

```
The connection to the server localhost:8080 was refused - did you specify the right host or port?  
```

**다음 명령어를 통해 kube-master 노드의 모든 컨테이너를 종료/삭제 하면 해결됩니다**

```
docker stop $(docker ps -aq) && docker rm $(docker ps -aq)
```