#!/usr/bin/bash

#List of commands to setup the system with all dependencies

echo "Update system"

apt update -y
apt upgrade -y

#zsh
apt install -y zsh
chsh -s "/usr/bin/zsh" root
apt install -y zsh-syntax-highlighting zsh-autosuggestions
apt install -y ranger
mv zshrc ~/.zshrc 
mv vimrc ~/.vimrc

#Basic and Nym dependencies
echo "Basic dependencies"
apt -y install pkg-config build-essential libssl-dev curl jq

#rustup
echo "Rust"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

#wireshark
echo "Wireshark"
add-apt-repository ppa:wireshark-dev/stable
apt update
apt install -y wireshark

#network
if [[ ! -d /sys/class/net/veth1 ]]; then ./setup_ns_eth0.sh ; else echo "network already exists"; fi

#nym
echo "Nym codebase"
if [[ ! -f nym/target/release/nym-network-requester ]]; then
git clone https://github.com/nymtech/nym.git
cd nym

## TODO change with the release you want, tested releases in the original paper was not 2024.1-marabou
git checkout release/ #Change to your version 
cargo build --release
#./target/release/nym-network-requester init --id nym-requester
#./target/release/nym-socks5-client init --id docs-example3 --provider nym-requester
cd
else echo "nym already exists"
fi

#capture script dependencies
echo "Capturing script dependencies : python"
apt install -y python3-pip
pip3 install selenium
pip3 install psutil

echo "Install stable firefox from mozilla repos"
snap remove firefox
install -d -m 0755 /etc/apt/keyrings
wget -q https://packages.mozilla.org/apt/repo-signing-key.gpg -O- | sudo tee /etc/apt/keyrings/packages.mozilla.org.asc > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/packages.mozilla.org.asc] https://packages.mozilla.org/apt mozilla main" | sudo tee -a /etc/apt/sources.list.d/mozilla.list > /dev/null
echo '
Package: *
Pin: origin packages.mozilla.org
Pin-Priority: 1000
' | sudo tee /etc/apt/preferences.d/mozilla
sudo apt update && sudo apt install firefox

echo "Capturing script dependencies : geckodriver"
wget https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-linux64.tar.gz
tar -xzvf geckodriver-v0.34.0-linux64.tar.gz
rm geckodriver-v0.34.0-linux64.tar.gz
chmod +x geckodriver
mv geckodriver /usr/local/bin


echo "Install systemd services"
cp wfp-proxy1.service /etc/systemd/system/
cp wfp-requester1.service /etc/systemd/system/
systemctl daemon-reload