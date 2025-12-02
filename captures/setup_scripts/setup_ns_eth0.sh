#!/bin/sh

# NOTE: it is best to run the commands one by one, some of them are blocking anyway

ip netns add netns1
ip netns exec netns1 ip link list
ip netns exec netns1 ip link set dev lo up
#sudo ip netns exec netns1 ping 127.0.0.1 # test

ip link add veth1 type veth peer name veth0
ip link set veth0 netns netns1

ip netns exec netns1 ip addr add 10.1.1.1/24 dev veth0
ip netns exec netns1 ip link set dev veth0 up

ip addr add 10.1.1.2/24 dev veth1
ip link set dev veth1 up

#ping 10.1.1.1 # test
#sudo ip netns exec netns1 ping 10.1.1.2 # test

sysctl net.ipv4.ip_forward=1
iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE # maybe change interface !

ip netns exec netns1 ip ro add default via 10.1.1.2

ufw route allow out on veth1 in on eth0 from any to any
ufw route allow in on veth1 out on eth0 from any to any
ufw reload
sh -c "echo 'nameserver 8.8.8.8' > /etc/resolv.conf"
#sudo ip netns exec netns1 ping 1.1.1.1 # test

# References:
# https://github.com/Lekensteyn/netns
# http://sgros.blogspot.com/2017/04/how-to-run-firefox-in-separate-network.html
# https://lwn.net/Articles/580893/
