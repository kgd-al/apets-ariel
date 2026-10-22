# Re-Enable bluetooth
Someone thought we do not need it. They are wrong (are they?), this fixes it:

```
sudo sed -i 's/dtoverlay=disable-bt/dtoverlay=pi3-miniuart-bt/' /boot/firmware/config.txt
```

Then make sure we have required packages

```
sudo apt install -y bluetooth pi-bluetooth bluez
sudo systemctl enable --now bluetooth
sudo usermod -a -G bluetooth $USER
sudo reboot
```

Pair the controller:

```
bluetoothctl
agent on
default-agent
power on
scan on
```

Hold Share/Create (upper left) + PS until the light flashes rapidly, then:

```
pair <MAC_ADDRESS>
connect <MAC_ADDRESS>
trust <MAC_ADDRESS>
scan off
```

where MAC_ADDRESS is the name you just see pop up when the controller started pairing mode.
Trust makes it auto-reconnect on power-on with no further steps.
