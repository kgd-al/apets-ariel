## Switching Wi-Fi networks from the CLI

Bookworm uses NetworkManager (`nmcli`), not `wpa_supplicant` directly.

```bash
nmcli con show                        # saved networks
sudo nmcli con up "SSID_NAME"         # connect to a known one
sudo nmcli dev wifi connect "SSID" password "yourpassword"   # new one
nmcli dev wifi list                   # scan results
```

**If `nmcli dev wifi list` only shows the currently connected network:** the scan result is cached and can go stale after boot.

```bash
sudo nmcli dev wifi rescan; sleep 3; nmcli dev wifi list
```

(Rescans are rate-limited to roughly every 30s — a second rescan right after the first silently no-ops.) To bypass `nmcli` entirely as a sanity check: `sudo iw dev wlan0 scan | grep SSID`.

**If connecting a new network fails with:**

```
Error: 802-11-wireless-security.key-mgmt: property is missing.
```

This is a known NetworkManager regression where the "quick connect" shorthand stopped auto-filling `key-mgmt`. Fix, in order of reliability:

```bash
# 1. delete any stale profile with the same name, then retry
nmcli connection delete "SSID_NAME"
sudo nmcli dev wifi connect "SSID_NAME" password "yourpassword"

# 2. or use --ask for an interactive prompt
sudo nmcli --ask dev wifi connect "SSID_NAME"

# 3. or build the profile explicitly (most reliable, scriptable)
sudo nmcli connection add type wifi con-name "SSID_NAME" ifname wlan0 \
  ssid "SSID_NAME" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "yourpassword"
sudo nmcli connection up "SSID_NAME"
```

(Use `wifi-sec.key-mgmt sae` instead of `wpa-psk` for a WPA3 network.)
