### To run the kilo demos

For anyone wanting to run the hardware version of kilo (the diagonal spider robot with camera on the front edge), you will need:
- be one the same network as the robot
    - iotroam at the time of writing
- ssh connection to the robot
    - username is robohat, password is raspberry
    - potential ips were: 10.15.3.68 & 10.15.3.216
- The robot will make a sound when it is ready for ssh connection + a bip when you actually connect
- Once you are in, everything is located under `kgd` so cd there
- Two scripts are provided:
    - run_after_ball.sh
    - run_with_joystick.sh
    - These just wrap the generic `aapets.bin.run_hardware` module with sensible arguments

## If (when) things break

Many things can, and will break, here are a few things that could help:
- If a hinge breaks it is either:
    - the structural (plastic) part -> replace it
    - the screws -> also replace but you may need to remove the part still stuck in (requires plier)
    - the servo, harder to diagnose. The `--test-hinges` argument can help. Remove hinge from the faulting group sequentially and test with a new one before committing to changing anything
- If the software complains about unconnected pins being referenced:
    - First try to restart
    - If not enough, try the `SerTest.py` script from the robohat lib. It sometimes helped for some reason
- If the robot made a happy sound but you cannot connect:
    - ip may have changed, best way to get is to unscrew the raspberry and directly connect to it (with the battery on, just plug the hdmi cable)
- If the controller does not want to connect:
    - if bluetooth, that is a known pain. If you can use gui otherwise `/docs/PS_bluetooth_control.md` as some information
- If anything else: sorry, out of luck
