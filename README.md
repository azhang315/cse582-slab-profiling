contains:
* collection scripts: `scripts`
* plotting scripts: `plotting`
* selection of graphs

usage:
* easiest to mount to VMs individually to test collection, etc.



```
Mount
Create a mount point:

bash
Copy code
mkdir -p /mnt/shared_folder
Mount the shared folder:

bash
Copy code
mount -t 9p -o trans=virtio shared_folder /mnt/shared_folder

Replace shared_folder with the mount_tag value you used in the QEMU script.

bash
Copy code
ls /mnt/shared_folder
```
