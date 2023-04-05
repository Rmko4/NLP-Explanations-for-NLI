$Path = "~/models/checkpoints/"
# Take all files in the directory and add the extension .ckpt
Get-ChildItem $Path | Rename-Item -NewName { $_.Name + ".ckpt" } -Confirm