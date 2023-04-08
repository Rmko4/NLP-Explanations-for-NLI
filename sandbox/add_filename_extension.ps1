$Path = "~/models/checkpoints/"
Get-ChildItem $Path | Rename-Item -NewName { $_.Name + ".ckpt" } -Confirm