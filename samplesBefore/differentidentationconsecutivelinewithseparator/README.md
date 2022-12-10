the line with the print statement has '(' and ')', so we expect it to create 2 different diffs and no merge conflicts due to the separator doing something like
  
before:
x = ""
print("base")
  
after:
x = ""
print
$$$$$
(
$$$$$
"base"
$$$$$
)
$$$$$
  
separating the "base" string diff from the x assignment diff
