since the 2 consecutive lines that actually has diffs don't have separators on tem, the current csdiff won't separate them and it will generates only one conflict block, instead of solving it.
 
This is a false positive we want to solve by using the identation information to separate the diffs
