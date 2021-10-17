#!/bin/bash
############################################################
# Help                                                     #
############################################################
helpText()
{
   # Display Help
   echo "Syntax: csdiff MYFILE OLDFILE YOURFILE [-options]"
   echo "options:"
   echo "-h                    Print this Help."
   echo "-s \"<separators>\"     Specify the list of separators, e.g. \"{ } ( ) ; ,\""
   echo
}

############################################################
# sed options used                                         #
############################################################
## Descriptions extracted from sed man page: https://linux.die.net/man/1/sed
# -e       - add the script to the commands to be executed
# :a       - Defines a label 'a'
# N        - Append the next line of input into the pattern space.
# $        - Match the last line.
# !        - After the address (or address-range), and before the command, a ! may be inserted, which specifies that the command shall only be executed if the address (or address-range) does not match.
# b[label] - Branch to [label]; if [label] is omitted, branch to end of script.
# s/       - Form: [s/regexp/replacement/] - Attempt to match regexp against the pattern space. If successful, replace that portion matched with replacement.

############################################################
############################################################
#######                   CSDiff                     #######
############################################################
############################################################
############################################################
# Process the input options. Add options as needed.        #
############################################################
while getopts s:h option
do
  case $option in
    h) # display Help
      helpText
      exit 0
      ;;
    s)
      set -f                 # turn off filename expansion
      separators=($OPTARG)   # variable is unquoted
      set +f                 # turn it back on
      ;;
   esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

############################################################
# Main logic                                               #
############################################################

files=("$@")
myFile=${files[0]}
oldFile=${files[1]}
yourFile=${files[2]}

sedCommandMyFile=""
sedCommandOldFile=""
sedCommandYourFile=""

# Dynamically builds the sed command pipeline based on the number of synctatic separators provided
for separator in "${separators[@]}";
  do
    # Build the base substitution script to be passed to the sed command
    sedScript="s/$separator/\n\$\$\$\$\$\$\$$separator\n\$\$\$\$\$\$\$/g"

    # When the separator is the first one in the array of separators, call sed with the substitution script and with the file
    # When the separator is the last one in the array of separators, call the final sed with the substitution script (piping with the previous call) and output the result to a temp file
    # When none of the above, call sed with the substitution script, piping with the previous call.
    if [[ $separator = ${separators[0]} ]]
    then
      sedCommandMyFile+="sed '${sedScript}' ${myFile}"
      sedCommandOldFile+="sed '${sedScript}' ${oldFile}"
      sedCommandYourFile+="sed '${sedScript}' ${yourFile}"
    elif [[ $separator = ${separators[-1]} ]]
    then
      sedCommandMyFile+=" | sed '${sedScript}' > ${myFile}_temp"
      sedCommandOldFile+=" | sed '${sedScript}' > ${oldFile}_temp"
      sedCommandYourFile+=" | sed '${sedScript}' > ${yourFile}_temp"
    else
      sedCommandMyFile+=" | sed '${sedScript}'"
      sedCommandOldFile+=" | sed '${sedScript}'"
      sedCommandYourFile+=" | sed '${sedScript}'"
    fi
  done

# Perform the tokenization of the input files based on the provided separators
eval ${sedCommandMyFile}
eval ${sedCommandOldFile}
eval ${sedCommandYourFile}

# Runs diff3 against the tokenized inputs, generating a tokenized merged file
diff3 -m "${myFile}_temp" "${oldFile}_temp" "${yourFile}_temp" > mid_merged

# Removes the tokenized input files
rm "${myFile}_temp"
rm "${oldFile}_temp"
rm "${yourFile}_temp"

# Removes the tokens from the merged file, generating the final merged file
sed ':a;N;$!ba;s/\n\$\$\$\$\$\$\$//g' mid_merged > merged

# Removes the tokenized merged file
rm mid_merged

# Get the names of left/base/right files
ESCAPED_LEFT=$(printf '%s\n' "${myFile}" | sed -e 's/[\/&]/\\&/g')
ESCAPED_BASE=$(printf '%s\n' "${oldFile}" | sed -e 's/[\/&]/\\&/g')
ESCAPED_RIGHT=$(printf '%s\n' "${yourFile}" | sed -e 's/[\/&]/\\&/g')

ESCAPED_TEMP_LEFT=$(printf '%s\n' "${myFile}_temp" | sed -e 's/[\/&]/\\&/g')
ESCAPED_TEMP_BASE=$(printf '%s\n' "${oldFile}_temp" | sed -e 's/[\/&]/\\&/g')
ESCAPED_TEMP_RIGHT=$(printf '%s\n' "${yourFile}_temp" | sed -e 's/[\/&]/\\&/g')

# Add merge conflict annotations to the merged file.
# This will be the output printed  after the execution of this script.
sed "s/\(<<<<<<< $ESCAPED_TEMP_LEFT\)\(.\+\)/\1\n\2/" merged | sed "s/\(||||||| $ESCAPED_TEMP_BASE\)\(.\+\)/\1\n\2/" | sed "s/\(=======\)\(.\+\)/\1\n\2/" | sed "s/\(>>>>>>> $ESCAPED_TEMP_RIGHT\)\(.\+\)/\1\n\2/" | sed "s/$ESCAPED_TEMP_LEFT/$ESCAPED_LEFT/g" | sed "s/$ESCAPED_TEMP_BASE/$ESCAPED_BASE/g" | sed "s/$ESCAPED_TEMP_RIGHT/$ESCAPED_RIGHT/g"

# Remove the merged file
rm merged
