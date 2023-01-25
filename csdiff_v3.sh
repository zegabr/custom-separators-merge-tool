#!/bin/bash
############################################################
# Help                                                     #
############################################################
helpText()
{
   # Display Help
   echo "Syntax: csdiff [-options] MYFILE OLDFILE YOURFILE"
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

parameters=("$@")
myFile=${parameters[0]}
oldFile=${parameters[1]}
yourFile=${parameters[2]}
parentFolder=$(echo "${myFile%/*}")
myFileBaseName="$(basename "${myFile}")"
fileExt=$([[ "$myFileBaseName" = *.* ]] && echo ".${myFileBaseName##*.}" || echo '')

sedCommandMyFile=""
sedCommandOldFile=""
sedCommandYourFile=""

myTempFile="${myFile}_temp${fileExt}"
oldTempFile="${oldFile}_temp${fileExt}"
yourTempFile="${yourFile}_temp${fileExt}"

# Dynamically builds the sed command pipeline based on the number of synctatic separators provided
for separator in "${separators[@]}";
  do
    # Treat some specific symbols that need to be escaped before including them into the regex
    escapedSeparator=$separator
    if [[ $separator = '\' || $separator = '[' || $separator = ']' || $separator = '+' || $separator = '.' || $separator = '*' || $separator = '?' || $separator = '^' || $separator = '$' ]]
    then
      escapedSeparator="\\${separator}"
    fi

    # Build the base substitution script to be passed to the sed command
    sedScript="s/$escapedSeparator/\n\$\$\$\$\$\$\$$escapedSeparator\n\$\$\$\$\$\$\$/g"

    # When the separator is the first one in the array of separators, call sed with the substitution script and with the file
    # When the separator is the last one in the array of separators, call the final sed with the substitution script (piping with the previous call) and output the result to a temp file
    # When none of the above, call sed with the substitution script, piping with the previous call.


    if [[ $separator = ${separators[0]} ]]
    then
      if [[ ${#separators[@]} = 1 ]]
      then
        sedCommandMyFile+="sed '${sedScript}' ${myFile} > $myTempFile"
        sedCommandOldFile+="sed '${sedScript}' ${oldFile} > $oldTempFile"
        sedCommandYourFile+="sed '${sedScript}' ${yourFile}  > $yourTempFile"
      else
        sedCommandMyFile+="sed '${sedScript}' ${myFile}"
        sedCommandOldFile+="sed '${sedScript}' ${oldFile}"
        sedCommandYourFile+="sed '${sedScript}' ${yourFile}"
      fi
    elif [[ $separator = ${separators[-1]} ]]
    then
      sedCommandMyFile+=" | sed '${sedScript}' > $myTempFile"
      sedCommandOldFile+=" | sed '${sedScript}' > $oldTempFile"
      sedCommandYourFile+=" | sed '${sedScript}' > $yourTempFile"
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

# TODO: add if to check if files are python?
# run the script to consider identation
python3 csdiff_python.py < "$myTempFile" > myOut
python3 csdiff_python.py < "$oldTempFile" > oldOut
python3 csdiff_python.py < "$yourTempFile" > yourOut
# override the temporary files again
cat myOut > "$myTempFile"
cat oldOut > "$oldTempFile"
cat yourOut > "$yourTempFile"
rm myOut
rm oldOut
rm yourOut

# Runs diff3 against the tokenized inputs, generating a tokenized merged file
midMergedFile="${parentFolder}/mid_merged${fileExt}"
diff3 -m -E "$myTempFile" "$oldTempFile" "$yourTempFile" > $midMergedFile

# Removes the tokenized input files
rm "$myTempFile"
rm "$oldTempFile"
rm "$yourTempFile"

# Removes the tokens from the merged file, generating the final merged file
mergedFile="${parentFolder}/merged${fileExt}"
sed ':a;N;$!ba;s/\n\$\$\$\$\$\$\$//g' $midMergedFile > $mergedFile

# Removes the tokenized merged file
rm "$midMergedFile"

# Get the names of left/base/right files
ESCAPED_LEFT=$(printf '%s\n' "${myFile}" | sed -e 's/[\/&]/\\&/g')
ESCAPED_BASE=$(printf '%s\n' "${oldFile}" | sed -e 's/[\/&]/\\&/g')
ESCAPED_RIGHT=$(printf '%s\n' "${yourFile}" | sed -e 's/[\/&]/\\&/g')

ESCAPED_TEMP_LEFT=$(printf '%s\n' "$myTempFile" | sed -e 's/[\/&]/\\&/g')
ESCAPED_TEMP_BASE=$(printf '%s\n' "$oldTempFile" | sed -e 's/[\/&]/\\&/g')
ESCAPED_TEMP_RIGHT=$(printf '%s\n' "$yourTempFile" | sed -e 's/[\/&]/\\&/g')

# Fix the merged file line breaks that got messed up by the CSDiff algorithm.
# TODO: make this universal to other languages, here it will work with python
comment_string="#"
sed -e "/^$comment_string/!s/\(<<<<<<< $ESCAPED_TEMP_LEFT\)\(.\+\)/\1\n\2/" $mergedFile \
| sed -e "/^$comment_string/!s/\(<<<<<<< $ESCAPED_TEMP_BASE\)\(.\+\)/\1\n\2/" \
| sed -e "/^$comment_string/!s/\(<<<<<<< $ESCAPED_TEMP_RIGHT\)\(.\+\)/\1\n\2/" \
| sed -e "/^$comment_string/!s/\(||||||| $ESCAPED_TEMP_BASE\)\(.\+\)/\1\n\2/" \
| sed -e "/^$comment_string/!s/\(||||||| $ESCAPED_TEMP_LEFT\)\(.\+\)/\1\n\2/" \
| sed -e "/^$comment_string/!s/\(||||||| $ESCAPED_TEMP_RIGHT\)\(.\+\)/\1\n\2/" \
| sed -e "/^$comment_string/!s/\(>>>>>>> $ESCAPED_TEMP_RIGHT\)\(.\+\)/\1\n\2/" \
| sed -e "/^$comment_string/!s/\(>>>>>>> $ESCAPED_TEMP_LEFT\)\(.\+\)/\1\n\2/" \
| sed -e "/^$comment_string/!s/\(>>>>>>> $ESCAPED_TEMP_BASE\)\(.\+\)/\1\n\2/" \
| sed -e "/^$comment_string/!s/\(=======\)\(.\+\)/\1\n\2/" \
| sed -e "/^$comment_string/!s/$ESCAPED_TEMP_LEFT/$ESCAPED_LEFT/g" \
| sed -e "/^$comment_string/!s/$ESCAPED_TEMP_BASE/$ESCAPED_BASE/g" \
| sed -e "/^$comment_string/!s/$ESCAPED_TEMP_RIGHT/$ESCAPED_RIGHT/g" \
| sed -e "/^$comment_string/!s/=======/\n=======/" \
| sed -e "/^$comment_string/!s/>>>>>>>/\n>>>>>>>/" > "${parentFolder}/csdiff${fileExt}"
sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' "${parentFolder}/csdiff${fileExt}"

# Outputs two other files that will be useful for the study: one generated by the diff3 merge
# and another one generated by the 'git merge-file' command, using the diff3.
diff3 -E -m ${myFile} ${oldFile} ${yourFile} > "${parentFolder}/diff3${fileExt}"
git merge-file -p --diff3 ${myFile} ${oldFile} ${yourFile} > "${parentFolder}/git_merge${fileExt}"
# echo  "merge saved at: ${parentFolder}/csdiff${fileExt}"
# Remove the merged file, since we already saved it
rm "$mergedFile"
