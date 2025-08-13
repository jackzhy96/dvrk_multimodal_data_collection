#!/bin/bash

#File: tree-md

target="${1:-..}"

tree=$(tree -tf --noreport -I '*~' --charset ascii "$target" |
       sed -e 's/| \+/  /g' -e 's/[|`]-\+/ */g' -e 's:\(* \)\(\(.*/\)\([^/]\+\)\):\1[\4](\2):g')

printf "# Project tree\n\n${tree}"