
BINDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPDIR=`dirname ${BINDIR}`

checkPyModule()
{
    PACKAGE=$1
    VERSION=$(python -c "$2" 2>/dev/null )

    if [ $? -ne 0 ]
    then
        printf "%12s (%b)\n" "${PACKAGE}" "\e[31mnot found\e[0m"
    else
        printf "%12s (%b)\n" "${PACKAGE}" "\e[32mfound ${VERSION}\e[0m"
    fi
}

if [ ".$QUIET" == "." ]
then
    printf "\n"
    printf "\e[30m\e[107mIMPORTANT\e[0m: make sure you __source__ this script\n"
    printf "rather than execute it:\n"
    printf "\n"
    printf "\e[1m    yourprompt$ . \$FD_PATH/setup_bash.sh\e[0m\n"
    printf "\n"
    printf "otherwise the paths will not be correctly set.  If you'd like to\n"
    printf "silence this annoying message in the future, source the script\n"
    printf "like this:"
    printf "\n"
    printf "\e[1m    yourprompt$ QUIET=1 . \$FD_PATH/setup_bash.sh\e[0m\n"
    printf "\n"

    printf "Python package dependency check:\n"
    for PACKAGE in numpy scipy numexpr matplotlib
    do
        checkPyModule $PACKAGE "import ${PACKAGE}; print ${PACKAGE}.__version__; exit()"
    done
    checkPyModule "ROOT" "import ROOT; print ROOT.gROOT.GetVersion(); exit()"

    printf "\n"
    printf "ROOT is __optional__, and is required only to import Tree objects from\n"
    printf "  .root files.  If ROOT is not installed, you can still import .csv\n"
    printf "  files."
    printf "\n"
fi

[[ ":$PATH:"       != *${BINDIR}* ]] && export       PATH="${BINDIR}:${PATH}"
[[ ":$PYTHONPATH:" != *${TOPDIR}* ]] && export PYTHONPATH="${TOPDIR}:${PYTHONPATH}"

export FD_DIR=${TOPDIR}
