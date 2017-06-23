@ECHO OFF


SET ROOTDIR=%SKLEARN_WRAPPER_HOME%
SET MODEL=%1
shift

: create var with remaining arguments
set r=%1
:loop
shift
if [%1]==[] goto done
set r=%r% %1
goto loop
:done


:: TODO: how to figure out how to run python??
python %ROOTDIR%/python/sklearnApply.py %model% %r%
