@ECHO OFF


SET ROOTDIR=%1
SET MODEL=%2
shift
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
