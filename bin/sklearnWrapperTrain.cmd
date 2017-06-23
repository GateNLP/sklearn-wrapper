@ECHO OFF

SET ROOTDIR=%SKLEARN_WRAPPER_HOME%
SET data=%1
shift
SET model=%1
shift
SET algorithm=%1
shift

: create var with remaining arguments
set r=%1
:loop
shift
if [%1]==[] goto done
set r=%r% %1
goto loop
:done

:: TODO: find out how to run python
python %ROOTDIR%\python\sklearnTrain.py %data% %model% %algorithm% %r%
