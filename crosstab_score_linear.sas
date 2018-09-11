ODS html file="C:\Users\yyang42\Desktop\kkCourse\final\OUTPUT\crosstab_linear.html";
proc freq data=save.scrtrain;
tables bgscore*def_in_24_months1 / norow nopercent;
format bgscore bgscore. ;
run;
ODS html close;
