PROC FREQ data=save.train;

TABLES OIR*def_in_24_months / norow nopercent;

FORMAT OIR OIR.;  

Run;
