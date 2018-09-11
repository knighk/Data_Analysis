PROC FREQ data=save.train;

TABLES OIR;

FORMAT OIR OIR.;   *** The first vage is the variable name. The second vage with the period after it refers to the format name to be applied ****

Run;
