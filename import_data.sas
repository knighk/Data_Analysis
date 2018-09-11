PROC IMPORT OUT= WORK.raw_data 
            DATAFILE= "C:\Users\yyang42\Desktop\kkCourse\final\DATA\D2_v
1.csv" 
            DBMS=CSV REPLACE;
     GETNAMES=YES;
     DATAROW=2; 
RUN;
