DATA save.train save.valid;

Set Work.Raw_data;

 

Random1 = ranuni(14380132);       ***use your own 7-8 digit seed number****;

IF Random1 < 0.7 THEN output save.train;

      ELSE output save.valid;

Run;
