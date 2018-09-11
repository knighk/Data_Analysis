Data save.train2;
set save.train;

if score > 652 and score <= 678   then score1 = 1;  else score1 = 0; 
if score > 678 and score <= 693   then score2 = 1;  else score2 = 0; 
if score > 693 and score <= 706   then score3 = 1;  else score3 = 0; 
if score > 706 and score <= 717   then score4 = 1;  else score4 = 0; 
if score > 717 and score <= 727   then score5 = 1;  else score5 = 0; 
if score > 727 and score <= 737   then score6 = 1;  else score6 = 0; 
if score > 737 and score <= 746   then score7 = 1;  else score7 = 0; 
if score > 746 and score <= 753   then score8 = 1;  else score8 = 0; 
if score > 753 and score <= 760   then score9 = 1;  else score9 = 0; 
if score > 760 and score <= 766   then score10 = 1;  else score10 = 0; 
if score > 766 and score <= 772   then score11 = 1;  else score11 = 0; 
if score > 772 and score <= 777   then score12 = 1;  else score12 = 0; 
if score > 777 and score <= 782   then score13 = 1;  else score13 = 0; 
if score > 782 and score <= 787   then score14 = 1;  else score14 = 0; 
if score > 787 and score <= 792   then score15 = 1;  else score15 = 0; 
if score > 792 and score <= 797   then score16 = 1;  else score16 = 0; 
if score > 797 and score <= 802   then score17 = 1;  else score17 = 0; 
if score > 802 and score <= 808   then score18 = 1;  else score18 = 0; 
if score > 808   then score19 = 1;  else score19 = 0; 

if first_time_homebuyer = 'Y'   then first_time_homebuyer1 = 1;  else first_time_homebuyer1 = 0; 

if insurance > 5 and insurance <= 9   then insurance1 = 1;  else insurance1 = 0; 
if insurance > 9 and insurance <= 12   then insurance2 = 1;  else insurance2 = 0; 
if insurance > 12 and insurance <= 16   then insurance3 = 1;  else insurance3 = 0; 
if insurance > 16 and insurance <= 17   then insurance4 = 1;  else insurance4 = 0; 
if insurance > 17 and insurance <= 22   then insurance5 = 1;  else insurance5 = 0; 
if insurance > 22 and insurance <= 25   then insurance6 = 1;  else insurance6 = 0; 
if insurance > 25 and insurance <= 27   then insurance7 = 1;  else insurance7 = 0; 
if insurance > 27 and insurance <= 30   then insurance8 = 1;  else insurance8 = 0; 
if insurance > 30   then insurance9 = 1;  else insurance9 = 0; 

if number_units > 1 and number_units <= 2   then number_units1 = 1;  else number_units1 = 0; 
if number_units > 2 and number_units <= 3   then number_units2 = 1;  else number_units2 = 0; 
if number_units > 3 and number_units <= 4   then number_units3 = 1;  else number_units3 = 0; 

if occupancy_status = 'I'   then occupancy_status1 = 1;  else occupancy_status1 = 0; 
if occupancy_status = 'S'   then occupancy_status2 = 1;  else occupancy_status2 = 0; 

if DTI > 18 and DTI <= 21   then DTI1 = 1;  else DTI1 = 0; 
if DTI > 21 and DTI <= 25   then DTI2 = 1;  else DTI2 = 0; 
if DTI > 25 and DTI <= 34   then DTI3 = 1;  else DTI3 = 0; 
if DTI > 34 and DTI <= 36   then DTI4 = 1;  else DTI4 = 0; 
if DTI > 36 and DTI <= 38   then DTI5 = 1;  else DTI5 = 0; 
if DTI > 38 and DTI <= 40   then DTI6 = 1;  else DTI6 = 0; 
if DTI > 40 and DTI <= 42   then DTI7 = 1;  else DTI7 = 0; 
if DTI > 42 and DTI <= 44   then DTI8 = 1;  else DTI8 = 0; 
if DTI > 44   then DTI9 = 1;  else DTI9 = 0; 

if UPB > 132000 and UPB <= 146000   then UPB1 = 1;  else UPB1 = 0; 
if UPB > 146000 and UPB <= 159000   then UPB2 = 1;  else UPB2 = 0; 
if UPB > 159000 and UPB <= 172000   then UPB3 = 1;  else UPB3 = 0; 
if UPB > 172000 and UPB <= 186000   then UPB4 = 1;  else UPB4 = 0; 
if UPB > 186000 and UPB <= 200000   then UPB5 = 1;  else UPB5 = 0; 
if UPB > 200000 and UPB <= 216000   then UPB6 = 1;  else UPB6 = 0; 
if UPB > 216000 and UPB <= 233000   then UPB7 = 1;  else UPB7 = 0; 
if UPB > 233000 and UPB <= 251000   then UPB8 = 1;  else UPB8 = 0; 
if UPB > 251000 and UPB <= 273000   then UPB9 = 1;  else UPB9 = 0; 
if UPB > 273000 and UPB <= 298000   then UPB10 = 1;  else UPB10 = 0; 
if UPB > 298000 and UPB <= 325000   then UPB11 = 1;  else UPB11 = 0; 
if UPB > 325000 and UPB <= 359000   then UPB12 = 1;  else UPB12 = 0; 
if UPB > 359000 and UPB <= 400000   then UPB13 = 1;  else UPB13 = 0; 
if UPB > 400000   then UPB14 = 1;  else UPB14 = 0; 

if LTV > 56 and LTV <= 60   then LTV1 = 1;  else LTV1 = 0; 
if LTV > 60 and LTV <= 70   then LTV2 = 1;  else LTV2 = 0; 
if LTV > 70 and LTV <= 76   then LTV3 = 1;  else LTV3 = 0; 
if LTV > 76 and LTV <= 89   then LTV4 = 1;  else LTV4 = 0; 
if LTV > 89   then LTV5 = 1;  else LTV5 = 0; 

if OIR > 4.876 and OIR <= 5   then OIR1 = 1;  else OIR1 = 0; 
if OIR > 5.001 and OIR <= 5.25   then OIR2 = 1;  else OIR2 = 0; 
if OIR > 5.251 and OIR <= 5.375   then OIR3 = 1;  else OIR3 = 0; 
if OIR > 5.376 and OIR <= 5.5   then OIR4 = 1;  else OIR4 = 0; 
if OIR > 5.501 and OIR <= 5.75   then OIR5 = 1;  else OIR5 = 0; 
if OIR > 5.751 and OIR <= 6   then OIR6 = 1;  else OIR6 = 0; 
if OIR > 6.001 and OIR <= 6.125   then OIR7 = 1;  else OIR7 = 0; 
if OIR > 6.126 and OIR <= 6.25   then OIR8 = 1;  else OIR8 = 0; 
if OIR > 6.251   then OIR9 = 1;  else OIR9 = 0; 

if channel = 'B'   then channel1 = 1;  else channel1 = 0; 
if channel = 'C'   then channel2 = 1;  else channel2 = 0; 
if channel = 'T'   then channel3 = 1;  else channel3 = 0; 

if PPM = 'Y'   then PPM1 = 1;  else PPM1 = 0; 

if property_state = 'AL'    then property_state1 = 1;  else property_state1 = 0;
if property_state = 'AR'    then property_state2 = 1;  else property_state2 = 0;
if property_state = 'AZ'    then property_state3 = 1;  else property_state3 = 0;
if property_state = 'CA'    then property_state4 = 1;  else property_state4 = 0;
if property_state = 'CO'    then property_state5 = 1;  else property_state5 = 0;
if property_state = 'CT'    then property_state6 = 1;  else property_state6 = 0;
if property_state = 'DC'    then property_state7 = 1;  else property_state7 = 0;
if property_state = 'DE'    then property_state8 = 1;  else property_state8 = 0;
if property_state = 'FL'    then property_state9 = 1;  else property_state9 = 0;
if property_state = 'GA'    then property_state10 = 1;  else property_state10 = 0;
if property_state = 'GU'    then property_state11 = 1;  else property_state11 = 0;
if property_state = 'HI'    then property_state12 = 1;  else property_state12 = 0;
if property_state = 'IA'    then property_state13 = 1;  else property_state13 = 0;
if property_state = 'ID'    then property_state14 = 1;  else property_state14 = 0;
if property_state = 'IL'    then property_state15 = 1;  else property_state15 = 0;
if property_state = 'IN'    then property_state16 = 1;  else property_state16 = 0;
if property_state = 'KS'    then property_state17 = 1;  else property_state17 = 0;
if property_state = 'KY'    then property_state18 = 1;  else property_state18 = 0;
if property_state = 'LA'    then property_state19 = 1;  else property_state19 = 0;
if property_state = 'MA'    then property_state20 = 1;  else property_state20 = 0;
if property_state = 'MD'    then property_state21 = 1;  else property_state21 = 0;
if property_state = 'ME'    then property_state22 = 1;  else property_state22 = 0;
if property_state = 'MI'    then property_state23 = 1;  else property_state23 = 0;
if property_state = 'MN'    then property_state24 = 1;  else property_state24 = 0;
if property_state = 'MO'    then property_state25 = 1;  else property_state25 = 0;
if property_state = 'MS'    then property_state26 = 1;  else property_state26 = 0;
if property_state = 'MT'    then property_state27 = 1;  else property_state27 = 0;
if property_state = 'NC'    then property_state28 = 1;  else property_state28 = 0;
if property_state = 'ND'    then property_state29 = 1;  else property_state29 = 0;
if property_state = 'NE'    then property_state30 = 1;  else property_state30 = 0;
if property_state = 'NH'    then property_state31 = 1;  else property_state31 = 0;
if property_state = 'NJ'    then property_state32 = 1;  else property_state32 = 0;
if property_state = 'NM'    then property_state33 = 1;  else property_state33 = 0;
if property_state = 'NV'    then property_state34 = 1;  else property_state34 = 0;
if property_state = 'NY'    then property_state35 = 1;  else property_state35 = 0;
if property_state = 'OH'    then property_state36 = 1;  else property_state36 = 0;
if property_state = 'OK'    then property_state37 = 1;  else property_state37 = 0;
if property_state = 'OR'    then property_state38 = 1;  else property_state38 = 0;
if property_state = 'PA'    then property_state39 = 1;  else property_state39 = 0;
if property_state = 'PR'    then property_state40 = 1;  else property_state40 = 0;
if property_state = 'RI'    then property_state41 = 1;  else property_state41 = 0;
if property_state = 'SC'    then property_state42 = 1;  else property_state42 = 0;
if property_state = 'SD'    then property_state43 = 1;  else property_state43 = 0;
if property_state = 'TN'    then property_state44 = 1;  else property_state44 = 0;
if property_state = 'TX'    then property_state45 = 1;  else property_state45 = 0;
if property_state = 'UT'    then property_state46 = 1;  else property_state46 = 0;
if property_state = 'VA'    then property_state47 = 1;  else property_state47 = 0;
if property_state = 'VI'    then property_state48 = 1;  else property_state48 = 0;
if property_state = 'VT'    then property_state49 = 1;  else property_state49 = 0;
if property_state = 'WA'    then property_state50 = 1;  else property_state50 = 0;
if property_state = 'WI'    then property_state51 = 1;  else property_state51 = 0;
if property_state = 'WV'    then property_state52 = 1;  else property_state52 = 0;
if property_state = 'WY'    then property_state53 = 1;  else property_state53 = 0;

if property_type = 'CP'    then property_type1 = 1;  else property_type1 = 0;
if property_type = 'LH'    then property_type2 = 1;  else property_type2 = 0;
if property_type = 'MH'    then property_type3 = 1;  else property_type3 = 0;
if property_type = 'PU'    then property_type4 = 1;  else property_type4 = 0;
if property_type = 'SF'    then property_type5 = 1;  else property_type5 = 0;

if loan_purpose = 'N'    then loan_purpose1 = 1;  else loan_purpose1 = 0;
if loan_purpose = 'P'    then loan_purpose2 = 1;  else loan_purpose2 = 0;

if orig_loan_term > 300 and orig_loan_term <= 359   then orig_loan_term1 = 1;  else orig_loan_term1 = 0; 
if orig_loan_term > 360   then orig_loan_term2 = 1;  else orig_loan_term2 = 0; 

if number_borrowers = 2   then number_borrowers1 = 1;  else number_borrowers1 = 0; 

if seller = 'BANKOFAMERICA,NA'    then seller1 = 1;  else seller1 = 0;
if seller = 'BRANCHBANKING&TRUSTC'    then seller2 = 1;  else seller2 = 0;
if seller = 'CHASEHOMEFINANCELLC'    then seller3 = 1;  else seller3 = 0;
if seller = 'CITIMORTGAGE,INC'    then seller4 = 1;  else seller4 = 0;
if seller = 'COUNTRYWIDE'    then seller5 = 1;  else seller5 = 0;
if seller = 'FIFTHTHIRDBANK'    then seller6 = 1;  else seller6 = 0;
if seller = 'FIRSTHORIZONHOMELOAN'    then seller7 = 1;  else seller7 = 0;
if seller = 'FLAGSTARCAPITALMARKE'    then seller8 = 1;  else seller8 = 0;
if seller = 'GMACMORTGAGE,LLC'    then seller9 = 1;  else seller9 = 0;
if seller = 'METLIFEHOMELOANS,ADI'    then seller10 = 1;  else seller10 = 0;
if seller = 'NATLCITYBANK'    then seller11 = 1;  else seller11 = 0;
if seller = 'NATLCITYMTGECO'    then seller12 = 1;  else seller12 = 0;
if seller = 'Other sellers'    then seller13 = 1;  else seller13 = 0;
if seller = 'PHHMTGECORP'    then seller14 = 1;  else seller14 = 0;
if seller = 'PROVIDENTFUNDINGASSO'    then seller15 = 1;  else seller15 = 0;
if seller = 'REGIONSBANKDBAREGION'    then seller16 = 1;  else seller16 = 0;
if seller = 'SUNTRUSTMORTGAGE,INC'    then seller17 = 1;  else seller17 = 0;
if seller = 'TAYLOR,BEAN&WHITAKER'    then seller18 = 1;  else seller18 = 0;
if seller = 'USBANKNA'    then seller19 = 1;  else seller19 = 0;
if seller = 'WACHOVIAMORTGAGE,FSB'    then seller20 = 1;  else seller20 = 0;
if seller = 'WASHINGTONMUTUALBANK'    then seller21 = 1;  else seller21 = 0;
if seller = 'WELLSFARGOBANK,NA'    then seller22 = 1;  else seller22 = 0;

if servicer = 'AMTRUSTBANK'    then servicer1 = 1;  else servicer1 = 0;
if servicer = 'BANKOFAMERICA,NA'    then servicer2 = 1;  else servicer2 = 0;
if servicer = 'BRANCHBANKING&TRUSTC'    then servicer3 = 1;  else servicer3 = 0;
if servicer = 'CENLARFSB'    then servicer4 = 1;  else servicer4 = 0;
if servicer = 'CENTRALMTGECO'    then servicer5 = 1;  else servicer5 = 0;
if servicer = 'CITIMORTGAGE,INC'    then servicer6 = 1;  else servicer6 = 0;
if servicer = 'COUNTRYWIDE'    then servicer7 = 1;  else servicer7 = 0;
if servicer = 'EVERBANK'    then servicer8 = 1;  else servicer8 = 0;
if servicer = 'FIFTHTHIRDBANK'    then servicer9 = 1;  else servicer9 = 0;
if servicer = 'FLAGSTARCAPITALMARKE'    then servicer10 = 1;  else servicer10 = 0;
if servicer = 'GMACMORTGAGE,LLC'    then servicer11 = 1;  else servicer11 = 0;
if servicer = 'JPMORGANCHASEBANK,NA'    then servicer12 = 1;  else servicer12 = 0;
if servicer = 'METLIFEHOMELOANS,ADI'    then servicer13 = 1;  else servicer13 = 0;
if servicer = 'NATIONSTARMORTGAGE,L'    then servicer14 = 1;  else servicer14 = 0;
if servicer = 'OCWENLOANSERVICING,L'    then servicer15 = 1;  else servicer15 = 0;
if servicer = 'Other servicers'    then servicer16 = 1;  else servicer16 = 0;
if servicer = 'PHHMTGECORP'    then servicer17 = 1;  else servicer17 = 0;
if servicer = 'PNCBANK,NATL'    then servicer18 = 1;  else servicer18 = 0;
if servicer = 'PROVIDENTFUNDINGASSO'    then servicer19 = 1;  else servicer19 = 0;
if servicer = 'REGIONSBANKDBAREGION'    then servicer20 = 1;  else servicer20 = 0;
if servicer = 'SUNTRUSTMORTGAGE,INC'    then servicer21 = 1;  else servicer21 = 0;
if servicer = 'TAYLOR,BEAN&WHITAKER'    then servicer22 = 1;  else servicer22 = 0;
if servicer = 'USBANKNA'    then servicer23 = 1;  else servicer23 = 0;
if servicer = 'WELLSFARGOBANK,NA'    then servicer24 = 1;  else servicer24 = 0;

if def_in_24_months = 'TRUE'    then def_in_24_months1 = 1;  else def_in_24_months1 = 0;

run;
quit;
