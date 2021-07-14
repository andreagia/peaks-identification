##Automated NMR Peak Assignment

Example of simple peak assignement prediction
```
estimate_shifts.py --input1 free.csv --input2 with_ligand.csv --output est_shift.csv 
```



Peak assignement prediction with known target peaks
```
assign.py --input1 free.csv --input2 with_ligand.csv --output est_shift.csv --labeled 
```